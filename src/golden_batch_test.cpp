// ============================================================================
// A LYING METRIC MIMICS EVERY OTHER FAILURE MODE.
//
// Every major defect this project has shipped — the FFN that never trained on
// GPU, the layers-only checkpoints, the RMSNorm restored as LayerNorm, the
// OOV ids gathered from out-of-bounds GPU memory, the engine that cleared F32
// weights it could not recover — produced plausible outputs while measuring
// something other than the model. Forward-output spot checks cannot catch
// that class of bug: a backend can produce sane activations while silently
// updating (or serving) a different parameter copy. The strongest test is
// differential:  same initial state + same batch + one step -> same state.
//
// This tool implements the golden-batch legs that run anywhere (CPU build):
//   save    : load ckpt -> forward golden prompts -> save -> reload ->
//             re-forward -> logits must be IDENTICAL (max |delta| reported)
//   dump    : write per-prompt last-position logits to TSV for differential
//             comparison against the inference engine on the same GGUF
//             (tinyllama.cpp: TINYLLAMA_LOGITS_DUMP=path, then
//             scripts/compare_golden_logits.py)
// CUDA-only legs (CPU-vs-GPU activations, gradient equality, one-step state
// equality) ride the trainer's [TRACE]/[SUBOP]/XCHECK instrumentation on a
// GPU host; this tool is the deploy-side anchor.
// ============================================================================

#include <cstdio>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "../include/transformer.hpp"
#include "../include/model_saver.hpp"
#include "../include/tiktoken_tokenizer.hpp"

namespace {

struct Args {
    std::string ckpt;
    std::string vocab_list;  // id-ordered word list (extract_gguf_vocab.py) —
                             // the ONLY cross-machine-safe vocab source
    std::vector<std::string> vocab_files;  // built IN ORDER, must mirror training
    std::vector<std::string> prompts;
    std::string dump_path;
    bool mid = false;
    bool tiny = false;
    size_t seq_len = 256;
};

TransformerConfig make_config(const Args& a, size_t tok_vocab) {
    // Mirrors train_wikitext.cpp preset logic (CPU build branch for default).
    TransformerConfig c;
    if (a.mid) {
        c.vocab_size = std::min(tok_vocab, static_cast<size_t>(5000));
        c.hidden_size = 256; c.num_heads = 8; c.num_layers = 4;
        c.intermediate_size = 1024;
    } else if (a.tiny) {
        c.vocab_size = std::min(tok_vocab, static_cast<size_t>(5000));
        c.hidden_size = 128; c.num_heads = 4; c.num_layers = 2;
        c.intermediate_size = 512;
    } else {
        c.vocab_size = std::min(tok_vocab, static_cast<size_t>(5000));
        c.hidden_size = 128; c.num_heads = 4; c.num_layers = 2;
        c.intermediate_size = 512;
    }
    c.max_seq_length = a.seq_len;
    c.head_dim = c.hidden_size / c.num_heads;
    return c;
}

std::vector<float> last_row(const Matrix& logits) {
    std::vector<float> out(logits.cols());
    size_t r = logits.rows() - 1;
    for (size_t v = 0; v < logits.cols(); ++v) out[v] = logits(r, v);
    return out;
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i)
        m = std::max(m, std::abs(a[i] - b[i]));
    return m;
}

}  // namespace

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        std::string s = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::cerr << s << " needs a value\n"; exit(2); }
            return argv[++i];
        };
        if (s == "--ckpt") a.ckpt = next();
        else if (s == "--vocab-list") a.vocab_list = next();
        else if (s == "--vocab-file") a.vocab_files.push_back(next());
        else if (s == "--prompt") a.prompts.push_back(next());
        else if (s == "--dump") a.dump_path = next();
        else if (s == "--mid") a.mid = true;
        else if (s == "--tiny") a.tiny = true;
        else if (s == "--seq-len") a.seq_len = std::stoul(next());
        else { std::cerr << "unknown arg: " << s << "\n"; return 2; }
    }
    if (a.ckpt.empty() || (a.vocab_files.empty() && a.vocab_list.empty())
        || a.prompts.empty()) {
        std::cerr << "usage: golden_batch_test --ckpt latest.ckpt "
                     "--vocab-file train-0.txt [--vocab-file train-1.txt] "
                     "--mid --seq-len 256 --prompt \"user: hi assistant:\" "
                     "[--dump logits.tsv]\n";
        return 2;
    }

    auto tokenizer = std::make_shared<TiktokenTokenizer>();
    if (!a.vocab_list.empty()) {
        tokenizer->load_vocabulary_from_id_list(a.vocab_list);
        std::cout << "[GB] vocab " << tokenizer->vocab_size()
                  << " words from id list " << a.vocab_list << std::endl;
    } else {
        for (const auto& f : a.vocab_files)
            tokenizer->build_vocabulary_from_plain_text(f);
        std::cout << "[GB] vocab " << tokenizer->vocab_size() << " words from "
                  << a.vocab_files.size()
                  << " file(s) — WARNING: file-built vocab ids are only valid "
                     "on the platform that trained the checkpoint (no sort "
                     "tie-break); prefer --vocab-list from the run's GGUF"
                  << std::endl;
    }

    TransformerConfig cfg = make_config(a, tokenizer->vocab_size());
    Transformer model(cfg, tokenizer);
    ModelSaver saver;
    if (!saver.loadCheckpoint(model, a.ckpt)) {
        std::cerr << "[GB] FAIL: cannot load " << a.ckpt << std::endl;
        return 1;
    }
    model.set_tokenizer(tokenizer);
    model.set_training(false);  // dropout at inference makes every leg a coin
                                // flip (cross-process max|dlogit| 1.68 on an
                                // identical ckpt before this line existed)
    std::cout << "[GB] checkpoint loaded: " << a.ckpt << std::endl;

    // Forward all golden prompts on the loaded model.
    // Use forward_batch (batch of 1): it is the path TRAINING optimizes
    // through, and the single-seq forward() is historically untrustworthy
    // (fused no-RoPE kernel on CUDA; on CPU it predicted a constant token
    // regardless of context when this tool was first run, 2026-07-22).
    std::vector<std::vector<int>> enc;
    std::vector<std::vector<float>> logits1;
    for (const auto& p : a.prompts) {
        auto ids = tokenizer->encode(p);
        TransformerOutput out = model.forward_batch({ids}, ids.size());
        if (out.logits.rows() < ids.size()) {
            std::cerr << "[GB] FAIL: batched logits not materialized (rows="
                      << out.logits.rows() << ") for prompt: " << p << std::endl;
            return 1;
        }
        // row (len-1) predicts the token AFTER the prompt
        std::vector<float> row(out.logits.cols());
        for (size_t v = 0; v < out.logits.cols(); ++v)
            row[v] = out.logits(ids.size() - 1, v);
        enc.push_back(ids);
        logits1.push_back(row);
    }

    // Leg: save -> reload into a FRESH model -> identical logits.
    const std::string tmpdir = "/tmp/golden_ckpt";
    std::system(("rm -rf " + tmpdir + " && mkdir -p " + tmpdir).c_str());
    if (!saver.saveCheckpoint(model, tmpdir, "golden", 0, 0.0f, 0)) {
        std::cerr << "[GB] FAIL: save" << std::endl;
        return 1;
    }
    Transformer model2(cfg, tokenizer);
    // saveCheckpoint names the file; find it.
    std::string saved;
    {
        FILE* p = popen(("ls -t " + tmpdir + "/*.ckpt 2>/dev/null | head -1").c_str(), "r");
        char buf[512];
        if (p && fgets(buf, sizeof buf, p)) { saved = buf; }
        if (p) pclose(p);
        while (!saved.empty() && (saved.back() == '\n' || saved.back() == '\r'))
            saved.pop_back();
    }
    if (saved.empty() || !saver.loadCheckpoint(model2, saved)) {
        std::cerr << "[GB] FAIL: reload from " << (saved.empty() ? tmpdir : saved)
                  << std::endl;
        return 1;
    }
    model2.set_tokenizer(tokenizer);
    model2.set_training(false);

    float worst = 0.f;
    for (size_t i = 0; i < enc.size(); ++i) {
        TransformerOutput out = model2.forward_batch({enc[i]}, enc[i].size());
        std::vector<float> row(out.logits.cols());
        for (size_t v = 0; v < out.logits.cols(); ++v)
            row[v] = out.logits(enc[i].size() - 1, v);
        worst = std::max(worst, max_abs_diff(logits1[i], row));
    }
    const bool save_ok = worst < 1e-4f;
    std::cout << "[GB] save->reload->forward max|delta logit| = " << worst
              << (save_ok ? "  PASS" : "  *** FAIL ***") << std::endl;

    // Leg: dump last-position logits for engine-side differential comparison.
    if (!a.dump_path.empty()) {
        std::ofstream f(a.dump_path);
        for (size_t i = 0; i < a.prompts.size(); ++i) {
            f << a.prompts[i] << "\t";
            for (size_t v = 0; v < logits1[i].size(); ++v)
                f << (v ? " " : "") << logits1[i][v];
            f << "\n";
        }
        std::cout << "[GB] dumped " << a.prompts.size() << " logit rows -> "
                  << a.dump_path << std::endl;
    }
    return save_ok ? 0 : 1;
}

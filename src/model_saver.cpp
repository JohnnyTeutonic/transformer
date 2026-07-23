#include "../include/model_saver.hpp"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <sstream>
#include <nlohmann/json.hpp>

namespace fs = std::filesystem;
using json = nlohmann::json;

ModelSaver::ModelSaver() : logger(Logger::getInstance()) {}

bool ModelSaver::saveModel(const Transformer& transformer, const std::string& directory,
                           const std::string& model_name) {
    try {
        // Convert to absolute path
        std::filesystem::path current_path = std::filesystem::current_path();
        std::filesystem::path dir_path = current_path / directory;
        
        // Create directory if it doesn't exist
        if (!std::filesystem::exists(dir_path)) {
            std::cout << "Creating directory path: " << dir_path << std::endl;
            if (!std::filesystem::create_directories(dir_path)) {
                logger.log("Failed to create directory: " + dir_path.string(), true);
                return false;
            }
        }

        // Verify directory is writable
        if (!std::filesystem::is_directory(dir_path)) {
            logger.log("Path exists but is not a directory: " + dir_path.string(), true);
            return false;
        }

        std::error_code ec;
        auto perms = std::filesystem::status(dir_path, ec).permissions();
        if (ec || (perms & std::filesystem::perms::owner_write) == std::filesystem::perms::none) {
            logger.log("Directory is not writable: " + dir_path.string(), true);
            return false;
        }

        // Create model file path
        std::filesystem::path model_path = dir_path / (model_name + ".model");
        logger.log("Saving model to: " + model_path.string());

        // Save model configuration first
        if (!writeMetadata(dir_path.string(), model_name, transformer.getConfig())) {
            logger.log("Failed to save model metadata", true);
            return false;
        }

        // Open model file for writing with validation
        std::ofstream model_file(model_path, std::ios::binary);
        if (!model_file) {
            logger.log("Failed to open model file for writing", true);
            return false;
        }

        // Write model version and timestamp
        nlohmann::json header;
        header["version"] = "1.0";
        header["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
        std::string header_str = header.dump();
        size_t header_size = header_str.size();
        
        if (!model_file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size)) ||
            !model_file.write(header_str.c_str(), header_size)) {
            logger.log("Failed to write model header", true);
            return false;
        }

        // Save each layer's weights with validation
        size_t total_bytes_written = 0;
        const auto& layers = transformer.getLayers();
        
        for (const auto& layer : layers) {
            size_t bytes_before = model_file.tellp();
            layer->save(model_file);
            size_t bytes_after = model_file.tellp();
            
            if (!model_file) {
                logger.log("Failed to write layer data", true);
                return false;
            }
            
            total_bytes_written += (bytes_after - bytes_before);
        }

        // Ensure everything is written
        model_file.flush();
        if (!model_file) {
            logger.log("Error occurred while writing model file", true);
            return false;
        }

        // Verify file size
        model_file.close();
        uintmax_t file_size = std::filesystem::file_size(model_path);
        if (file_size == 0 || file_size < total_bytes_written) {
            logger.log("Model file size verification failed. Expected at least " + 
                      std::to_string(total_bytes_written) + " bytes, got " + 
                      std::to_string(file_size), true);
            return false;
        }

        logger.log("Model saved successfully (" + std::to_string(file_size) + " bytes written)");
        return true;
    } catch (const std::exception& e) {
        logger.log("Error saving model: " + std::string(e.what()), true);
        return false;
    }
}

bool ModelSaver::loadModel(Transformer& transformer, const std::string& directory,
                           const std::string& model_name) {
    try {
        std::string model_path = directory + "/" + model_name;
        logger.log("Loading model from: " + model_path);

        // Load and verify configuration
        TransformerConfig config;
        if (!readMetadata(directory, model_name, config)) {
            logger.log("Failed to read model metadata", true);
            return false;
        }

        // Verify configuration matches
        if (config != transformer.getConfig()) {
            logger.log("Model configuration mismatch", true);
            return false;
        }

        // Load model weights
        std::ifstream model_file(model_path + ".bin", std::ios::binary);
        if (!model_file) {
            logger.log("Failed to open model file for reading", true);
            return false;
        }

        // Load each layer's weights
        auto& layers = transformer.getLayers();
        for (auto& layer : layers) {
            layer->load(model_file);
        }

        logger.log("Model loaded successfully");
        return true;
    } catch (const std::exception& e) {
        logger.log("Error loading model: " + std::string(e.what()), true);
        return false;
    }
}

bool ModelSaver::saveCheckpoint(const Transformer& transformer, const std::string& directory,
                                const std::string& model_name, int epoch, float loss, int step) {
    try {
        // Convert to absolute path
        std::filesystem::path current_path = std::filesystem::current_path();
        std::filesystem::path dir_path = current_path / directory;
        std::cerr << "[CKPT DEBUG] cwd=" << current_path << " dir=" << dir_path << std::endl;
        
        // Create full directory path if it doesn't exist
        if (!std::filesystem::exists(dir_path)) {
            std::cerr << "[CKPT DEBUG] Creating directory: " << dir_path << std::endl;
            if (!std::filesystem::create_directories(dir_path)) {
                std::cerr << "[CKPT ERROR] Failed to create directory: " << dir_path << std::endl;
                return false;
            }
        }

        // Verify directory is writable
        if (!std::filesystem::is_directory(dir_path)) {
            std::cerr << "[CKPT ERROR] Not a directory: " << dir_path << std::endl;
            return false;
        }

        // Skip Unix-style permission check on Windows (uses ACLs instead)
        // The actual file write test below is the reliable check

        // Create checkpoint filename with absolute path
        std::string checkpoint_file = (dir_path / getCheckpointFilename("", model_name, epoch)).string();
        std::cerr << "[CKPT DEBUG] Saving to: " << checkpoint_file << std::endl;

        // ATOMIC WRITE: stream into a .tmp sibling and rename at the end.
        // The old code truncated and rewrote the live file in place (twice —
        // a writability probe, then the real write), so any concurrent copy
        // (the Colab relay loop mirrors this file) could capture an empty or
        // partial checkpoint.
        std::string tmp_file = checkpoint_file + ".tmp";
        std::ofstream ckpt_file(tmp_file, std::ios::binary);
        if (!ckpt_file) {
            std::cerr << "[CKPT ERROR] Failed to open for binary write: " << tmp_file << std::endl;
            return false;
        }
        std::cerr << "[CKPT DEBUG] File opened, writing metadata..." << std::endl;

        // Serialize the full model payload to memory first so its hash can be
        // recorded in the metadata. The Colab relay (download -> OneDrive ->
        // upload) was caught delivering full-length checkpoints whose tensors
        // mixed two save generations (2026-07-18: restored models measured
        // 15-18 CE vs ~5-6 recorded); a content hash makes every consumer
        // able to reject such files outright.
        std::ostringstream payload_stream(std::ios::binary);
        transformer.save(payload_stream);
        const std::string payload = payload_stream.str();
        uint64_t fnv = 14695981039346656037ULL;
        for (unsigned char ch : payload) {
            fnv ^= ch;
            fnv *= 1099511628211ULL;
        }
        char fnv_hex[17];
        std::snprintf(fnv_hex, sizeof(fnv_hex), "%016llx",
                      static_cast<unsigned long long>(fnv));

        // Write metadata as JSON
        nlohmann::json checkpoint_meta;
        const auto& config = transformer.getConfig();

        checkpoint_meta["epoch"] = epoch;
        checkpoint_meta["step"] = step;
        checkpoint_meta["loss"] = loss;
        // Format 2 (2026-07-17): full-model payload via Transformer::save
        // (embeddings + pos encoding + layers + final_ln + LM head). Format-1
        // checkpoints contained ONLY the transformer layers and cannot
        // restore a working model.
        // Format 3 (2026-07-18): LayerNorm blocks now carry eps + rms flag
        // (format-2 loads rebuilt every norm as non-RMS LayerNorm — llama
        // models restored with the wrong normalizer and scored ~2x loss).
        checkpoint_meta["format"] = 3;
        checkpoint_meta["payload_fnv64"] = std::string(fnv_hex);
        checkpoint_meta["timestamp"] = std::chrono::system_clock::now().time_since_epoch().count();
        checkpoint_meta["model_config"] = {
            {"hidden_size", config.hidden_size},
            {"num_heads", config.num_heads},
            {"num_layers", config.num_layers},
            {"head_dim", config.head_dim},
            {"intermediate_size", config.intermediate_size},
            {"max_seq_length", config.max_seq_length},
            // Architecture primitives (added 2026-07-23): without these the
            // norm/positional/bias scheme is not recoverable from the
            // checkpoint, so a config reconstructed elsewhere (e.g. a test
            // harness or an external resume) silently used additive PE +
            // LayerNorm + live biases and diverged ~5x from the RoPE/RMSNorm/
            // bias-free model it actually was. Same bug class as the format-3
            // LayerNorm eps/rms fix above.
            {"use_rope", config.use_rope},
            {"use_rms_norm", config.use_rms_norm},
            {"use_biases", config.use_biases},
        };

        std::string meta_str = checkpoint_meta.dump();
        size_t meta_size = meta_str.size();

        // Write metadata size and content with validation
        if (!ckpt_file.write(reinterpret_cast<const char*>(&meta_size), sizeof(meta_size))) {
            logger.log("Failed to write metadata size", true);
            return false;
        }
        if (!ckpt_file.write(meta_str.c_str(), meta_size)) {
            logger.log("Failed to write metadata content", true);
            return false;
        }

        // Write the pre-serialized full-model payload (hashed above).
        // (The previous code iterated only transformer.getLayers(), so
        // embeddings, final_ln and the LM head were never in the checkpoint —
        // every resume restarted them from random init.)
        if (!ckpt_file.write(payload.data(), static_cast<std::streamsize>(payload.size()))) {
            logger.log("Failed to write model data", true);
            return false;
        }
        size_t total_bytes_written = payload.size();

        // Ensure everything is written and validate
        ckpt_file.flush();
        if (!ckpt_file) {
            logger.log("Error occurred while writing checkpoint file", true);
            return false;
        }
        ckpt_file.close();

        // Atomically publish the finished file
        std::error_code rename_ec;
        std::filesystem::rename(tmp_file, checkpoint_file, rename_ec);
        if (rename_ec) {
            std::cerr << "[CKPT ERROR] Failed to publish checkpoint (rename): "
                      << rename_ec.message() << std::endl;
            return false;
        }

        // Verify file size
        std::filesystem::path checkpoint_path(checkpoint_file);
        uintmax_t file_size = std::filesystem::file_size(checkpoint_path);
        if (file_size == 0 || file_size < total_bytes_written) {
            logger.log("Checkpoint file size verification failed. Expected at least " + 
                      std::to_string(total_bytes_written) + " bytes, got " + 
                      std::to_string(file_size), true);
            return false;
        }

        logger.log("Checkpoint saved successfully (" + 
                  std::to_string(file_size) + " bytes written)");
        return true;
    } catch (const std::exception& e) {
        logger.log("Error saving checkpoint: " + std::string(e.what()), true);
        return false;
    }
}

bool ModelSaver::loadCheckpoint(Transformer& transformer, const std::string& checkpoint_path) {
    std::ifstream ckpt_file(checkpoint_path, std::ios::binary);
    if (!ckpt_file) {
        std::cerr << "[LOAD ERROR] Failed to open checkpoint file: " << checkpoint_path << std::endl;
        return false;
    }

    try {
        // Read metadata size
        size_t meta_size;
        ckpt_file.read(reinterpret_cast<char*>(&meta_size), sizeof(meta_size));
        std::cerr << "[LOAD DEBUG] Metadata size: " << meta_size << std::endl;
        
        if (meta_size > 10000 || meta_size == 0) {
            std::cerr << "[LOAD ERROR] Invalid metadata size (likely corrupted checkpoint)" << std::endl;
            return false;
        }

        // Read metadata JSON
        std::string meta_str(meta_size, '\0');
        ckpt_file.read(&meta_str[0], meta_size);
        std::cerr << "[LOAD DEBUG] Metadata: " << meta_str.substr(0, 200) << "..." << std::endl;

        json checkpoint_meta = json::parse(meta_str);

        // Verify model configuration
        const auto& config = transformer.getConfig();
        const auto& saved_config = checkpoint_meta["model_config"];
        
        std::cerr << "[LOAD DEBUG] Saved config: hidden=" << saved_config["hidden_size"] 
                  << " heads=" << saved_config["num_heads"]
                  << " layers=" << saved_config["num_layers"] << std::endl;
        std::cerr << "[LOAD DEBUG] Model config: hidden=" << config.hidden_size 
                  << " heads=" << config.num_heads
                  << " layers=" << config.num_layers << std::endl;

        if (saved_config["hidden_size"] != config.hidden_size ||
            saved_config["num_heads"] != config.num_heads ||
            saved_config["num_layers"] != config.num_layers ||
            saved_config["head_dim"] != config.head_dim ||
            saved_config["intermediate_size"] != config.intermediate_size ||
            saved_config["max_seq_length"] != config.max_seq_length) {
            std::cerr << "[LOAD ERROR] Model configuration mismatch in checkpoint" << std::endl;
            return false;
        }

        // Reject format-1 (layers-only) checkpoints: they lack embeddings,
        // final_ln and the LM head, so "restoring" one produces a broken
        // model that scores ~ln(vocab) uniform loss.
        int ckpt_format = checkpoint_meta.value("format", 1);
        if (ckpt_format != 3) {
            std::cerr << "[LOAD ERROR] Checkpoint format " << ckpt_format
                      << " cannot restore a working model (format 1: layers "
                      << "only; format 2: norm layers restored with the wrong "
                      << "normalizer type). Requires format 3." << std::endl;
            return false;
        }

        // Read the remaining payload into memory, verify its hash against the
        // metadata (when present), then load. Rejects relay-corrupted files —
        // full-length checkpoints mixing two save generations were caught in
        // the Colab download path (2026-07-18).
        std::cerr << "[LOAD DEBUG] Loading full model from position " << ckpt_file.tellg() << std::endl;
        std::string payload((std::istreambuf_iterator<char>(ckpt_file)),
                            std::istreambuf_iterator<char>());
        if (checkpoint_meta.contains("payload_fnv64")) {
            uint64_t fnv = 14695981039346656037ULL;
            for (unsigned char ch : payload) {
                fnv ^= ch;
                fnv *= 1099511628211ULL;
            }
            char fnv_hex[17];
            std::snprintf(fnv_hex, sizeof(fnv_hex), "%016llx",
                          static_cast<unsigned long long>(fnv));
            std::string expected = checkpoint_meta["payload_fnv64"].get<std::string>();
            if (expected != fnv_hex) {
                std::cerr << "[LOAD ERROR] Checkpoint payload hash mismatch (file "
                          << fnv_hex << " vs recorded " << expected
                          << ") — corrupt/mixed relay copy, refusing to load." << std::endl;
                return false;
            }
            std::cerr << "[LOAD DEBUG] Payload hash verified (" << fnv_hex << ")" << std::endl;
        }
        std::istringstream payload_stream(payload, std::ios::binary);
        transformer.load(payload_stream);
        if (!payload_stream && !payload_stream.eof()) {
            std::cerr << "[LOAD ERROR] Failed to load model data" << std::endl;
            return false;
        }

        std::cerr << "[LOAD SUCCESS] Loaded checkpoint from epoch " 
                  << checkpoint_meta["epoch"].get<int>() << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[LOAD ERROR] Exception: " << e.what() << std::endl;
        return false;
    }
}

bool ModelSaver::loadLatestCheckpoint(Transformer& transformer, const std::string& directory,
                                      const std::string& model_name, int& epoch, float& loss) {
    try {
        // Find latest checkpoint
        int latest_epoch = -1;
        for (const auto& entry : fs::directory_iterator(directory)) {
            if (entry.path().extension() == ".ckpt") {
                std::string filename = entry.path().stem().string();
                if (filename.find(model_name + "_checkpoint_") == 0) {
                    int checkpoint_epoch =
                        std::stoi(filename.substr(filename.find_last_of("_") + 1));
                    latest_epoch = std::max(latest_epoch, checkpoint_epoch);
                }
            }
        }

        if (latest_epoch == -1) {
            logger.log("No checkpoints found", true);
            return false;
        }

        // Load the latest checkpoint
        std::string checkpoint_file = getCheckpointFilename(directory, model_name, latest_epoch);
        if (!loadCheckpoint(transformer, checkpoint_file)) {
            return false;
        }

        // Update epoch and loss from the loaded checkpoint
        std::ifstream ckpt_file(checkpoint_file, std::ios::binary);
        size_t meta_size;
        ckpt_file.read(reinterpret_cast<char*>(&meta_size), sizeof(meta_size));

        std::string meta_str(meta_size, '\0');
        ckpt_file.read(&meta_str[0], meta_size);

        json checkpoint_meta = json::parse(meta_str);
        epoch = checkpoint_meta["epoch"];
        loss = checkpoint_meta["loss"];

        return true;
    } catch (const std::exception& e) {
        logger.log("Error loading latest checkpoint: " + std::string(e.what()), true);
        return false;
    }
}

std::string ModelSaver::createDirectory(const std::string& base_dir) const {
    fs::path dir_path(base_dir);
    fs::create_directories(dir_path);
    return dir_path.string();
}

std::string ModelSaver::getCheckpointFilename(const std::string& directory,
                                              const std::string& model_name, int epoch) const {
    if (directory.empty()) {
        return model_name + "_checkpoint_" + std::to_string(epoch) + ".ckpt";
    }
    return directory + "/" + model_name + "_checkpoint_" + std::to_string(epoch) + ".ckpt";
}

bool ModelSaver::writeMetadata(const std::string& directory, const std::string& model_name,
                               const TransformerConfig& config) const {
    json meta;
    meta["model_name"] = model_name;
    meta["hidden_size"] = config.hidden_size;
    meta["num_heads"] = config.num_heads;
    meta["num_layers"] = config.num_layers;
    meta["head_dim"] = config.head_dim;
    meta["intermediate_size"] = config.intermediate_size;
    meta["max_seq_length"] = config.max_seq_length;
    meta["use_flash_attention"] = config.use_flash_attention;
    meta["use_rope"] = config.use_rope;
    meta["use_sliding_window"] = config.use_sliding_window;
    meta["window_size"] = config.window_size;

    std::ofstream meta_file(directory + "/" + model_name + ".meta.json");
    meta_file << std::setw(4) << meta << std::endl;
    return true;
}

bool ModelSaver::readMetadata(const std::string& directory, const std::string& model_name,
                              TransformerConfig& config) const {
    std::ifstream meta_file(directory + "/" + model_name + ".meta.json");
    if (!meta_file) {
        return false;
    }

    json meta;
    meta_file >> meta;

    config.hidden_size = meta["hidden_size"];
    config.num_heads = meta["num_heads"];
    config.num_layers = meta["num_layers"];
    config.head_dim = meta["head_dim"];
    config.intermediate_size = meta["intermediate_size"];
    config.max_seq_length = meta["max_seq_length"];
    config.use_flash_attention = meta["use_flash_attention"];
    config.use_rope = meta["use_rope"];
    config.use_sliding_window = meta["use_sliding_window"];
    config.window_size = meta["window_size"];

    return true;
}

void ModelSaver::save_vocabulary(const std::string& path, const Vocabulary& vocab) {
    try {
        // Save special token mappings in consistent order
        special_tokens_json = {
            {"<pad>", 0},
            {"<unk>", 1},
            {"<bos>", 2},
            {"<eos>", 3},
            {"<mask>", 4}
        };

        // Write to file
        std::ofstream file(path);
        if (!file) {
            logger.log("Failed to open vocabulary file for writing: " + path, true);
            return;
        }
        file << std::setw(4) << special_tokens_json << std::endl;
    } catch (const std::exception& e) {
        logger.log("Error saving vocabulary: " + std::string(e.what()), true);
    }
}
# Chat Experiments — findings ledger (v5–v8, 2026-07-19 → )

Companion to EXPORT_NOTES.md (export/engine bugs) and PERF_NOTES.md
(throughput). This file records the *scientific* findings from the chat
model campaign — several are publishable-shape observations in their own
right, and two feed directly into `AI_ML/choquet_transformer` (the
repetition-hunger of binding) and future scaling work.

## Model ledger

| model | config | data | key number | verdict |
|---|---|---|---|---|
| chat5 (v5) | 2L/128h/seq128 | TinyStories-Instruct (fragmented) | loss 3.04 | first coherent GPU model; stories, weak conditioning |
| chat6 (v6) | 2L/128h/seq128 | DailyDialog | loss 3.61 | dialogue register, wandering content |
| chat7a | 4L/256h/seq256 | DailyDialog+Empathetic (prose-windowed) | loss ~4.1 | topic-following multi-turn; QA semantics absent |
| chat7b | 4L/256h/seq256 | Instruct (aggregation-fixed) | loss 3.25 | fluent narrative arcs; zero conditioning (~0.2 epochs) |
| micro-bind | 2L/128h, CPU | 1.5k TinyChat pairs, 60 epochs | ppl 1.4 | PERFECT 6/6 dialogue-act binding (greedy) |
| chat8b @1.3ep | 4L/256h/seq256 | 150k TinyChat (doc-aligned) | loss 0.35 | partial binding 4/6; extended to ~4 epochs |
| chat8b @3ep | 4L/256h/seq256 | 150k TinyChat x3 (doc-aligned) | loss 0.35; gen eval below | intent-wise binding; bound intents GENERALISE |
| chat8a @18k | 4L/256h/seq256 | Instruct aggregated, ~1.5 epochs | loss ~2.8-3.0 | fluent; ZERO conditioning (dragon test failed) |
| chat8c | 4L/256h/seq256 | tinychatmix 8000 steps | loss 4.28 | mix dilutes binding — CONFOUNDED by F5 |

## chat8a dragon test (2026-07-21, greedy, raw prompt) — repetition test CLOSED

18,000 steps (~1.5 epochs over 2.28M aggregated instruct examples):
"words: dragon , dark , brave story:" and summary-conditioning probes all
produce fluent stories that ignore the fields; greedy collapses to one
corpus-head opening ("once upon a time, there was a little girl named
lily..."). vs chat7b at 0.2 epochs: same zero conditioning, better
fluency. CONCLUSION (with Finding 1): epochs alone cannot rescue
conditioning when the pattern inventory itself is combinatorial — the
words: field draws from thousands of combinations, so 1.5 or even 10
epochs of the corpus is <<1 exposure per pattern. Binding budgets must
count exposures per semantic equivalence class (Jonathan's framing),
and the instruct corpus has ~none. TinyChat's inventory is small enough
to repeat; that is WHY 8b binds. (OOV caveat F5 applies to 8a's tail
words but probe words are high-frequency; the conclusion stands.)

## chat8b generalisation eval (2026-07-21, greedy, raw prompt, engine rebuilt)

Result-matrix row (eval_tinychat_generalisation.py, 20 probes):
- Tier A (train-form): answer-type 6/11, semantic-slot 6/11
- Tier B (unseen paraphrase): answer-type 4/9, semantic-slot 3/9
- Sampled@0.7 pass: A 6/11, B 4/9 — no greedy/sampled gap.
- Control floor (chat5 story model): slot 0/20.

Reading: binding at ~3 epochs is INTENT-WISE, and it transfers: tier B
tracks tier A per intent (wellbeing/food/day_activity generalise to unseen
surface forms — semantic abstraction, not template memorisation), while
drink_offer/hobby/pet fail in BOTH tiers. What is learned is learned as a
relation; what is missing is missing everywhere. Refines Finding 1: more
epochs should convert intents wholesale, not polish surfaces.

ANOMALY (open, feeds golden-batch suite): failures are mostly intent-
correct but function-word-dropped ("no thank milk ."). [TOPK] logging
(new TINYLLAMA_TOPK_DEBUG=1) shows the dropped word losing by 0.11 logits
(5.27 vs 5.15) at a corpus-deterministic position, while neighbouring
positions win by 12+. Ruled out: repetition penalty (identical at 1.0),
prompt wrapping (word-level path is raw), stale binary (rebuilt — NB the
CLI binary had been 9 days / 3 fix-generations stale and silently killed
F32 models via the pre-fix layer-clearing path; quantized models masked
it). Remaining suspects: genuine underlearning of those intents vs
trainer/engine logit divergence at specific positions — exactly what the
golden-batch GGUF-parity test decides. Engine got --rep-penalty,
--raw-prompt, and [TOPK] debug flags out of the investigation.

## Finding 1 — Binding is repetition-hungry and is learned last

Register, fluency, and corpus CE saturate long before conditional binding
(question-type -> answer-type; words: -> story content) emerges. Evidence,
three independent setups:
- chat7b: 937k-seq corpus, each conditioning pattern seen <=1x -> fluent,
  zero conditioning.
- micro-bind vs chat8b: identical data distribution; 60 epochs -> 6/6
  perfect binding; 1.3 epochs -> 4/6 with class confusions.
- choq2 cell 1 (33.6M, different task family): CE saturates by step ~1500,
  probe accuracy leaves chance only after ~3000.
Practical rule: for binding-type capabilities at small scale, budget
EPOCHS OVER THE PATTERN INVENTORY, not tokens. A dataset too large to
repeat is a dataset too large to bind.

## Finding 2 — Capacity-matched data unlocks coherence (TinyStories thesis
## transfers to dialogue)

A 13M model on open-domain dialogue (DailyDialog, 20k-cap vocab) learns
the *manifold* (turn-taking, register, reply shapes) but not the
*conditional map* — greedy decoding collapses to one generic reply across
question classes. The same architecture on a 145-word synthetic dialogue
world (TinyChat, gen in scripts/prepare_chat_data.py) binds
question->answer semantics. Coherence at small scale is a data-bandwidth
match problem before it is a parameter-count problem. Escalation path:
grow the world's vocabulary with capacity (curriculum), not ahead of it.

## Finding 3 — Prose windowing destroys dialogue conditioning signal

The stream-packing loader (built for WikiText/TinyStories) applied to
dialogue: 97% of training windows began MID-dialogue (often mid-sentence),
2.3 dialogues per window with attention crossing eos, 50%-overlap stride
re-exposing every dialogue in differently-truncated form (audit
2026-07-20, DailyDialog). Responses train without their prompts; sparse QA
signal drowns. Fix: --doc-aligned loader (whole-document bin-packing,
turn-boundary splits, no overlap, PAD ignore-index). Pairing itself was
never misaligned (post-eos token = "user:" in 2932/2932 windows) — the
harm was windowing, not shift.

## Finding 5 — Vocab-cap mismatch = silent OOV embedding corruption
## (found 2026-07-21 by the tokenizer audit; affects EVERY lane except pure
## TinyChat — chat5, chat6, chat7a, chat7b, chat8a, chat8c; log-verified
## for 8a/8c: "Total vocabulary size: 20000" + "Vocab size: 5000")

The tokenizer caps its vocabulary at 20k; the --mid/--tiny presets cap the
MODEL at 5k. Nothing reconciled the two, and the CUDA embedding gather
(token_embedding_cuda.cu) is unchecked: every token with id >= 5000 read
out-of-bounds GPU memory as its embedding vector — silent garbage, run
after run. Ids are frequency-ranked, so the corruption hit exactly the
open-domain tail (TinyChat's 148 words all rank low and were untouched;
that is why 8b was clean). CONSEQUENCE FOR FINDINGS: the mix-dilution
result (8c) and the open-domain incoherence attributions (chat6/7a inside
Finding 2) are CONFOUNDED — part of "capacity mismatch" may be literal
input corruption on tail words. The clean TinyChat results stand. Fix
(2026-07-21): trainer remaps OOV ids to UNK with a reported rate
([VOCAB] line), and forward_cuda bounds-checks before the gather. Any
future mixing/breadth experiment must rerun post-fix before comparison.
Moral (same family as Finding 4): an unchecked device gather is another
lying metric — plausible outputs, corrupt inputs.

## Roadmap adopted 2026-07-21 (Jonathan's review, recorded verbatim in spirit)

Sequence: (1) evaluate 8a/8b GENERALISATION, not just the six probes —
unseen paraphrases + held-out slot combinations distinguish abstraction
from memorisation; result matrix per run: {train probes, paraphrases, new
combos}. (2) assistant-only loss (--assistant-loss, implemented
2026-07-21: role-marker state machine masks user-turn/marker targets via
the ignore-index; arms to compare: full CE vs assistant-only vs ~20%
prompt loss [weighted variant TODO — needs kernel scaling]). (3) vocab as
architecture: audit numbers below; TIE embeddings (deferred into the
resident-tensor refactor — one authoritative weight representation, host
tensors as snapshots, version-numbered if dual state is unavoidable).
(4) hidden-state intent probe at the assistant boundary (linear decode of
prompt intent from the pre-first-answer-token vector; decodable-but-wrong
=> head/objective/decoding problem, not-decodable => representation never
formed). (5) TinyChat++ as a controlled world grown one axis at a time
(intents / lexicon / surface / state), with COMBINATION holdouts; track
exposures per semantic equivalence class, not corpus epochs. (6) golden-
batch differential test suite (CPU-vs-CUDA one-step state equality is the
strongest test; save->load->identical loss; GGUF round-trip logits;
resume->identical next update). (7) resident-tensor refactor, then
6L/512h flagship at the SHORTEST sufficient context (pilot seq 128 vs 256
against the empirical dialogue-length distribution; compare at equal
PATTERN EXPOSURES, not tokens). (8) no more compute on DailyDialog
through current capacity.

Vocab audit numbers (mid preset, hidden 256, untied): at V=5000,
embedding+head = 2.56M of 5.73M total = 45% of parameters serving
vocabulary, most rows nearly untrained under Zipf sparsity; at V=148
(TinyChat) = 2.3%. Tying input/output embeddings halves the vocab share
(~22% freed at V=5000) and couples lexical representation to prediction.

## Finding 4 — (engineering, cross-ref) Silent train/deploy divergence

The FeedForward stale-GPU-cache bug (README, resolved 2026-07-19) meant
every pre-fix GPU model trained around a frozen random FFN. Post-fix, loss
improved from a lying ~6.5-plateau to an honest 3.0 and exports became
servable. Moral for the paper-adjacent record: verify train==deploy
function equality with differential activation traces before interpreting
any quality result; "the metric was lying" outranks most modeling
hypotheses in prior probability.

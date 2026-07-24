# The actual edge: a volunteer training *commons*, not a token network

Written 2026-07-24. The distinctive position isn't the trainer, the engine, or
"decentralized training" in the abstract — those are crowded (DiLoCo, DeMo,
Prime Intellect, Nous). It's a design stance already sitting in this repo's
trust layer that almost no one else is taking.

## What everyone else is building

- **Closed labs** (Prime Intellect INTELLECT, Nous Psyche): decentralized
  *within a coordinated org*, not open to arbitrary individuals. Impressive,
  but not a commons.
- **Token/crypto networks** (Bittensor, Gensyn): open, but the incentive is a
  speculative asset. Contribution is mediated by a coin, which pulls in
  mercenary compute and adversarial gaming as the *primary* dynamic.

Nobody serious is building the third thing.

## The third thing (what `incentive_alignment.hpp` already encodes)

A **BOINC / Folding@home model for transformer training** — a volunteer
commons whose incentive is *mission*, not a token. The design is already in
the code, and it is coherent:

- `academic_reserved_share = 0.4` — 40% of capacity reserved for academic use.
- `contribution_ratio` priority, `credit_decay_rate` to reward *ongoing*
  donation, `new_contributor_grace_days` — a credit economy with **no coin**,
  no speculation, no exit-to-cash. You earn priority by contributing, and it
  decays if you stop. That is Folding@home's social contract, not Bittensor's.
- `organization_type` distinguishes individual / academic / commercial and
  prioritizes the first two.

This is the exact thing you said you cared about — *open source meaning
individuals, not companies* — expressed as a resource-allocation policy. It's
a **values-differentiated** position the funded players structurally cannot
occupy: a closed lab can't be a commons, and a token network can't not be a
market.

## Why the C++ node matters *here* specifically

A commons runs on whatever hardware volunteers already own — old GPUs, laptops,
heterogeneous consumer machines. A lean, dependency-light C++ node is a genuine
advantage in that setting where a heavy Python/CUDA stack is a deployment tax.
Folding@home won on exactly this: a small client anyone could run.

## The one hard problem worth owning (and the honest gap)

The blocker for an *open* commons (vs. a closed lab) is **trust**: a volunteer
you don't control can fake work or poison the model. The repo's current answer:

- `GradientFingerprint` + clustering → flag outlier (poisoned) gradients.
- `CrossValidationTask` → hand a node a known input with a known expected
  gradient and check tolerance.

Honest weakness: **spot-checks are spoofable.** A malicious node can detect the
known test inputs and compute honestly on *those* while cheating on real work.
This is the actual open research problem, and it's the contribution worth
building:

- **Trajectory-based proof-of-learning**: the node commits (hash) to its
  intermediate states along the training path; the coordinator re-executes a
  *randomly chosen, short* segment and checks it reproduces the committed
  states. Cheap to verify, expensive to fake, and can't be special-cased
  because the checked segment is chosen after commitment. The `golden-batch`
  equality machinery already in this repo (bit-exact reproduce-a-recorded-state)
  is *exactly* the primitive proof-of-learning needs — you already built the
  verifier; it just needs to run on a challenged segment instead of a checkpoint.

That is a real, publishable systems contribution, it is not crowded, and it
sits one connection away from code that already exists here.

## What this is NOT

Not a claim to out-scale the labs — a volunteer commons will not pretrain a
frontier model first. The claim is narrower and defensible: **the only open,
non-token, individual-and-academic-first training commons, on a node light
enough to actually run on donated hardware, with verifiable contribution.**
That niche is empty because the people with funding are structurally barred
from it. It's yours to take precisely because you're not one of them.

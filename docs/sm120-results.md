# Measurements and qualification

## C1 decode results

One RTX PRO 4500 Blackwell 32GB, SM120, Linux, driver 615.78.08 open module,
PyTorch 2.14.1+cu132. Official target `nvidia/Qwen3.8-27B-NVFP4@482ca0f`.
All packages include the FlashInfer PV-tail fix over `03de0b66`.
Three replicates per case and arm, 2048-token output budget, natural EOS:

| Case | Plain | MTP3 | MTP7 | BF16 DFlash2 | NVFP4 DFlash2 |
|---|---:|---:|---:|---:|---:|
| Chat | 37.3 | 74.7 | 64.0 | 84.6 | 87.6 |
| Code | 37.3 | 90.8 | 96.8 | 150.8 | 154.2 |
| Reasoning | 37.3 | 99.8 | 116.9 | 173.0 | 181.3 |
| Agent | 37.3 | 75.3 | 62.1 | 75.6 | 85.1 |

Observed **min–max native decode rates (tok/s)** across the three runs per cell
follow, rounded to two decimal places.
These are sample ranges, not confidence intervals; the generated text and lengths
differ across arms. The headline uses the medians above.

| Case | Plain | MTP3 | MTP7 | BF16 DFlash2 | NVFP4 DFlash2 |
|---|---:|---:|---:|---:|---:|
| Chat | 37.31–37.33 | 71.83–76.33 | 63.73–65.95 | 81.75–86.11 | 85.81–89.88 |
| Code | 37.29–37.31 | 84.74–92.57 | 78.48–100.14 | 149.43–157.41 | 134.86–179.18 |
| Reasoning | 37.28–37.31 | 97.99–100.16 | 112.47–119.23 | 168.47–183.81 | 178.52–188.50 |
| Agent | 37.27–37.30 | 72.20–80.86 | 62.04–72.11 | 70.23–76.19 | 84.24–85.35 |

The [12 synthetic request bodies](../benchmarks/qwen38-c1-requests.json)
(4 cases × 3 seeds) and [60 per-request measurement rows](../benchmarks/qwen38-c1-results.csv)
are included. Each arm used the same bodies; generated outputs are not bundled.

Numbers are median native per-request decode tok/s, excluding prefill/first-token
latency. All 60 requests completed without a runtime failure. MTP3/MTP7 mean fixed
3/7 drafts (widths 4/8), adaptive off, using the checkpoint's built-in MTP head
through the same dense recipe family; DFlash2 offers 7 drafts at width 8.
Chat/code/agent: thinking off, temperature .7, top-p .8, top-k 20, presence 1.5.
Reasoning: thinking on, temperature 1, top-p .95, top-k 20, presence 0.
No reasoning-effort setting was requested. Different arms produce different text
and lengths. Small natural-output samples do not isolate algorithmic acceptance
from content effects or establish quality equivalence.

DFlash2 uses the target's quantized head, FP32 selector arithmetic and Flare's
sampling filters for q; the q retained for verification is the distribution actually
sampled. The published drafter reference uses different q filtering/arithmetic.
These are not published-acceptance parity claims. Controlled common-prefix tests
showed DFlash2 below MTP on chat acceptance (about 3.2–3.3 versus 3.46 emitted per
round); agent acceptance gains were small. Cheaper rounds can still improve decode
rate. Prefix tests are conditional acceptance screens, not end-to-end speed estimates.

## Other engineering results

- Calibrated NVFP4 draft retention saves **2,402,382,848 bytes (2.24 GiB)** versus
  the BF16 draft. This is drafter weight storage, not total server-memory reduction.
- MLP scratch sharing reduces the calculated 4096-token graph-catalog storage by
  **1,688,207,360 bytes (1.57 GiB)**. The MLP-only 4096 variant was GPU-qualified.
- The attention-memory experiment at C4/64K/prefill1024 reduced package mappings
  **22.23 → 22.07 GiB**, increasing the reported KV allocation **192,512 → 196,608
  tokens** and snapshot slots **30 → 31** with the same reserve. This is separate
  from the much larger calculated prefill-catalog MLP saving.
- The 128K capacity package completed **130,800 prompt + 128 generated tokens**.
  This is a capacity test, not 128K model-quality or full C4-pressure qualification.
- A separate matched scheduler experiment removed final-prefill starvation:
  plain C4 aggregate median **85.0 → 154.7 tok/s**, worst first-token spread
  **6.148 → 0.309 s**. C1 and speculative arms showed no detectable change at this
  sample size. This comparison used a ce7a4ba-era runtime plus the fix, not this
  release backport; plain/MTP used the third-party all-NVFP4 target, and DFlash2
  used a private official-target build. There were two engine starts per arm,
  two cohorts nested in each start. This +82% is not a DFlash2 gain and must not
  be multiplied by the C1 ratios.

## Validation and limits

The combined release has the full recipe/engine source and pinned dependency.
The scheduler port deliberately scores higher-value stale feedback winners and
drains the pipeline before executing them; this can change stale-key refresh
timing. Grammar/host-stop and every lifecycle/vision combination have not been
exhaustively regressed, despite the generic barrier invariants and CPU tests.
Clean-environment installation and the one-line alias quickstart have not been
validated end to end. The successful fresh export used the public source with
the existing pinned environment and local copies of the exact checkpoints.

Its exact frozen-base scheduler backport passed **220 scheduler tests** and built
as a release native CLI. The recipe configuration/feedback/scratch CPU selection
passed **37 tests, 2 skipped** in the existing preparation environment. Current
combined-tree native C4 check completed **52/52 requests total** across plain
and fixed7: **per arm, 24 pressure requests + 2 cold/warm probes** (2 × 26). Both arms
had 14 pressure-time prefix hits and cold/warm reuse of 0 → 4,880 cached tokens, using the earlier
C4/64K/prefill1024 package. This checks the new backported runtime, not a fresh
export or a new throughput comparison.

Earlier component and native checks covered committed context rings, inactive
lanes, prefix reuse, snapshots, scoring and variable output budgets. The official
target's earlier B4 plain/fixed3/adaptive3 run passed 216 requests, plus post-drain
reuse probes; pressure-time hits were absent in that small official cache budget.
Later DFlash2 stress passed 222/222 requests. Those checks used earlier measured
packages; they do not assert exhaustive qualification of every new combination.

The combined-tree poisoned-tail attention CUDA regression also passed **4/4**
cases (BF16/FP8 × C1/C4, including inactive rows and boundary lengths).

The public export example then completed a **fresh C4/64K/prefill1024 package**,
including compilation, warm-up, capture, verified readback and final assembly.
The first assembly attempt ran out of workspace disk space; a retry succeeded
on a larger output filesystem. It retained about 42 GiB including its local
export cache. This fresh package passed **52/52 native C4 requests total** across
plain and fixed7: **per arm, 24 pressure requests + 2 cold/warm probes** (2 × 26),
with 13 and 12 pressure-time cache hits respectively, and 0 → 4,880 cached tokens in both
post-drain probes. This is a runtime qualification, not a new speed comparison.

Known gaps remain:

- Attention scratch at **prefill4096 is not GPU-qualified**. Use the example's
  prefill1024. The separate current-main MLP port in issue45 was CPU-only; this
  distribution carries the older experimentally qualified implementation instead.
- Independent HF ±1-layer tap parity and every window/convolution reference case
  are not complete. Retained-q correctness is distinct from reference agreement.
- Full unrelated-model GPU regression (including Nemotron) has not been run.
- A separate larger 1024-row parity fixture exceeded its reference backing size
  before comparison; it was not counted as a pass. The 512-row parity set passed.
- No quality-equivalence claim: an authored 96-request set had 0 transport failures,
  but 12 plain and 11 DFlash2 automatic semantic failures, including budget/literal
  checks, with 68 cases needing manual review. It is not a statistical quality gate.
- Long context increases prefill latency; a 65K prompt took about 17.5 seconds
  before generation in the earlier **third-party all-NVFP4 target** setup, rather
  than the official-target C1 table above. Decode gains do not remove that cost.
- RTX5090 compatibility follows the intended SM120 target; this repository's
  measurements are on RTX PRO4500, not a measured RTX5090 result.

No model weights, private prompts, request ledgers, signing keys, local machine
configuration or generated runtime packages are included. Models retain their
original publisher licenses; review the linked model cards.

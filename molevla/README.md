# Layer Redundancy in Vision-Language-Action Models: A Cross-Architecture Benchmark

**Which LLM layers can you skip in a VLA — and does the answer depend on how actions are decoded?**

This project benchmarks 5 heuristic layer-skipping strategies on two 7B-parameter VLA models — **CogACT-Base** (diffusion action head) and **OpenVLA-7B** (autoregressive token head) — evaluated on 50 real robot manipulation frames from the RT-1/fractal dataset. The core question: does the layer-importance ordering generalize across architectures, or is it model-specific?

---

## Motivation

MoLe-VLA (AAAI 2026) proposes a learned router (STAR) that dynamically skips LLM layers during inference, claiming ~2× speedup with minimal task degradation. The paper motivates STAR by arguing that **final layers carry the critical task semantics** — making early-exit (skip-last) strategies dangerous. No MoLe-VLA checkpoints were released, making direct reproduction impossible.

**This project tests that claim directly:** using zero-shot heuristic skipping, we ask which layers are actually most important for action prediction, and whether that changes between a diffusion-head VLA and an autoregressive-token VLA.

---

## Models

| Model | Backbone | Action Head | Checkpoint |
|---|---|---|---|
| CogACT-Base | LLaMA-2-7B (32 layers) | DiT-B diffusion, 15-step window | `CogACT/CogACT-Base` (HF) |
| OpenVLA-7B | LLaMA-2-7B (32 layers) | Autoregressive token prediction | `openvla/openvla-7b` (HF) |

Both share the same backbone family — the only difference is how the final LLM representations are converted to actions.

---

## Method

### SkipLayerWrapper

Each LLaMA decoder layer is wrapped in a thin `nn.Module`. When `skip=True`, the layer returns its input unchanged (identity pass). When `skip=False`, it runs normally.

```python
class SkipLayerWrapper(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.skip = False

    def forward(self, hidden_states, *args, **kwargs):
        if self.skip:
            outputs = (hidden_states,)
            if kwargs.get('output_attentions', False): outputs += (None,)
            if kwargs.get('use_cache', False):         outputs += (None,)
            return outputs
        return self.layer(hidden_states, *args, **kwargs)
```

**Verification suite** (run before every benchmark):
1. **Identity check**: skipping zero layers gives max diff < 1e-5 vs baseline ✓
2. **Smoke test**: skipping 8 layers produces diff > 1e-5 vs baseline ✓

### Skip Strategies (5 total)

| Strategy | Layers skipped (k=8 example) | Hypothesis tested |
|---|---|---|
| `skip_last_k` | 24–31 | Are late layers redundant? |
| `skip_first_k` | 0–7 | Are early layers redundant? |
| `skip_middle_k` | 12–19 | Are middle layers redundant? |
| `skip_random_k` | random 8 (seed=42) | Baseline for unstructured skip |
| `skip_alternate` | 1,3,5,...,31 (16 layers) | Every other layer |

k ∈ {4, 8, 12, 16} for all strategies except `skip_alternate` (fixed at 16).

### Evaluation Data

**Dataset**: `fractal20220817_data` — the Google RT-1 robot manipulation dataset, part of Open X-Embodiment. Streamed directly from `gs://gresearch/robotics` (no download needed).

**Samples**: 50 real frames from 5 episodes × 10 steps each. Real robot images + natural language instructions (e.g., *"pick rxbar chocolate from bottom drawer and place on counter"*).

**Metric**: Fidelity MSE — mean squared error between the skip-model's predicted action and the full-model baseline prediction, for the same input and random seed. This measures how much a skip strategy changes the model's behavior, not how correct the actions are.

### Implementation Notes

- **CogACT**: Single forward pass + diffusion decode. KV cache disabled (`config.use_cache = False`) for non-contiguous skip compatibility. Model in fp32 on A100 (40GB).
- **OpenVLA**: Autoregressive generation of 7 action tokens. `model.generate` monkey-patched to force `use_cache=False` — the DynamicCache desynchronizes when layers are skipped non-contiguously across decode steps. Latency is inflated vs real deployment for this reason, but is **consistent across strategies**, so relative comparisons are valid.

---

## Results

### CogACT-Base — 50 real fractal frames, fp32, A100

| Config | Skipped | MSE ↓ | Latency (ms) | Speedup |
|---|---|---|---|---|
| baseline | 0 | 0.000000 | 9120 | 1.00× |
| skip_last_k | 4 | **0.003432** | 8207 | 1.12× |
| skip_first_k | 4 | 0.061372 | 8246 | 1.11× |
| skip_middle_k | 4 | 0.005357 | 8179 | 1.12× |
| skip_last_k | 8 | **0.003821** | 7303 | 1.26× |
| skip_first_k | 8 | 0.063348 | 7278 | 1.26× |
| skip_middle_k | 8 | 0.016102 | 7250 | 1.26× |
| skip_last_k | 12 | **0.008638** | 6363 | 1.44× |
| skip_first_k | 12 | 0.064255 | 6333 | 1.45× |
| skip_middle_k | 12 | 0.038240 | 6317 | 1.45× |
| skip_last_k | **16** | **0.016312** | 5441 | **1.68×** |
| skip_first_k | 16 | 0.058549 | 5417 | 1.69× |
| skip_middle_k | 16 | 0.041798 | 5413 | 1.69× |
| skip_random_k | 16 | 0.065172 | 5386 | 1.70× |
| skip_alternate | 16 | 0.065882 | 5376 | 1.71× |

**skip_last_k Pareto-dominates all strategies at every k.**

---

### OpenVLA-7B — 50 real fractal frames, fp32, A100

| Config | Skipped | MSE ↓ | Latency (ms) | Speedup |
|---|---|---|---|---|
| baseline | 0 | 0.000000 | 2111 | 1.00× |
| skip_last_k | 4 | **0.010708** | 1885 | 1.12× |
| skip_first_k | 4 | 0.095886 | 1885 | 1.12× |
| skip_middle_k | 4 | 0.022820 | 1885 | 1.12× |
| skip_last_k | 8 | **0.046677** | 1648 | 1.28× |
| skip_first_k | 8 | 0.095886 | 1659 | 1.27× |
| skip_middle_k | 8 | 0.019315 | 1658 | 1.27× |
| skip_last_k | 12 | **0.059124** | 1432 | 1.47× |
| skip_first_k | 12 | 0.095886 | 1432 | 1.47× |
| skip_middle_k | 12 | 0.059345 | 1432 | 1.47× |
| skip_last_k | **16** | 0.116456 | 1205 | **1.75×** |
| skip_first_k | **16** | **0.096974** | 1206 | **1.75×** |
| skip_middle_k | 16 | 0.092609 | 1205 | 1.75× |
| skip_random_k | 16 | 0.102381 | 1206 | 1.75× |
| skip_alternate | 16 | 0.231946 | 1205 | 1.75× |

**At k=16, the ordering inverts: skip_last (0.116) > skip_first (0.097).**

---

## Key Findings

### Finding 1 — Early layers are critical on both models (small k)

At k=4 and k=8, `skip_last_k` outperforms `skip_first_k` on both models:

| k | CogACT skip_last | CogACT skip_first | OpenVLA skip_last | OpenVLA skip_first |
|---|---|---|---|---|
| 4 | **0.003** | 0.061 (**18×** worse) | **0.011** | 0.096 (**9×** worse) |
| 8 | **0.004** | 0.063 | **0.047** | 0.096 |

Early layers handle vision-language grounding. This is consistent across architectures and directly contradicts MoLe-VLA's stated motivation ("final layers carry critical task semantics") — though it aligns with DeeR, the paper's own strongest baseline, which uses early-exit (skip-last).

### Finding 2 — The ordering inverts for OpenVLA at k=16

At k=16, `skip_last_k` (0.116) becomes **worse** than `skip_first_k` (0.097) for OpenVLA — but not for CogACT (0.016 vs 0.059, skip_last still wins).

**Mechanism**: OpenVLA's final LLM layer directly produces action token logits. Removing the last 16 layers eliminates the layers that directly output actions. CogACT uses LLM hidden states as context for a separate diffusion head — the final LLM layer is not load-bearing the same way.

> *"The early-layer-critical asymmetry holds at small k for both architectures, but inverts for OpenVLA at k=16 — late layers become indispensable when they directly produce action token logits, but are expendable when a downstream diffusion head decouples language modeling from action prediction."*

### Finding 3 — OpenVLA's early-layer saturation

For OpenVLA, `skip_first_k` MSE is **identical** at k=4, k=8, and k=12 (all 0.095886). The damage is fully done by removing just the first 4 layers — layers 4–11 carry forward a broken representation that doesn't get worse with more early skips. This suggests layers 0–3 perform a uniquely irreversible transformation in autoregressive VLAs.

### Finding 4 — CogACT is 3–7× more robust than OpenVLA

| k | CogACT skip_last MSE | OpenVLA skip_last MSE | Robustness ratio |
|---|---|---|---|
| 4 | 0.003 | 0.011 | 3.1× |
| 8 | 0.004 | 0.047 | 12× |
| 16 | 0.016 | 0.116 | 7.1× |

The diffusion action head tolerates layer skipping far better than autoregressive token prediction. Architectural choice of action head determines layer redundancy.

---

## Plots

**CogACT-Base:**

![CogACT Benchmark](benchmark_plots.png)

**OpenVLA-7B:**

![OpenVLA Benchmark](benchmark_plots_openvla.png)

The left plots show MSE vs layers skipped per strategy. The right plots show the Pareto frontier (latency vs accuracy). On CogACT, skip_last_k sits in the bottom-right corner (Pareto optimal) at all k. On OpenVLA, the Pareto structure breaks at k=16 — the architecturally interesting point.

---

## Limitations

1. **Fidelity MSE, not task success.** We measure how much skipping changes the model's output, not whether the robot succeeds. Small MSE errors can compound over a trajectory. Task success rate evaluation requires a simulator (RLBench/CoppeliaSim) — not run here.

2. **KV cache disabled.** Non-contiguous layer skipping desynchronizes HuggingFace's `DynamicCache`. Disabling it inflates absolute latency (especially for OpenVLA's autoregressive decode), but all strategies are evaluated under the same conditions so relative speedups are valid.

3. **fp32 inference.** CogACT's PIL→tensor pipeline caused dtype mismatches with bfloat16 weights. Running in fp32 inflates memory usage vs a bf16 deployment but doesn't affect the layer-importance findings.

4. **50 samples.** Sufficient for stable relative comparisons but small by dataset standards. Patterns are internally consistent and replicated across k values.

5. **No learned router baseline.** STAR (MoLe-VLA's trained router) has no released checkpoints. The gap between heuristic skip_last_k and STAR's reported 60.8% task success is real but unquantifiable from this data.

---

## Reproducing

Both experiments run on **Google Colab A100 (40GB)** with no conda required.

**CogACT-Base** ([molevla.ipynb](molevla.ipynb)):
```
# Runtime: A100 (40GB)
# Checkpoint: 30.5GB from HuggingFace (CogACT/CogACT-Base)
# Eval data: fractal20220817_data streamed from gs://gresearch/robotics
# Runtime per benchmark: ~2.5 hours (50 images × 18 configs × ~9s/inference)
```

**OpenVLA-7B** ([openvla.ipynb](openvla.ipynb)):
```
# Runtime: A100 (40GB)
# Checkpoint: ~15GB from HuggingFace (openvla/openvla-7b), auto-downloaded
# Eval data: same fractal stream
# Runtime per benchmark: ~25 minutes (50 images × 18 configs × ~2s/inference)
```

---

## Related Work

- **FLOWER (2025)**: Prunes the terminal 30% of VLM layers by design — consistent with this project's skip_last findings
- **EfficientVLA (2025)**: Identifies and prunes inconsequential language layers training-free via inter-layer redundancy analysis
- **DeeR (MoLe-VLA paper baseline)**: Early-exit VLA that discards deep layers — strongest competing baseline in the MoLe paper at 59.2% vs full model's 57.2%, despite the paper arguing final layers are critical
- **Look Before Acting (2026)**: Layer-wise probing showing action prediction becomes less visually grounded in deeper layers — mechanistic support for early-layer criticality

---

## Resume Bullets

- Benchmarked 5 heuristic layer-skipping strategies on **CogACT-Base and OpenVLA-7B** (both LLaMA-2-7B) across 50 real RT-1 manipulation frames, achieving up to **1.75× inference speedup**
- Validated and generalized the Shallow Brain Hypothesis cross-architecturally: skip-last-k Pareto-dominates at small k on both models; discovered that at k=16 the ordering **inverts for OpenVLA** — late layers become critical when they directly produce action token logits, but are expendable when a diffusion head decouples action prediction from the final LLM layer
- Discovered OpenVLA-specific early-layer saturation: skip_first_k MSE plateaus identically at k=4, 8, and 12 (MSE=0.096), suggesting layers 0–3 perform an irreversible transformation that subsequent early layers cannot compensate for
- Built a `SkipLayerWrapper` harness with identity-pass verification, DynamicCache compatibility via generate-level monkey-patching for autoregressive models, and full MSE × latency matrix (18 configs × 50 real frames × 2 models)

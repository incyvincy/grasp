# GRASP — Robot Learning Experiments

Two independent research experiments on improving robot manipulation policies.

---

## Projects

### 1. ACT CVAE Image Encoder and Conditional Prior

**Directory:** [`cvae_encoder_task/`](cvae_encoder_task/)

Extends LeRobot's ACT policy so that visual observations influence the CVAE latent variable during both training and inference.

**What was changed:**
- Added `use_images_in_vae_encoder` flag: injects a pooled image token into the CVAE posterior encoder alongside robot state and expert actions.
- Added `use_conditional_prior` flag: learns a prior `p(z | images)` and trains the posterior against it with `KL(q(z|s,I,a) || p(z|I))` instead of against `N(0, I)`.
- During inference, the learned prior mean replaces the all-zero latent used in vanilla ACT.

**Task:** MetaWorld `shelf-place-v3` | **Dataset:** 200 recorded expert episodes | **Backbone:** LeRobot ACT

**Ablations supported:**

| Experiment | Images in CVAE | Conditional Prior |
|---|:---:|:---:|
| Original ACT baseline | false | false |
| Posterior image token only | true | false |
| Conditional prior only | false | true |
| Complete model | true | true |

See [`cvae_encoder_task/README.md`](cvae_encoder_task/README.md) for full setup, training commands, and architecture diagrams.

---

### 2. Layer Redundancy in Vision-Language-Action Models

**Directory:** [`molevla/`](molevla/)

Benchmarks 5 heuristic layer-skipping strategies on two 7B-parameter VLA models to determine which LLM layers are redundant for action prediction — and whether the answer depends on the action head architecture.

**Models evaluated:**
- **CogACT-Base** — LLaMA-2-7B backbone with a DiT-B diffusion action head
- **OpenVLA-7B** — LLaMA-2-7B backbone with an autoregressive token action head

**Skip strategies:** `skip_last_k`, `skip_first_k`, `skip_middle_k`, `skip_random_k`, `skip_alternate` at k ∈ {4, 8, 12, 16}

**Dataset:** 50 real robot manipulation frames from `fractal20220817_data` (Google RT-1 / Open X-Embodiment), streamed from GCS.

**Key findings:**
- `skip_last_k` Pareto-dominates all strategies on **CogACT** at every k (up to 1.68× speedup, MSE 0.016 at k=16).
- At k=16 the ordering **inverts for OpenVLA**: removing final layers is worse than removing early ones — because the final LLM layer directly produces action token logits in autoregressive heads.
- CogACT is 3–12× more robust to layer skipping than OpenVLA, attributable to the diffusion head decoupling action prediction from the final LLM layer.
- OpenVLA shows early-layer saturation: `skip_first_k` MSE is identical at k=4, 8, and 12 (MSE = 0.096), suggesting layers 0–3 perform an irreversible transformation.

See [`molevla/README.md`](molevla/README.md) for full results tables, plots, and reproduction instructions.

---

## Notebooks

| Notebook | Task |
|---|---|
| [`cvae_encoder_task/normal_act.ipynb`](cvae_encoder_task/normal_act.ipynb) | ACT baseline training |
| [`cvae_encoder_task/modified_act.ipynb`](cvae_encoder_task/modified_act.ipynb) | Modified ACT with image encoder + conditional prior |
| [`molevla/molevla.ipynb`](molevla/molevla.ipynb) | CogACT-Base layer-skipping benchmark |
| [`molevla/openvla.ipynb`](molevla/openvla.ipynb) | OpenVLA-7B layer-skipping benchmark |

---

## Hardware

Both MoLe-VLA experiments were run on **Google Colab A100 (40 GB)**. ACT training requires a CUDA GPU with sufficient memory for `batch_size=32`.

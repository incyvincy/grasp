# ACT CVAE Image Encoder and Conditional Prior

## Goal

This task extends LeRobot's ACT policy so that visual observations can influence
the CVAE latent variable during both training and inference.

Original ACT behavior:

- The CVAE posterior is conditioned on robot state and expert actions.
- Training regularizes the posterior against a standard normal distribution.
- Inference does not have expert actions, so the latent vector is set to zero.

Our target behavior:

- Optionally add image information to the CVAE posterior encoder.
- Learn an image-conditioned prior `p(z | images)`.
- Train the posterior against this learned prior.
- Use the learned prior mean during inference instead of an all-zero latent.

Both additions are optional and disabled by default. This allows the modified
code to run either the new experiment or the original ACT baseline.

## Files Changed

### `src/lerobot/policies/act/configuration_act.py`

Two configuration flags were added:

```python
use_images_in_vae_encoder: bool = False
use_conditional_prior: bool = False
```

### `src/lerobot/policies/act/modeling_act.py`

The ACT model now supports:

1. Extracting ResNet camera features once and reusing them.
2. Spatially pooling each camera feature map and averaging across cameras.
3. Projecting the pooled visual feature into an image token.
4. Inserting that token into the CVAE encoder sequence before the actions.
5. Expanding the CVAE positional encoding and padding mask for the image token.
6. Predicting the conditional prior mean and log variance from images.
7. Computing:

```text
KL(q(z | state, images, actions) || p(z | images))
```

8. Using the conditional prior mean as the latent during inference.
9. Returning posterior and prior parameters to `ACTPolicy` for loss calculation.

When the new flags are disabled, ACT continues to use:

```text
KL(q(z | state, actions) || N(0, I))
```

and uses an all-zero latent during inference.

### `recort_expert.py`

The expert recording script was updated to:

- Record 200 successful episodes by default.
- Call `_ensure_env()` in both `reset()` and `step()`.
- Let the current LeRobot API manage the episode buffer automatically.
- Use `dataset.clear_episode_buffer()` to discard failed episodes.
- Save only successful MetaWorld `shelf-place-v3` episodes.

## Architecture

### Training with both features enabled

```text
Images -> ResNet -> pooled image feature
                       |
                       +-> image token
                       |      |
[CLS, robot state, image token, expert actions]
                       |
                 CVAE encoder
                       |
             posterior q(z | s, I, a)

pooled image feature -> conditional prior p(z | I)

Loss = L1 action loss + KL weight * KL(q || p)
```

### Inference with both features enabled

```text
Images -> ResNet -> pooled image feature -> prior mean -> latent z
                                                        |
Robot state + image features + z -> ACT -> action chunk
```

Expert actions are not available during inference, so the learned image prior
provides the latent information.

## Setup

Run commands from the LeRobot repository directory containing
`pyproject.toml`, `recort_expert.py`, and `src/`.

Use the Python environment in which LeRobot, PyTorch, MetaWorld, and the
training dependencies are already installed.

If uploading the policy or using online W&B logging, authenticate with the
corresponding services before starting training.

## Record the Dataset

Check these values in `recort_expert.py` before recording:

```python
REPO_ID = "lerobot/shelf-place-v3"
TASK_NAME = "shelf-place-v3"
NUM_EPISODES = 200
ROOT_DIR = Path("data")
```

Record 200 successful episodes:

```bash
python recort_expert.py --num-episodes 200
```

Verify an existing dataset:

```bash
python recort_expert.py --verify-only
```

Failed attempts are discarded with `clear_episode_buffer()`. Only successful
episodes are saved.

## Train the New Model

`--dataset.root` must point to the actual dataset directory. The examples below
use `dataset-1`, matching the current training command. Change it if the
recorded dataset is stored elsewhere.

### Jupyter or Colab

```python
!python -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --policy.use_images_in_vae_encoder=true \
  --policy.use_conditional_prior=true \
  --policy.repo_id=incyvincy/new \
  --dataset.repo_id=lerobot/shelf-place-v3 \
  --dataset.root=dataset-1 \
  --env.type=metaworld \
  --env.task=shelf-place-v3 \
  --output_dir=outputs2/train/new \
  --job_name=new \
  --steps=20000 \
  --batch_size=32 \
  --wandb.enable=true \
  --policy.device=cuda
```

### PowerShell

Do not include `!` in a terminal. PowerShell uses the backtick for multiline
commands:

```powershell
python -m lerobot.scripts.lerobot_train `
  --policy.type=act `
  --policy.use_images_in_vae_encoder=true `
  --policy.use_conditional_prior=true `
  --policy.repo_id=incyvincy/new `
  --dataset.repo_id=lerobot/shelf-place-v3 `
  --dataset.root=dataset-1 `
  --env.type=metaworld `
  --env.task=shelf-place-v3 `
  --output_dir=outputs2/train/new `
  --job_name=new `
  --steps=20000 `
  --batch_size=32 `
  --wandb.enable=true `
  --policy.device=cuda
```

## Train the Original ACT Baseline

Set both new flags to `false`:

```python
!python -m lerobot.scripts.lerobot_train \
  --policy.type=act \
  --policy.use_images_in_vae_encoder=false \
  --policy.use_conditional_prior=false \
  --policy.repo_id=incyvincy/new-baseline \
  --dataset.repo_id=lerobot/shelf-place-v3 \
  --dataset.root=dataset-1 \
  --env.type=metaworld \
  --env.task=shelf-place-v3 \
  --output_dir=outputs2/train/new-baseline \
  --job_name=new-baseline \
  --steps=20000 \
  --batch_size=32 \
  --wandb.enable=true \
  --policy.device=cuda
```

Because both options default to `false`, removing the two flags also runs the
original ACT behavior.

## Run Individual Ablations

Only add images to the posterior CVAE encoder:

```text
--policy.use_images_in_vae_encoder=true
--policy.use_conditional_prior=false
```

Only learn an image-conditioned prior:

```text
--policy.use_images_in_vae_encoder=false
--policy.use_conditional_prior=true
```

Enable the complete proposed model:

```text
--policy.use_images_in_vae_encoder=true
--policy.use_conditional_prior=true
```

Recommended experiment comparison:

| Experiment | Images in CVAE | Conditional prior |
|---|---:|---:|
| Original ACT baseline | false | false |
| Posterior image token only | true | false |
| Conditional prior only | false | true |
| Complete model | true | true |

Use different `policy.repo_id`, `output_dir`, and `job_name` values for each
experiment so checkpoints and W&B runs do not overwrite or mix results.

## Expected Outputs

Training outputs are written under the selected directory:

```text
outputs2/train/new
```

The loss dictionary contains:

- `l1_loss`: action reconstruction loss.
- `kld_loss`: posterior-to-prior KL loss.

With the conditional prior disabled, `kld_loss` is computed against `N(0, I)`.
With it enabled, `kld_loss` is computed against the image-conditioned prior.

## Checks Already Completed

The implementation has been checked with:

- Python compilation.
- Ruff on the modified ACT source files.
- Existing focused ACT tests.
- A CPU smoke test covering training, backward propagation, and inference with
  both new flags enabled.

## Important Notes

- CUDA must be available for `--policy.device=cuda`.
- `batch_size=32` must fit in GPU memory.
- `--wandb.enable=true` enables W&B only when its project configuration and
  authentication are available.
- `policy.repo_id` controls the policy repository used when pushing to the Hub.
- The dataset root must contain the metadata and episode data expected by
  `LeRobotDataset`; it is not merely an arbitrary parent directory.
- Use separate output directories when comparing the baseline and new model.

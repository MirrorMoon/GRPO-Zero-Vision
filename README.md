# GRPO-Zero Vision

This repository adapts Group Relative Policy Optimisation (GRPO) to vision-only
models.  Instead of sampling textual trajectories from a language model, we use
Monte Carlo Dropout to draw multiple tensor predictions from a Vision
Transformer (ViT) and optimise them directly with reinforcement signals that
operate on logits, bounding boxes or segmentation masks.  The codebase keeps the
original emphasis on minimal dependencies and explicit implementations.

## Highlights

- **Vision-first GRPO** – Monte Carlo Dropout generates diverse tensor
  hypotheses for each input image so the algorithm can compute per-group
  advantages without language generation.
- **Lightweight ViT backbone** – a self-contained Vision Transformer with
  configurable depth, width and dropout is provided in `vit_model.py`.
- **Image-folder training loop** – `train.py` reads standard image folder
  datasets, rolls out Monte Carlo samples, scores them with verifiable rewards
  and updates the policy via the tensor-aware GRPO path.
- **Pure PyTorch stack** – only PyTorch, Pillow and TensorBoard are required at
  runtime.

## Algorithm Overview

For each training step we sample a batch of images and, for every image, draw
`T` stochastic forward passes by keeping dropout layers active during inference.
These `T` logits vectors replace the `T` textual responses used in
language-model GRPO.  Rewards are computed directly on the logits (e.g.
classification accuracy) and normalised per input image before computing the
policy gradient.  The adapter in `grpo.classification_policy_adapter` interprets
the logits as categorical distributions and provides log-probabilities for the
chosen action so that the PPO-style update can run unchanged.

## Dataset Format

`VisionClassificationDataset` expects an image folder with one subdirectory per
class:

```
data/
  train/
    class_a/
      img_001.png
      ...
    class_b/
      ...
  eval/
    class_a/
      ...
```

Images are resized to the configured resolution, optionally augmented with a
horizontal flip during training, normalised using ImageNet statistics and
batched into `TensorMiniBatch` objects for the rollout stage.

## Reward Function

`vision_task.classification_reward` compares the Monte Carlo logits with the
ground-truth label and assigns a reward of `1` to correct predictions and `0`
otherwise.  The predicted class index is also stored in the episode metadata to
inform the policy adapter.

## Training

1. Install dependencies and prepare data:

   ```bash
   pip install -r requirements.txt
   # Place your dataset under data/train and data/eval following the structure above
   ```

   If you prefer using `uv`, install it first and run `uv sync` to resolve the
   same dependency set defined in `pyproject.toml`:

   ```bash
   pip install uv
   uv sync
   ```

2. Launch training (default configuration assumes 10 classes and 224×224
   inputs):

   ```bash
   uv run train.py
   ```

3. To use an alternative configuration (e.g. a larger ViT for 24 GB GPUs):

   ```bash
   uv run train.py --config config_24GB.yaml
   ```

Checkpoints are saved under the directory specified by `training.ckpt_dir`, and
TensorBoard logs are written to `training.log_dir`.

## Acknowledgements

This project draws inspiration from the original GRPO research by DeepSeek, the
TinyZero and nano-aha-moment implementations, as well as the ViT architecture
introduced by Dosovitskiy et al.  The MC Dropout adaptation follows the Bayesian
interpretation popularised by Gal & Ghahramani.


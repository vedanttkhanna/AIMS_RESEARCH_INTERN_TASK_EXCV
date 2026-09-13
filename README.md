# COVID-19 Chest X-ray Classification: CNN vs Vision Transformer + Explainability

This notebook trains and compares a CNN and two Vision Transformer architectures on a COVID-19 chest X-ray classification task, then rigorously evaluates *how* each model makes its decisions using multiple explainability and faithfulness-testing techniques.

## Dataset

- **Source:** `covid19-xray-aims/COVID_19_dataset` (Kaggle), pre-split into `train/`, `val/`, `test/` folders organized by class (loaded via `torchvision.datasets.ImageFolder`).
- **Preprocessing:** images resized to 224×224, normalized with ImageNet mean/std.
- **Augmentation (train only):** random horizontal flip, random rotation (±10°), color jitter.
- Class names/counts are inferred automatically from the folder structure.

## Models Trained

| Model | Backbone | Library | Notes |
|---|---|---|---|
| EfficientNet-B0 | CNN | `torchvision` | ImageNet-pretrained, fine-tuned classifier head |
| DeiT-Small (patch16, 224) | Vision Transformer | `timm` | ImageNet-pretrained, fully fine-tuned |
| ViT-Base (patch16, 224) | Vision Transformer | `timm` | ImageNet-pretrained, fully fine-tuned |

All models use:
- Adam optimizer + cosine annealing LR schedule
- Early stopping on validation loss (patience = 5)
- Up to 25 epochs
- Cross-entropy loss
- Best checkpoint saved to `outputs/*_best.pth`
- Test-set evaluation with classification report + confusion matrix, saved as JSON/PNG

## Explainability Methods

Since CNNs and Transformers "see" images differently, the notebook applies architecture-appropriate interpretability techniques:

- **Grad-CAM** and **Grad-CAM++** (EfficientNet) — gradient-based class activation heatmaps from the final conv layer.
- **Attention Rollout** (DeiT/ViT) — recursively multiplies each transformer block's self-attention maps (extracted via forward hooks on the QKV projection) to trace how the `[CLS]` token attends to image patches.
- **LIME** — perturbation-based, model-agnostic superpixel explanations (applied to EfficientNet).

## Faithfulness / Quality Evaluation of Explanations

Beyond just visualizing heatmaps, the notebook quantitatively scores how *trustworthy* each explanation method is:

- **Insertion / Deletion curves** — measure how model confidence changes as the most "important" pixels (per the saliency map) are progressively inserted into / removed from a blurred baseline image.
- **AOPC (Area Over the Perturbation Curve)** — summarizes deletion-curve confidence drop into a single faithfulness score.
- **Entropy** of saliency maps — measures how concentrated vs. diffuse an explanation is.
- **Baseline Sensitivity Score (BSS)** — checks how stable AOPC is across different perturbation baselines (blur, black, noise).
- **Deletion Correlation** — Pearson correlation between saliency magnitude and actual confidence drop, i.e. how well the map's ranking matches real impact.
- A final **bonus multi-metric comparison** benchmarks Grad-CAM, Grad-CAM++, LIME (EfficientNet) against Attention Rollout (DeiT) side-by-side on the same test images.
- A **per-class explainability pass** generates Grad-CAM / Attention Rollout overlays for sample images from each class (COVID / Normal / Viral Pneumonia), to sanity-check whether the models focus on clinically plausible regions (e.g., bilateral ground-glass opacities for COVID).

## Outputs

All artifacts are written to `outputs/`, including:
- Trained model weights (`*_best.pth`)
- Training curve plots and confusion matrices
- Grad-CAM / Grad-CAM++ / Attention Rollout / LIME overlay images
- Insertion-deletion curves and faithfulness metric comparison charts
- JSON result summaries (`*_results.json`, `phase4_results.json`, `bonus_results.json`)

## Requirements

- `torch`, `torchvision`
- `timm` (DeiT/ViT model zoo)
- `pytorch-grad-cam` (Grad-CAM, Grad-CAM++)
- `lime`, `scikit-image` (LIME segmentation)
- `scikit-learn`, `scipy`, `opencv-python`, `matplotlib`

## How to Run

1. Point `DATASET_ROOT` at the local path of the COVID-19 X-ray dataset (`train/val/test` folder structure).
2. Run cells top to bottom — each explainability/evaluation section depends on model checkpoints saved by the earlier training cells.
3. Outputs (plots, metrics, weights) will be saved under `OUTPUT_DIR`.

## Disclaimer

This is a research/educational exploration of model interpretability on medical imaging data — it is **not** a validated diagnostic tool and should not be used for real clinical decision-making.

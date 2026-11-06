# Solar filament instance segmentation: S4V8

Delivered solution of team JerryMouse. This repository stays private until the competition closes and is made public at the close.

Solution for the Kaggle competition `filament-segmentation-2026`.
- Input: H-alpha full-disk images at 2048 x 2048.
- Output: one binary mask per filament, in the submission CSV format.
- Inference uses only the test H-alpha images.

## Pipeline
1. **Preprocessing.** The image is read as greyscale and resized to 1536. Each model family reads it differently:
   - Mask R-CNN `ep24`: grey x 3 channels.
   - Mask R-CNN `e034`: grey, limb-darkening-flattened and multi-scale black top-hat channels, all computed from the image alone.
   - Dense U-Nets `D1` and `D2`: one grey channel (PIL box resize), standardised per image.
2. **Detector members.** Two Mask R-CNN R50-FPN v2 families (`ep24`, `e034`), each with a ResNet18 U-Net crop refiner.
3. **Dense members.** Two ConvNeXt-T U-Net families (`D1`, `D2`). Instances are 8-connected components of probability above 0.5 with at least 64 pixels.
4. **Test-time augmentation.** Each of the 12 fold models runs on 8 flip and transpose views, giving 96 members.
5. **Fusion.** Instances are clustered at IoU above 0.5. A cluster's fused score is the sum of its member confidences divided by the number of members; clusters below 0.30 are dropped, then NMS and overlap removal.
6. **Rescoring (C2).** A logistic regression on 9 instance features keeps instances scoring at least tau = 0.29.

`params.json` holds the members, views, C2 parameters and tau. The fixed thresholds of fusion and instance extraction are constants in `code/` (for example `e042_rescore.py` FLOOR 0.30, `e056_infer.py` THR 0.5 and MIN_AREA 64). No training labels, out-of-fold data or cached predictions are read at inference.

## Results
| Measure | Value |
|---|---|
| Out-of-fold PQ, model selection estimate (nested halves, mean of 3 seeds) | 0.4613 |
| Out-of-fold PQ, C2 layer left out per fold (`train/fit_c2.py --evaluate`, `docs/evaluation.json`) | 0.4581 (folds 0.4665, 0.4556, 0.4512) |
| Public leaderboard | 0.40 |

How to read the second figure:
- Each image is predicted by the four fold models that did not see its fold, with 32 members per image (deployment uses 96).
- C2 and tau are fitted on two folds' candidates and applied to the third.
- The base models behind those two folds were trained with the evaluated fold included. So this leaves out the C2 layer only; it is not a fully nested cross-validation of the whole pipeline.

Both out-of-fold figures come from development data that we reused many times. S4V8 was also chosen among 14 slimmed variants, so both are optimistic. On our early submissions the public score was 0.06 to 0.07 below the out-of-fold score. Dice, IoU and one-to-many distributions are in `docs/rubric-metrics.md`. Their definitions are ours, because the competition page does not fix them.

## Run (stored weights, no training)
The weights are in `weight_parts/`, split into pieces of at most 90,000,000 bytes.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
python assemble_weights.py
python code/e080_infer.py --images <folder with test jpg> --out submission.csv
```

`assemble_weights.py` uses only the standard library. It checks every part and every rebuilt file against SHA-256 and stops on a mismatch.

`demo.ipynb` runs the whole pipeline in order: preprocessing, optional training, inference with post-processing, evaluation, and pictures. By default it loads the stored weights and does not train.

Speed and memory:
- One RTX 3080 Ti: about 14.4 s per image, 3.77 GiB RAM, 3.48 GiB GPU memory.
- Kaggle T4: 41.3 s per image.

## Training
Everything needed to retrain is in `train/`. The settings of every delivered checkpoint are in `train/provenance.json`: script hash, full arguments, seed, epochs, and the pretrained initial weights with URL (Hugging Face revision for ConvNeXt-T) and SHA-256.

Short forms below; the full arguments of every run are in `train/provenance.json` ("args").

| Step | Script | What it does | Our cost |
|---|---|---|---|
| 1a | `train/maskrcnn_ep24.py --fold F` | Mask R-CNN R50-FPN v2 (COCO init), grey input, 24 epochs; then the ResNet18 U-Net crop refiner (ImageNet init), 8 epochs | about 2 h per fold, 2 x T4 |
| 1b | `train/maskrcnn_e034.py --fold F` | the same with the physics input channels, 10 detector epochs | about 1.3 h per fold, 2 x T4 |
| 1c | `train/dense_unet.py --fold F --size 1536 --micro 2 [--seed 20261028]` | ConvNeXt-T U-Net (ImageNet init), 1536, 30 epochs; seed 20261027 = `D1`, 20261028 = `D2` | about 1 h per fold, RTX 3080 Ti |
| 2 | `train/collect.py weights` | copies the final checkpoints to `weights_retrained/`, writes `params_retrained.json` | seconds |
| 3 | `train/fit_c2.py --params params_retrained.json` then `train/collect.py c2` | out-of-fold candidates of the 12 fold models, C2 logistic and tau | about 1 h, RTX 3080 Ti |

How the folds work:
- Fold models are trained on two folds of `train/folds.csv` and validated on the third.
- Images of the same group never cross folds.
- C2 is fitted only on out-of-fold candidates.

Notes:
- The full commands are in `train/provenance.json` and in section 3 of `demo.ipynb` (`TRAIN = True` runs all of them).
- Internet is needed during training to download the pretrained initial weights.
- On one GPU, add `--nproc 0 --accum 2` to the Mask R-CNN commands (the notebook does this). This runs without DDP and gives the same effective batch as our 2 DDP ranks of batch 1.
- Retrained models will not be bit-identical to the delivered ones, because of GPU type, library versions and DDP.

What has been checked:
- The training scripts in `train/` are byte-identical to the scripts that produced the delivered weights. There are two documented exceptions, listed in `train/provenance.json`:
  - The fold 1 and 2 Mask R-CNN runs used copies that differ only in the default of `--fold`.
  - `D1` was trained by an earlier revision of `dense_unet.py` with the same fixed seed.
- `train/fit_c2.py` run on the delivered weights reproduces the delivered C2 and tau. Agreement: tau equal (0.29); every C2 parameter within 0.0006 of `params.json` (707 images, 6156 candidates).
- A one-epoch smoke run of each training script completed on our machine (Mask R-CNN at 512 px, 4 steps, no DDP; dense at 256 px). A full retrain from these scripts has not been repeated.
- `train/collect.py` accepts the receipts of all 12 delivered runs and refuses a smoke run.

## Reproducibility: what has and has not been checked
- **Checked:** on the development machine, this code and these weights produced a CSV byte-identical to the submitted file.
- **Checked:** reassembling the weights in a fresh copy of this folder gave the same hashes and identical rows on 5 images.
- **Checked:** on Kaggle T4 with no internet, all 180 test images completed and the CSV was valid. Final-layer agreement with the development machine:
  - 1115 vs 1116 rows.
  - 179 of 180 images with equal instance counts.
  - Mean IoU 0.9992.
- **Not achieved:** numeric reproduction across systems. The member-level gate of that run failed and the cause was not located, so expect small differences on another system:
  - Instance-count agreement of the `ep24` detector members 99.21 %.
  - Member IoU p01 0.960.
  - Score difference p99 0.00233.
- **Checked:** a new virtual environment on the development machine (Windows, Python 3.13, RTX 3080 Ti):
  - Installing the 9 direct packages and running all 180 test images gave a CSV byte-identical to the submitted file.
  - `requirements.txt` pins the resolved versions plus `notebook`.
- **Not checked:** this install on another operating system or GPU. The Kaggle run used other versions of numpy, OpenCV and Pillow, not these pins.

## Layout
| Path | Content |
|---|---|
| `demo.ipynb` | End-to-end notebook: preprocessing, optional training, inference, evaluation |
| `code/e080_infer.py` | Inference entry (`--params` selects the delivered or retrained parameters) |
| `code/` | Modules the entry imports, copied unchanged from development |
| `train/` | Training scripts, `folds.csv`, C2 fitting, `provenance.json` |
| `params.json` | Members, views, fusion and C2 parameters |
| `weight_parts/`, `weight_parts.json`, `weights.sha256` | Weight pieces and hashes |
| `docs/` | Rubric metrics and the stored out-of-fold evaluation |

## Licences and attribution
Our code in this repository is released under the MIT licence (`LICENSE`). The competition has no OSI-specific requirement (host, discussion 755907). The delivered weights are our fine-tuned models; the terms of the pretrained initial weights they started from still apply:

Pretrained initial weights (URLs and SHA-256 in `train/provenance.json`):
- torchvision `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1` and `ResNet18_Weights.IMAGENET1K_V1`. torchvision code is BSD-3-Clause; the weights are under the torchvision model terms.
- ConvNeXt-T `convnext_tiny.fb_in22k_ft_in1k` through `timm` (Apache-2.0) and `segmentation-models-pytorch` (MIT).

# Solar filament instance segmentation: S4V8

Delivered solution of team JerryMouse. This repository stays private until the competition closes and is made public at the close.

Solution for the Kaggle competition `filament-segmentation-2026`.
- Input: H-alpha full-disk images at 2048 x 2048.
- Output: one binary mask per filament, in the submission CSV format.
- Inference uses only the test H-alpha images.

## Names used in this repository
The code keeps the short identifiers from our development work. They mean:

| Identifier | Meaning |
|---|---|
| `S4V8` | The delivered ensemble: **4** model families, **8** test-time views each |
| `ep24` | Mask R-CNN family with grey input (24 detector epochs) and its crop refiner |
| `e034` | Mask R-CNN family with physics input channels (10 detector epochs) and its crop refiner; named after the experiment that introduced it |
| `D1`, `D2` | ConvNeXt-T U-Net (dense) family, trained twice with seeds 20261027 and 20261028 |
| `C2` | The learned instance filter: logistic rescoring of fused instances with threshold tau |
| `fold0`-`fold2` | The model trained without that fold of `train/folds.csv` |
| `e006`, `e042`, ... in file names | Experiment numbers of our development log; see `code/README.md` for what each file does |

## Pipeline
1. **Preprocessing.** The image is read as greyscale and resized to 1536. Each model family reads it differently:
   - Mask R-CNN `ep24`: grey x 3 channels.
   - Mask R-CNN `e034`: grey, limb-darkening-flattened and multi-scale black top-hat channels, all computed from the image alone.
   - Dense U-Nets `D1` and `D2`: one grey channel (PIL box resize), standardised per image.
2. **Detector members.** Two Mask R-CNN R50-FPN v2 families (`ep24`, `e034`). Detections with score at least 0.4 go through mask thresholding and NMS, then a ResNet18 U-Net crop refiner redraws each mask from a crop of the original image.
3. **Dense members.** Two ConvNeXt-T U-Net families (`D1`, `D2`, the same architecture with two seeds). Instances are 8-connected components of probability above 0.5 with at least 64 pixels.
4. **Test-time augmentation.** Each of the 12 fold models (4 families x 3 folds) runs on 8 flip and transpose views, giving 96 members. Predictions are mapped back to the original orientation.
5. **Fusion.**
   - All member instances are sorted by score. Each instance joins the first cluster whose anchor (its highest-scoring instance) it overlaps at IoU above 0.5, provided that member is not already in the cluster; otherwise it starts a new cluster.
   - A cluster's fused score is the sum of its member confidences divided by the number of members (96). Clusters below 0.30 are dropped.
   - A cluster's mask keeps the pixels covered by at least half of the cluster's members (`f1_fusion.vote_mask`). A cluster with one member, or whose vote is empty, keeps the mask of its anchor.
   - Clusters are taken in order of fused score. A cluster is skipped when its intersection with an already kept mask covers at least half of the smaller of the two; otherwise the pixels already taken are removed from it. No pixel belongs to two instances.
6. **Rescoring (C2).** A logistic regression on 9 instance features keeps instances scoring at least tau = 0.29. The features are the fused score, vote share (members in the cluster / 96), mean score of the members present, share of families that found the instance, share of the 8 views that found it, log area, contrast with the surroundings, distance of the instance centre from the image centre (pixel 1024, 1024) divided by 1024, and log number of instances in the image. The logistic regression is fitted on out-of-fold candidates with a soft target: the instance's IoU with the ground-truth instances it matches at IoU > 0.5, averaged over the annotation records of the image and clipped to [0, 1]. Tau is the threshold that maximises out-of-fold PQ. The output is a ranking score with a fitted threshold, not a calibrated probability that the instance is correct. For the family share, `D1` and `D2` count as one family, so `params.json` has `"families": 3`.

`params.json` holds the members, views, C2 parameters and tau. The fixed thresholds of fusion and instance extraction are constants in `code/` (for example `e042_rescore.py` FLOOR 0.30, `f1_fusion.py` MATCH 0.5, `e056_infer.py` THR 0.5 and MIN_AREA 64, `s1_candidates.py` NMS and refiner thresholds). No training labels, out-of-fold data or cached predictions are read at inference. `code/README.md` maps the modules.

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

Both out-of-fold figures come from development data that we reused many times. S4V8 was also chosen among 14 slimmed variants, so both are optimistic. On our early submissions the public score was 0.06 to 0.07 below the out-of-fold score; we have not established the cause of that gap. Dice, IoU and one-to-many distributions are in `docs/rubric-metrics.md`. Their definitions are ours, because the competition page does not fix them.

What the parts contribute, out of fold (same protocol, threshold or C2 refitted for each configuration; computed in our development repository after the submission was selected, nothing was changed because of it):

| Configuration | Out-of-fold PQ |
|---|---:|
| Mask R-CNN `ep24` alone, identity view, fused-score threshold | 0.428 |
| `ep24` alone, 8 views, fused-score threshold | 0.443 |
| all four families, 8 views, fused-score threshold | 0.461 |
| all four families, 8 views, C2 (delivered) | 0.461 |

Views and model families carry the gain. C2 adds +0.0006 over a tuned fused-score threshold (90 % paired group bootstrap interval -0.0012 to +0.0023), which is no detectable gain; it was adopted on an earlier three-family development ensemble and kept through later ensemble changes.

## Requirements
- Python 3.13 (the version we developed and tested with).
- One CUDA GPU with at least 4 GiB of memory (peak 3.48 GiB). CPU-only inference is not supported.
- An NVIDIA driver that supports the CUDA 12.8 wheels of PyTorch 2.11.
- About 7 GB of disk: 2.2 GB of weight parts, 2.1 GB of reassembled weights, plus the git history and the data.

## Data
The competition images are not part of this repository. Download them from the competition page (the Kaggle API needs your own credentials and accepted competition rules):

```bash
kaggle competitions download -c filament-segmentation-2026
unzip filament-segmentation-2026.zip -d data
```

The test images are then in `data/MAGFiLO_1.0_Kaggle_2026/test/test_images/*.jpg` (180 images). Training and the out-of-fold evaluation need the folder that holds `train/train_images/` and `train/MAGFiLO_1.0_Annotations_kaggle2026_train.json`, that is `S4V8_TRAIN_DATA=data/MAGFiLO_1.0_Kaggle_2026`.

## Run (stored weights, no training)
The weights are in `weight_parts/`, split into pieces of at most 90,000,000 bytes.

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128
python assemble_weights.py
python code/e080_infer.py --images data/MAGFiLO_1.0_Kaggle_2026/test/test_images --out submission.csv
```

`assemble_weights.py` uses only the standard library. It checks every part and every rebuilt file against SHA-256 and stops on a mismatch.

`demo.ipynb` runs the whole pipeline in order: preprocessing, optional training, inference with post-processing, evaluation, and pictures. It is configured by environment variables read in its first cell; paths are relative to the repository root:

| Variable | Default | Meaning |
|---|---|---|
| `S4V8_IMAGES` | `test_images` | folder with the test images |
| `S4V8_LIMIT` | `3` | number of test images for a quick look; `0` = all 180 (the validator marks a CSV with no instance at all as invalid, which can happen on a very small subset) |
| `S4V8_TRAIN` | `0` | `1` retrains every model (section 3) instead of loading the stored weights |
| `S4V8_TRAIN_DATA` | `train_data` | competition data folder, needed only for training or evaluation |
| `S4V8_EVALUATE` | `0` | `1` recomputes the out-of-fold evaluation (about 1 GPU hour) |

```bash
S4V8_IMAGES=data/MAGFiLO_1.0_Kaggle_2026/test/test_images S4V8_LIMIT=0 jupyter nbconvert --to notebook --execute demo.ipynb --output demo_executed.ipynb
```

## Reproduce the submitted file
The command in "Run", or the notebook with `S4V8_LIMIT=0`, predicts all 180 test images. Our submitted file has:
- 1116 rows, instances on 177 of 180 images;
- SHA-256 `69d5141c7aab0f912fe5ff6544279a702b2a1a1eef60881d6d264935b9c6661f`.

On our development machine (Windows, Python 3.13, RTX 3080 Ti) the output is byte-identical to this file. On other systems expect small numeric differences. On Kaggle T4 GPUs we ran it twice (see "Reproducibility"): the bundle gave 1115 rows, and a fresh clone of this repository executing `demo.ipynb` gave 1116 rows; both match the submitted instances with mean IoU 0.9992.

Time and memory for the 180 images:
- One RTX 3080 Ti: 2586 s (14.4 s per image), 3.77 GiB RAM, 3.48 GiB GPU memory.
- Kaggle T4: 7436 s (41.3 s per image).

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
- The full commands are in `train/provenance.json` and in section 3 of `demo.ipynb` (`S4V8_TRAIN=1` runs all of them).
- Internet is needed during training to download the pretrained initial weights.
- On one GPU, add `--nproc 0 --accum 2` to the Mask R-CNN commands (the notebook does this). This runs without DDP and gives the same effective batch as our 2 DDP ranks of batch 1.
- Retrained models will not be bit-identical to the delivered ones, because of GPU type, library versions and DDP.
- The three training scripts are self-contained files built for Kaggle kernels: each carries copies of its helper modules in an `EMBEDDED` string. They are kept byte-identical to the scripts that produced the weights (hashes in `train/provenance.json`), so they were not tidied.

What has been checked:
- The training scripts in `train/` are byte-identical to the scripts that produced the delivered weights. There are two documented exceptions, listed in `train/provenance.json`:
  - The fold 1 and 2 Mask R-CNN runs used copies that differ only in the default of `--fold`.
  - `D1` was trained by an earlier revision of `dense_unet.py` with the same fixed seed.
- `train/fit_c2.py` run on the delivered weights reproduces the delivered C2 and tau. Agreement: tau equal (0.29); every C2 parameter within 0.0006 of `params.json` (707 images, 6156 candidates).
- A one-epoch smoke run of each training script completed on our machine (Mask R-CNN at 512 px, 4 steps, no DDP; dense at 256 px). A full retrain from these scripts has not been repeated.
- `train/collect.py` accepts the receipts of all 12 delivered runs and refuses our smoke runs. It checks the fold, input mode, seed, image size, epoch counts and a few other recorded settings against `train/provenance.json`; it does not check every argument (for example a reduced number of steps per epoch would pass), so compare the receipts with `provenance.json` when in doubt.

## Reproducibility: what has and has not been checked
- **Checked:** on the development machine, this code and these weights produced a CSV byte-identical to the submitted file.
- **Checked:** reassembling the weights in a fresh copy of this folder gave the same hashes and identical rows on 5 images.
- **Checked (first Kaggle run, inference bundle):** on Kaggle T4 with no internet, all 180 test images completed and the CSV was valid. Final-layer agreement with the development machine:
  - 1115 vs 1116 rows.
  - 179 of 180 images with equal instance counts.
  - Mean IoU 0.9992.
- **Checked (second Kaggle run, this repository):** a fresh `git clone` of this repository (commit 556a959, code identical to the current one) executed `demo.ipynb` unchanged on Kaggle T4 with no internet, all 180 test images, stored weights. No cell error; valid CSV; agreement with the submitted file:
  - 1116 vs 1116 rows, 180 of 180 images with equal instance counts, every instance matched in both directions.
  - Mean IoU 0.9992, IoU p01 0.9944.
  - Inference 49.9 s per image; the whole notebook, including weight assembly and pictures, took 9059 s (about 50.3 s per image). Peak RAM 3.5 GiB, peak GPU memory 3.5 GiB.
  - Scope: `segmentation_models_pytorch` 0.5.0 is not on the Kaggle image, so it was supplied from a copy through `PYTHONPATH`; the other packages were Kaggle's, not the pins in `requirements.txt`. The optional training and evaluation sections were not run there. The outputs saved in `demo.ipynb` come from this run.
- **Not achieved:** numeric reproduction across systems. The member-level gate of the first Kaggle run failed and the cause was not located, so expect small differences on another system:
  - Instance-count agreement of the `ep24` detector members 99.21 %.
  - Member IoU p01 0.960.
  - Score difference p99 0.00233.
- **Checked:** a new virtual environment on the development machine (Windows, Python 3.13, RTX 3080 Ti):
  - Installing the 9 direct packages and running all 180 test images gave a CSV byte-identical to the submitted file.
  - `requirements.txt` pins the versions that install resolved, plus `notebook`.
- **Not checked:** this install on another operating system or GPU. The Kaggle run used other versions of numpy, OpenCV and Pillow, not these pins.

## Layout
| Path | Content |
|---|---|
| `demo.ipynb` | End-to-end notebook: preprocessing, optional training, inference, evaluation |
| `code/e080_infer.py` | Inference entry (`--params` selects the delivered or retrained parameters) |
| `code/` | Modules the entry imports, copied unchanged from development; `code/README.md` says what each one does |
| `train/` | Training scripts, `folds.csv`, C2 fitting, `provenance.json` |
| `params.json` | Members, views, C2 parameters and tau (the fixed fusion thresholds are constants in `code/`) |
| `weight_parts/`, `weight_parts.json`, `weights.sha256` | Weight pieces and hashes |
| `docs/` | Rubric metrics and the stored out-of-fold evaluation |

## Licences and attribution
Our code in this repository is released under the MIT licence (`LICENSE`). The competition has no OSI-specific requirement (host, discussion 755907).

Method source: the Mask R-CNN plus crop-refiner design follows the public Kaggle notebook `nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner`. `code/e006_maskrcnn_refiner.py` is our own re-implementation of that method; no code from the notebook is included.

The delivered weights are our fine-tuned models; the terms of the pretrained initial weights they started from still apply. Pretrained initial weights (URLs and SHA-256 in `train/provenance.json`):
- torchvision `MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1` and `ResNet18_Weights.IMAGENET1K_V1`. torchvision code is BSD-3-Clause; the weights are under the torchvision model terms.
- ConvNeXt-T `convnext_tiny.fb_in22k_ft_in1k` through `timm` (Apache-2.0) and `segmentation-models-pytorch` (MIT).

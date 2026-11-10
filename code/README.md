# Module map

The files in this folder are copied unchanged from our development repository, so that the delivered code is exactly the code that produced the submitted file (byte-identical output on our machine). Their names carry our experiment numbers (`e006`, `e042`, ...), and several contain development scripts (`main()` functions, docstrings that point to development paths such as `preds/`, `runs/` or `decisions.md`). Those parts are not used by the delivered pipeline and need files that are not in this repository; we left them in place rather than edit verified code.

Entry point: `e080_infer.py`. The table lists what the entry uses from each module.

| File | Role in the delivered pipeline | Used by inference |
|---|---|---|
| `e080_infer.py` | Entry: loads `params.json`, checks the weight hashes, runs the 96 members, fuses, computes the 9 C2 features, applies C2 and tau, writes and validates the CSV | yes (entry) |
| `e006_maskrcnn_refiner.py` | Image loading and resizing, physics input channels, Mask R-CNN and refiner model definitions and loading, detector forward pass, refiner crops. Our re-implementation of the method of the public notebook `nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner` | yes |
| `t1_tta.py` | Flip and transpose views for the detector members, mapping instances back to the original orientation | yes |
| `s1_candidates.py` | `pool`: detector instances above score 0.4, mask threshold, NMS, refiner redraw | yes |
| `e056_infer.py` | Dense U-Net input loading, the 8 dihedral views and their inverses, instance extraction (probability > 0.5, 8-connected, at least 64 px) | yes |
| `e056_test_infer.py` | `predict`: dense U-Net forward pass over the views | yes |
| `e047_c2plus.py` | `fuse_any`: clustering of member instances at IoU > 0.5 with fused score; feature helpers | yes |
| `f1_fusion.py` | `iou`, `MATCH` (0.5) and `vote_mask` (pixel-wise majority vote of a cluster) | yes |
| `e042_rescore.py` | `FLOOR` (0.30), `finalize_idx` (majority-vote masks and overlap removal), `area`, `contrast`; also the C2 fitting helpers used by `train/fit_c2.py` (`predict_split`, `best_tau`, `fit_on`) | yes |
| `validate_submission.py` | Checks the output CSV (ids, RLE, empty masks) | yes |
| `a3_postproc.py` | `area`, `subtract` (mask helpers imported by the modules above) | imported |
| `gpu_smoke_audit.py` | `parse_compressed_counts` (RLE parsing for the validator) | imported |
| `pq_metric.py` | NumPy reproduction of the official PQ self-evaluation, used by the C2 fitting and evaluation helpers | imported |
| `s2_rescore.py` | `fit_predict`: the logistic regression behind C2, used when fitting (`train/fit_c2.py`) | imported |
| `a2_error_analysis.py`, `e029_fusion_structure.py`, `g1_group_eval.py` | Development analyses; imported at module level by the files above for a few shared helpers | imported only |

Fixed constants of the pipeline: `s1_candidates.py` (`NMS_T`, `MASK_LEVEL`, `REF_T`), `e056_infer.py` (`THR` 0.5, `MIN_AREA` 64), `f1_fusion.py` (`MATCH` 0.5), `e042_rescore.py` (`FLOOR` 0.30). The fitted values (members, views, C2 weights, tau) are in `../params.json`.

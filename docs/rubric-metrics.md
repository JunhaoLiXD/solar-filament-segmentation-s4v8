# Out-of-fold rubric metrics of S4V8

These are descriptive numbers for the report. They were not used to choose or tune anything. They were computed in our development repository (script `e072_rubric.py`, not shipped here) from the out-of-fold candidates of the S4V8 model selection (the nested-halves evaluation behind the 0.4613 figure in the README).

The competition page lists these distributions but does not fix matching rules, thresholds or weights. **The definitions below are ours, not an official implementation.**

## Definitions
- Unit: one (image, annotation record) pair, the same unit as PQ. Predictions are the instances kept by C2 under nested selection, computed for partition seeds 0, 1 and 2. Tables show seed 0; the last table compares the three seeds.
- Dice: union of all ground-truth instances of the record against the union of the kept predictions.
- IoU: matched IoU of true positives; best IoU of each ground-truth instance (0 if no prediction touches it); best IoU of each prediction.
- Relations: g and p are linked when |g∩p|/|g| ≥ c or |g∩p|/|p| ≥ c, for c = any overlap, 0.1, 0.5. "One-to-many": a ground-truth instance linked to ≥ 2 predictions; "many-to-one": a prediction linked to ≥ 2 ground-truth instances.
- One record (20160810234034Lh, record 3) has two almost identical ground-truth instances; the same prediction has IoU above 0.5 with both and counts as 2 true positives under the official PQ. So TP + FP exceeds the number of predictions by 1.
- These are 3-fold out-of-fold predictions on 707 physical images, not test results. S4V8 was chosen among 14 variants, so the numbers are optimistic.

## Counts (seed 0)
| PQ | SQ | TP | FP | FN | Records without ground truth |
|---:|---:|---:|---:|---:|---:|
| 0.4625 | 0.6846 | 5211 | 2018 | 2988 | 0 |

## Dice per record (seed 0)
| n | mean | p5 | p10 | p25 | p50 | p75 | p90 | p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1154 | 0.660 | 0.372 | 0.476 | 0.596 | 0.695 | 0.766 | 0.814 | 0.839 |

Share per bin of 0.1:

| 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 2.5 % | 0.5 % | 0.5 % | 2.8 % | 5.9 % | 13.3 % | 25.9 % | 33.8 % | 14.6 % | 0.1 % |

## Matched IoU of true positives (seed 0)
| n | mean | p5 | p10 | p25 | p50 | p75 | p90 | p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5211 | 0.685 | 0.536 | 0.562 | 0.623 | 0.690 | 0.749 | 0.792 | 0.815 |

Share per bin of 0.1 (true positives have IoU above 0.5 by definition):

| 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---:|---:|---:|---:|---:|
| 18.8 % | 35.3 % | 37.8 % | 8.1 % | 0.0 % |

## Best IoU of each ground-truth instance (seed 0)
| n | mean | p5 | p10 | p25 | p50 | p75 | p90 | p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8199 | 0.484 | 0.000 | 0.000 | 0.240 | 0.609 | 0.716 | 0.774 | 0.800 |

| 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 22.8 % | 1.4 % | 2.4 % | 4.0 % | 5.9 % | 12.0 % | 22.4 % | 24.0 % | 5.1 % | 0.0 % |

## Best IoU of each prediction (seed 0)
| n | mean | p5 | p10 | p25 | p50 | p75 | p90 | p95 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7228 | 0.550 | 0.000 | 0.000 | 0.461 | 0.640 | 0.727 | 0.780 | 0.804 |

| 0.0–0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | 0.9–1.0 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 12.0 % | 2.0 % | 2.8 % | 4.3 % | 6.7 % | 13.6 % | 25.4 % | 27.3 % | 5.8 % | 0.0 % |

## One-to-many: predictions linked to each ground-truth instance (seed 0)
| Coverage c | n | 0 | 1 | 2 | ≥3 |
|---|---:|---:|---:|---:|---:|
| any | 8199 | 1835 (22.4 %) | 6169 (75.2 %) | 179 (2.2 %) | 16 (0.2 %) |
| 0.1 | 8199 | 1837 (22.4 %) | 6169 (75.2 %) | 177 (2.2 %) | 16 (0.2 %) |
| 0.5 | 8199 | 1856 (22.6 %) | 6156 (75.1 %) | 171 (2.1 %) | 16 (0.2 %) |

## Many-to-one: ground-truth instances linked to each prediction (seed 0)
| Coverage c | n | 0 | 1 | 2 | ≥3 |
|---|---:|---:|---:|---:|---:|
| any | 7228 | 811 (11.2 %) | 6266 (86.7 %) | 144 (2.0 %) | 7 (0.1 %) |
| 0.1 | 7228 | 811 (11.2 %) | 6270 (86.7 %) | 140 (1.9 %) | 7 (0.1 %) |
| 0.5 | 7228 | 814 (11.3 %) | 6287 (87.0 %) | 122 (1.7 %) | 5 (0.1 %) |

## Three C2 partition seeds
| Seed | PQ | mean Dice | Dice p50 | TP IoU p50 | one-to-many (c = 0.1, ≥ 2) | many-to-one (c = 0.1, ≥ 2) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.4625 | 0.660 | 0.695 | 0.690 | 193 | 147 |
| 1 | 0.4600 | 0.654 | 0.692 | 0.690 | 197 | 142 |
| 2 | 0.4614 | 0.655 | 0.693 | 0.691 | 180 | 143 |

The mean PQ of the three seeds, 0.4613, is the model-selection figure in the README.

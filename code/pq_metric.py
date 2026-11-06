"""NumPy reproduction of official self-evaluation notebook cells 3-4.

Source: snapshots/official-self-evaluation/self-evaluation-notebook.ipynb.
This establishes local source semantics only, not online-service equivalence.
Repeated annotator entries reuse physical-image predictions. All qualifying
pairs count, with no one-to-one assignment. Extra-only physical images are
ignored by the source entry construction; zero-GT entries cannot originate
from an annotation-row table. Do not exploit either limitation.
"""
import numpy as np


def overlap_matrix(gt, pred):
    gt, pred = np.asarray(gt), np.asarray(pred)
    if gt.ndim != 3 or pred.ndim != 3 or gt.shape[1:] != pred.shape[1:]:
        raise ValueError('Expected matching (instances,height,width) arrays')
    if not np.isin(gt, [0, 1]).all() or not np.isin(pred, [0, 1]).all():
        raise ValueError('Masks must be binary')
    # Official arrays are float32; preserve that arithmetic rather than uint8
    # matmul (which silently overflows) or a differently rounded float64 path.
    g = gt.astype(np.float32).reshape(len(gt), -1) if len(gt) else np.empty((0, np.prod(gt.shape[1:])), np.float32)
    p = pred.astype(np.float32).reshape(len(pred), -1) if len(pred) else np.empty((0, np.prod(pred.shape[1:])), np.float32)
    intersection = g @ p.T
    union = g.sum(axis=1)[:, None] + p.sum(axis=1)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)


def score_matrices(matrices):
    """Globally aggregate overlap matrices; preserve explicit empty-entry branch."""
    tp, fp, fn, tp_values = 0, 0, 0, []
    for matrix in matrices:
        matrix = np.asarray(matrix)
        if matrix.ndim != 2 or not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():
            raise ValueError('Invalid IoU matrix')
        n_gt, n_pred = matrix.shape
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        hit = matrix > .5
        values = matrix[hit].tolist()
        tp += len(values)
        tp_values.extend(values)
        fp += int((hit.sum(axis=0) == 0).sum())
        fn += int((hit.sum(axis=1) == 0).sum())
    denominator = tp + .5 * fp + .5 * fn
    iou_sum = sum(tp_values)
    return {'pq': iou_sum / denominator if denominator else 0.0, 'tp': tp, 'fp': fp, 'fn': fn, 'tp_iou_sum': iou_sum}


def evaluate_entries(entries, predictions):
    """entries: iterable (annotator-image ID, physical ID, GT stack).

    Caller supplies one entry per annotator-image represented in annotation
    rows, matching official get_overlap_df construction. Repeated physical IDs
    intentionally reuse predictions. Missing predictions become empty stacks.
    """
    matrices, seen, used = [], set(), set()
    for entry_id, physical_id, gt in entries:
        if entry_id in seen:
            raise ValueError('Duplicate annotator-image entry')
        seen.add(entry_id)
        gt = np.asarray(gt)
        if gt.ndim != 3:
            raise ValueError('Invalid GT shape')
        pred = predictions.get(physical_id, np.zeros((0, *gt.shape[1:]), dtype=np.uint8))
        matrices.append(overlap_matrix(gt, pred))
        used.add(physical_id)
    result = score_matrices(matrices)
    result.update(entries=len(seen), ignored_extra_prediction_images=sorted(set(predictions) - used),
                  scope='official local source cells 3-4; online scorer parity unknown')
    return result

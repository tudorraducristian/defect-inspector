"""Score images, measure how well defects are separated, and propose a threshold."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from defect_inspector.preprocessing import IMAGENET_MEAN, IMAGENET_STD, to_model_input
from defect_train.data import read_image


@torch.no_grad()
def score_paths(net, paths, image_size, device):
    """Return one score per image: the highest value of its anomaly map."""
    scores = []
    for path in paths:
        batch = to_model_input(read_image(path), image_size, IMAGENET_MEAN, IMAGENET_STD)
        scores.append(net(torch.from_numpy(batch).to(device)).max().item())
    return np.array(scores)


def image_auroc(good, defect):
    labels = [0] * len(good) + [1] * len(defect)
    return float(roc_auc_score(labels, list(good) + list(defect)))


def rates_at_threshold(good, defect, threshold):
    """A score strictly above the threshold means defect."""
    return {
        "detected_rate": float(np.mean(np.asarray(defect) > threshold)),
        "false_alarm_rate": float(np.mean(np.asarray(good) > threshold)),
    }


def propose_threshold(val_scores, margin=1.1):
    """Place the threshold above every validation score, also when they are negative."""
    highest = float(np.max(val_scores))
    return highest + abs(highest) * (margin - 1.0)


def evaluate(net, splits, image_size, device):
    """Measure on the test images; the threshold comes only from the validation images."""
    threshold = propose_threshold(score_paths(net, splits.val, image_size, device))

    kinds = sorted({kind for _, kind in splits.test})
    scores = {}
    for kind in kinds:
        paths = [path for path, path_kind in splits.test if path_kind == kind]
        scores[kind] = score_paths(net, paths, image_size, device)

    good = scores["good"]
    defect = np.concatenate([scores[kind] for kind in kinds if kind != "good"])
    metrics = {
        "image_auroc": image_auroc(good, defect),
        "per_defect": {kind: image_auroc(good, scores[kind]) for kind in kinds if kind != "good"},
        "threshold": threshold,
        "counts": {
            "train": len(splits.train),
            "val": len(splits.val),
            "test_good": len(good),
            "test_defect": len(defect),
        },
    }
    metrics.update(rates_at_threshold(good, defect, threshold))
    return metrics

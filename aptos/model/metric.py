import torch
from sklearn.metrics import cohen_kappa_score

MIN_LABEL = 0
MAX_LABEL = 4
LABELS = list(range(MAX_LABEL + 1))


def quadratic_weighted_kappa(output, target):
    with torch.no_grad():
        preds = output.squeeze(1).clamp(min=MIN_LABEL, max=MAX_LABEL).round()
        return cohen_kappa_score(
            preds.numpy(),
            target.numpy(),
            labels=LABELS,
            weights='quadratic')

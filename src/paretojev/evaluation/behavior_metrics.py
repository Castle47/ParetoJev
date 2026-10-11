"""Shared weighted evaluator for real KuaiRand behavior experiments."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from ..data.kuairand_27k import BINARY_LABELS


def metrics(y, p, w):
    result = {}
    for j, name in enumerate(BINARY_LABELS):
        truth = np.asarray(y[:, j], dtype='float32'); prob = np.clip(p[:, j].astype('float64'), 1e-7, 1-1e-7)
        ece = 0.0
        for b in range(10):
            mask = (prob >= b/10) & (prob < (b+1)/10)
            if mask.any():
                ece += w[mask].sum()/w.sum() * abs(np.average(truth[mask]-prob[mask], weights=w[mask]))
        both = 0 < truth.sum() < len(truth)
        result[name] = dict(sampled_positives=int(truth.sum()),
            prevalence=float(np.average(truth, weights=w)), mean_prediction=float(np.average(prob, weights=w)),
            auc=float(roc_auc_score(truth, prob, sample_weight=w)) if both else None,
            average_precision=float(average_precision_score(truth, prob, sample_weight=w)) if both else None,
            logloss=float(np.average(-truth*np.log(prob)-(1-truth)*np.log1p(-prob), weights=w)),
            brier=float(np.average((truth-prob)**2, weights=w)), ece10=float(ece))
    return result


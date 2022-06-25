from typing import List

import numpy as np
import sklearn


def silhouette_score(embeddings: np.ndarray, label_assignments: List[int]) -> float:
    """
    Measures the silhouette score of the cluster in embedding space
    1 is the best, -1 is the worst
    refer to: https://en.wikipedia.org/wiki/Silhouette_(clustering)
    """
    assert len(embeddings) == len(label_assignments), "should be of the same length"
    if len(np.unique(label_assignments)) == 1:
        return -1.0  # sklearn impl would fail
    return sklearn.metrics.silhouette_score(embeddings, label_assignments)

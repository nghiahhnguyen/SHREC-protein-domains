import numpy as np                                                                                     
import torch
from skbio.core.distance import DistanceMatrix
from sklearn.metrics.pairwise import euclidean_distances

def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)

def retrieval_precision(x1, x2, y1, y2):
    """ Not done yet """
    dist = torch.from_numpy(euclidean_distances(x1, x2))
    k = 2
    # ignore the smallest distance because that is the query
    top_k_min_dist_idx = torch.topk(dist, k=k+1, largest=False, dim=-1)[1][:, 1:]
    mask = y1.unsqueeze(1) == y2
    for i in range(dist.shape[0]):
        for j in range(dist.shape[0]):
            if mask[i][j] == True:
                flag = False
                for k in range(2):
                    if top_k_min_dist_idx[i][k] == j:
                        flag = True
                if not flag:
                    mask[i][j] = False
    

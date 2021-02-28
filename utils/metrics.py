import numpy as np                                                                                     
import torch
# from skbio.core.distance import DistanceMatrix
from sklearn.metrics import pairwise_distances_chunked
import pandas as pd
from glob import glob
import os.path as osp


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

def distance_matrix_from_file(x_files, y_files):
    # file_list = [(int(osp.basename(f).split['.'][0]), f) for f in glob(test_dir + '/*', recursive=True) if not osp.isdir(f)]
    # file_list = sorted(file_list)
    dist = np.array([(np.norm(np.load(x)-np.load(y)), y_idx) for x_idx, x in x_files for y_idx, y in y_files])


def retrieval_precision(dist, label_dict, k):
    """ Not done yet """
    # dist = torch.tensor([d for d in pairwise_distances_chunked(x1, x2)])
    # ignore the smallest distance during because that is the query
    top_k_min_dist_idx = torch.topk(dist, k=k+1, largest=False, dim=-1)[1][:, 1:]

    result = 
    for rtrv in top_k_min_dist_idx:
        

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
    

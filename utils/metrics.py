import numpy as np
from numpy.testing._private.utils import break_cycles                                                                                     
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

def label_distance_matrix_from_file(x_files, y_files):
    # file_list = [(int(osp.basename(f).split['.'][0]), f) for f in glob(test_dir + '/*', recursive=True) if not osp.isdir(f)]
    # file_list = sorted(file_list)
    label_dist_arr = []
    label_list = []
    for xf in x_files:
        x_label, x = np.load(xf, allow_pickle=True)
        label_list.append(x_label)
        label_dist_list = []
        for yf in y_files:
            y_label, y = np.load(yf, allow_pickle=True)
            label_dist_list.append((y_label, np.linalg.norm(x-y)))

        label_dist_arr.append(np.array(label_dist_list))
    return np.array(label_list), np.array(label_dist_arr)


def retrieval_precision(label_dist_arr, label_arr, k):
    # dist = torch.tensor([d for d in pairwise_distances_chunked(x1, x2)])
    labels, dist = np.dsplit(label_dist_arr, 2)
    labels = labels.squeeze()
    dist = dist.squeeze()
    dist = torch.tensor(dist)
    # ignore the smallest distance during because that is the query

    topk = torch.topk(dist, k=k+1, largest=False, dim=-1)[1][:, 1:]
    topk = topk.numpy()
    topk_labels = np.array([l[mask] for (l, mask) in zip(labels, topk)])
    # topk = np.apply_along_axis(lambda x: label_dict[x+1], axis=1, arr=topk)
    masked_topk = []
    for i in range(topk.shape[0]):
        mask = np.empty(topk.shape[1])
        mask.fill(label_arr[i])
        masked_topk.append(mask)
    masked_topk = np.array(masked_topk)
    masked_topk = (masked_topk == topk_labels)
    
    return np.array([precision_at_k(mask, k) for mask in masked_topk])

def retrieval_success(label_dist_arr, label_arr, k):
    # dist = torch.tensor([d for d in pairwise_distances_chunked(x1, x2)])
    labels, dist = np.dsplit(label_dist_arr, 2)
    labels = labels.squeeze()
    dist = dist.squeeze()
    dist = torch.tensor(dist)
    # ignore the smallest distance during because that is the query

    topk = torch.topk(dist, k=k+1, largest=False, dim=-1)[1][:, 1:]
    topk = topk.numpy()
    topk_labels = np.array([l[mask] for (l, mask) in zip(labels, topk)])
    # topk = np.apply_along_axis(lambda x: label_dict[x+1], axis=1, arr=topk)
    masked_topk = []
    for i in range(topk.shape[0]):
        mask = np.empty(topk.shape[1])
        mask.fill(label_arr[i])
        masked_topk.append(mask)
    masked_topk = np.array(masked_topk)
    masked_topk = (masked_topk == topk_labels)
    
    return np.mean(masked_topk.sum(axis=1))
        

    # mask = y1.unsqueeze(1) == y2
    # for i in range(dist.shape[0]):
    #     for j in range(dist.shape[0]):
    #         if mask[i][j] == True:
    #             flag = False
    #             for k in range(2):
    #                 if top_k_min_dist_idx[i][k] == j:
    #                     flag = True
    #             if not flag:
    #                 mask[i][j] = False
    

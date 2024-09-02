from typing import Dict
import numpy as np
import torch 

from torch import Tensor
from scipy.stats import gaussian_kde 
from sklearn.preprocessing import scale 
from astropy.stats import knuth_bin_width, bayesian_blocks, freedman_bin_width


def bhattacharyya_coef(
        u: np.ndarray, 
        v: np.ndarray, 
        method: str = 'continuous', 
        n_steps: int = 200, 
        n_bins: int = 10) -> float:
    """Computes the bhattacharrya coefficient in a single feature dimension between two classes using different binning techniques.

    Adapted from https://github.com/EricPWilliamson/bhattacharyya-distance/blob/master/bhatta_dist.py

    Args:
        u (np.ndarray): Feature values of one class.
        v (np.ndarray): Feature values of the other class.
        method (str, optional): Method to use for binning the features. Defaults to 'continuous'.
        n_steps (int, optional): Intervals for coninuous binning. Defaults to 200.
        n_bins (int, optional): Number of bins for histogram binning. Defaults to 10.

    Returns:
        float: Bhattacharyya coefficient between distributions of two classes.
    """

    def get_density(x):
        return gaussian_kde(x)
    
    z = np.concatenate((u, v))

    if np.count_nonzero(z) == 0:
        # if all values are zero the coefficient will become zero
        # this would indicate entirely non-overlapping distributions which is false
        # we therefore return 1 to indicate equal distributions
        return 1

    if method == 'noiseless':
        uz = np.unique(z)

        A1 = len(u) * (max(z)-min(z)) / len(z) 
        A2 = len(v) * (max(z)-min(z)) / len(z)

        bht = 0
        for x in uz:
            p1 = (u==x).sum() / (A1 + 1e-10)
            p2 = (v==x).sum() / (A2 + 1e-10)
            bht += np.sqrt(p1*p2) * (max(z)-min(z))/len(z)

    elif method == 'hist':
        h1 = np.histogram(u, bins=n_bins, range=(min(z), max(z)), density=True)[0]
        h2 = np.histogram(v, bins=n_bins, range=(min(z), max(z)), density=True)[0]

        bht = 0
        for i in range(n_bins):
            p1 = h1[i]
            p2 = h2[i]
            bht += np.sqrt(p1*p2) * (max(z)-min(z))/n_bins

    elif method == 'autohist':
        bins = np.histogram(z, bins='doane')[1]

        h1 = np.histogram(u, bins=bins, density=True)[0]
        h2 = np.histogram(v, bins=bins, density=True)[0]

        bht = 0
        for i in range(len(h1)):
            p1 = h1[i]
            p2 = h2[i]
            bht += np.sqrt(p1*p2) * (max(z)-min(z))/len(h1)

    elif method == 'knuth':
        _, bins = knuth_bin_width(z, return_bins=True, quiet=True)

        h1 = np.histogram(u, bins=bins, density=True)[0]
        h2 = np.histogram(v, bins=bins, density=True)[0]

        bht = 0
        for i in range(len(h1)):
            p1 = h1[i]
            p2 = h2[i]
            bht += np.sqrt(p1*p2) * (max(z)-min(z))/len(h1)

    elif method == 'friedmann':
        _, bins = freedman_bin_width(z, return_bins=True)

        h1 = np.histogram(u, bins=bins, density=True)[0]
        h2 = np.histogram(v, bins=bins, density=True)[0]

        bht = 0
        for i in range(len(h1)):
            p1 = h1[i]
            p2 = h2[i]
            bht += np.sqrt(p1*p2) * (max(z)-min(z))/len(h1)

    elif method == 'blocks':
        bins = bayesian_blocks(z)

        h1 = np.histogram(u, bins=bins, density=True)[0]
        h2 = np.histogram(v, bins=bins, density=True)[0]

        bht = 0
        for i in range(len(h1)):
            p1 = h1[i]
            p2 = h2[i]
            bht += np.sqrt(p1*p2) * (max(z)-min(z))/len(h1)

    elif method == 'continuous':
        interval = (max(z) - min(z)) / n_steps
        
        d1 = get_density(u)
        d2 = get_density(v)

        xs = np.linspace(min(z), max(z), n_steps)

        p1 = d1(xs)
        p2 = d2(xs)

        bht = np.sum(np.sqrt(np.multiply(p1, p2))) * interval

    return bht


def select_high_variance_dimensions(
        features: np.ndarray, 
        n: int
        ) -> np.ndarray:
    """Select first n dimensions with highest variance.

    Args:
        features (np.array): Input feature with shape (N, D).
        n (int, optional): The number of dimensions to select.

    Returns:
        np.array: The features with n highest variance (N, n).
    """
    if n > features.shape[1]:
        n = features.shape[1]

    # compute variance along feature dimensions
    variances = np.var(features, axis=0)

    # get indices of variances sorted in descending order
    sort_indices = np.argsort(variances)[::-1]

    # select top indices
    top_indices = sort_indices[:n]

    # select feature dimensions
    reduced_features = features[:, top_indices]

    return reduced_features



def _compute_hellinger_dinstance(
        u: np.ndarray, 
        v: np.ndarray, 
        method: str = 'autohist', 
        distance: bool = True,
        aggregation: str = 'mean',
        n_steps: int = 200, 
        n_bins: int = 10
        ) -> float:
    """Computes hellinger distance over a set of features based on the bhattacharyya coefficient.

    Args:
        u (np.ndarray): Features of one class.
        v (np.ndarray): Features of the other class.
        method (str, optional): Method to use for binning the features. Defaults to 'autohist'.
        distance (bool, optional): If False, returns bhattacharyya coefficient. Defaults to True.
        aggregation (str, optional): Use of mean or maximum aggregation . Defaults to 'mean'.
        n_steps (int, optional): Intervals for coninuous binning. Defaults to 200.
        n_bins (int, optional): Number of bins for histogram binning. Defaults to 10.

    Raises:
        ValueError: If U and V have different length.
        ValueError: If aggregation type is not in [mean, max].

    Returns:
        float: Aggregated hellinger distance (or bhattacharyya coefficient) over a set of features between two classes.
    """
    if u.shape[1] != v.shape[1]:
        raise ValueError(f'Features need to have same dimensions. Got {u.shape[1]} and {v.shape[1]}.')
    D = u.shape[1]
    dist = 0
    for i in range(D):
        coef = bhattacharyya_coef(u[:, i], v[:, i], method, n_steps, n_bins)
        if distance:
            coef = np.sqrt(1 - coef)
        if aggregation == 'mean':
            dist += coef / D
        elif aggregation == 'max':
            dist = np.maximum(dist, coef)
        else:
            raise ValueError(f'Unrecognized aggregation type: {aggregation}.')
    return dist


def hellinger_distance(
        features: np.ndarray, 
        targets: np.ndarray, 
        method: str = 'autohist', 
        distance: bool = True, 
        aggregation: str = 'mean',
        num_dimensions: int = None,
        n_steps: int = 200, 
        n_bins: int = 10, 
        ) -> float :
    """Computes binary or multi-class aggregated hellinger distance over a set of multidimensional feautre vectors. 
 
    Args:
        features (np.ndarray): Multidimensional feature values of shape [N, D].
        targets (np.ndarray): Vector of class labels of shape [N].
        method (str, optional): Method to use for binning the features. Defaults to 'autohist'.
        distance (bool, optional): If False, returns bhattacharyya coefficient. Defaults to True.
        aggregation (str, optional): Use of mean or maximum aggregation . Defaults to 'mean'.
        num_dimensions (int, optional): Selects dimensions with highest variance. Defaults to None.
        n_steps (int, optional): Intervals for coninuous binning. Defaults to 200.
        n_bins (int, optional): Number of bins for histogram binning. Defaults to 10.

    Raises:
        ValueError: If targets contain only a single class.

    Returns:
        float: Aggregated hellinger distance value.
    """
    L = np.unique(targets).tolist()



    if len(L) < 2:
        raise ValueError(f'Need at least two classes!')
    
    # select channels with most variance
    if num_dimensions is not None:
        features = select_high_variance_dimensions(features, num_dimensions)

    # binary HDV
    if len(L) == 2:
        u = features[targets == L[0]]
        v = features[targets == L[1]]
        dist = _compute_hellinger_dinstance(u, v, method, distance, aggregation, n_steps, n_bins)
    
    # mutli-class HDV
    else:
        dist_list = []

        # compute len(L) HDVs
        for idx, l in enumerate(L):
            u = features[targets == l, :]
            v = features[targets != l, :]
            dist_i = _compute_hellinger_dinstance(u, v, method, distance, aggregation, n_steps, n_bins)
            dist_list.append(dist_i)
        
        # aggregate HDVs
        dist = np.mean(dist_list)

    return dist
    
    

def domain_specific_hellinger_distance(
        features_dict: Dict[str, Tensor], 
        domains: Tensor, 
        labels: Tensor, 
        codes: Dict[int, str], 
        method: str = 'autohist', 
        distance: bool = True, 
        aggregation: str = 'mean',
        num_dimensions: int = None,
        decimals: int = 2, 
        n_steps: int = 200, 
        n_bins: int = 20
        ) -> Dict[str, float]:
    """Computes the hellinger distance over multiple sets of layers and domains.

    Args:
        features_dict (Dict[str, Tensor]): Dictionary with features from different layers [str, [N, D]].
        domains (Tensor): Tensor that contains domain labels [N].
        labels (Tensor): Tensor that contains class labels [N]
        codes (Dict[int, str]): Dictionary to match domain labels. 
        method (str, optional): Method to use for binning the features. Defaults to 'autohist'.
        distance (bool, optional): If False, returns bhattacharyya coefficient. Defaults to True.
        aggregation (str, optional): Use of mean or maximum aggregation . Defaults to 'mean'.
        num_dimensions (int, optional): Selects dimensions with highest variance. Defaults to None.
        decimals (int, optional): Number of decimals to round distance values. Defaults to 2.
        n_steps (int, optional): Intervals for coninuous binning. Defaults to 200.
        n_bins (int, optional): Number of bins for histogram binning. Defaults to 10.

    Returns:
        Dict[str, float]: _description_
    """
    all_dist = {}
    
    for layer, features in features_dict.items():
        layer_dist = {}

        for idx, name in codes.items():
            feats = features[(domains == idx).squeeze()]  # [N, D]
            lls = labels[(domains == idx).squeeze()]
            
            dist = hellinger_distance(
                features=feats.numpy(), 
                targets=lls.numpy(), 
                method=method, 
                distance=distance,
                aggregation=aggregation,
                num_dimensions=num_dimensions,
                n_steps=n_steps, 
                n_bins=n_bins
                )
            
            layer_dist[idx] = np.round(dist, decimals=decimals)
        
        all_dist[layer] = layer_dist

    return all_dist

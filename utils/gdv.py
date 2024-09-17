from typing import Dict
import numpy as np

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import scale
from torch import Tensor 



def upper_tri_masking(A: np.ndarray) -> np.ndarray:
    """Returns the values from above the diagonal of square matrix.

    Args:
        A (np.ndarray): Square matrix as numpy array.

    Returns:
        np.ndarray: 1 dimensional array of upper triangle values
    """
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:, None] < r
    return A[mask]


def intra_class_distance(
        u: np.ndarray, 
        metric: str = 'euclidean'
        ) -> float:
    """Calculates the intra class distance values. 

    Args:
        u (np.ndarray): 1-d vector with feature values of the same class.
        metric (str, optional): Distance metric. Defaults to 'euclidean'.

    Returns:
        float: Intra class distance value.
    """
    N = len(u)
    D = pairwise_distances(u, metric=metric)
    D_triu = upper_tri_masking(D)
    mean = 2 / (N*(N-1)) * np.sum(D_triu)
    return mean


def inter_class_distance(
        u: np.ndarray, 
        v: np.ndarray, 
        metric: str = 'euclidean'
        ) -> float:
    """Calculates the inter class distance values.

    Args:
        u (np.ndarray): 1-d vector with feature values of one class.
        v (np.ndarray): 1-d vector with feature values of another class.
        metric (str, optional): Distance metric. Defaults to 'euclidean'.

    Returns:
        float: _description_
    """
    D = pairwise_distances(u, v, metric=metric)
    mean = np.mean(D)
    return mean 


def generalized_discrimination_value(
        features: np.ndarray, 
        targets: np.ndarray, 
        metric: str = 'euclidean'
        ) -> float:
    """Calculates the generalized discrimination value (GDV) over a set of multidimensional feature vectors.

    Source: https://www.sciencedirect.com/science/article/pii/S0893608021001234

    Args:
        features (np.ndarray): Multidimensional feature values of shape [N, D].
        targets (np.ndarray): Vector of class labels of shape [N].
        metric (str, optional): Distance metric. Defaults to 'euclidean'.

    Returns:
        float: The GDV value
    """
    D = features.shape[1]
    L = np.unique(targets).tolist()

    intra_class_dists = 0
    inter_class_dists = 0

    # normalize features along each dimension
    feats = 0.5 * scale(features, axis=0)

    # compute intra class distances 
    for idx, l in enumerate(L):
        u = feats[targets == l, :]
        d = intra_class_distance(u, metric=metric)
        # print(f'{idx} intra {d}')
        intra_class_dists += d / len(L)

    # compute between class distances 
    for idx, l in enumerate(L[:-1]):
        for jdx, m in enumerate(L[1:]):
            if l != m:
                u = feats[targets == l, :]
                v = feats[targets == m, :]
                d = inter_class_distance(u, v, metric=metric)
                inter_class_dists += 2 * d / (len(L) * (len(L) - 1))

    # compute overall gdv 
    gdv = (intra_class_dists - inter_class_dists) / np.sqrt(D)

    return gdv



def domain_specific_gdv(
        features_dict: Dict[str, Tensor], 
        domains: Tensor, 
        labels: Tensor, 
        codes: Dict[int, str], 
        metric: str = 'euclidean', 
        decimals: int = 2
        ) -> Dict[str, float]:
    """Computes the GDV over multiple sets of layers and domains.

    Args:
        features_dict (Dict[str, Tensor]): Dictionary with features from different layers [str, [N, D]].
        domains (Tensor): Tensor that contains domain labels [N].
        labels (Tensor): Tensor that contains class labels [N]
        codes (Dict[int, str]): Dictionary to match domain labels. 
        metric (str, optional): Distance metric. Defaults to 'euclidean'.
        decimals (int, optional): Number of decimals to round distance values. Defaults to 2.

    Returns:
        Dict[str, float]: _description_
    """
    all_gdvs = {}
    
    for layer, features in features_dict.items():
        gdvs = {}
        
        for idx, name in codes.items():

            feats = features[(domains == idx).squeeze()]  # [N, C]
            lls = labels[(domains == idx).squeeze()]
            
            # feats = feats.view(feats.shape[0], -1)   # [N, C]

            gdv = generalized_discrimination_value(feats.numpy(), lls.numpy(), metric=metric)
            gdvs[idx] = np.round(gdv, decimals=decimals)

        all_gdvs[layer] = gdvs

    return all_gdvs





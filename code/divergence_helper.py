import numpy as np


def kl_divergence(p, q):
    """
    Calculates the Kullback-Leibler (KL) Divergence between two probability distributions.
    KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
    
    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.

    Returns:
        float: The KL Divergence.
    """
    # Replace 0s with a small epsilon to avoid division by zero and log(0) errors
    epsilon = 1e-10
    p = np.asarray(p, dtype=float) + epsilon
    q = np.asarray(q, dtype=float) + epsilon
    
    # Normalize the distributions to ensure they sum to 1
    p /= np.sum(p)
    q /= np.sum(q)
    
    return np.sum(p * np.log(p / q))

def js_divergence(p, q):
    """
    Calculates the Jensen-Shannon (JS) Divergence between two probability distributions.
    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
    where M = 0.5 * (P + Q)

    Args:
        p (np.ndarray): The first probability distribution.
        q (np.ndarray): The second probability distribution.
        
    Returns:
        float: The JS Divergence.
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    
    # Normalize the distributions
    p /= np.sum(p)
    q /= np.sum(q)
    
    m = 0.5 * (p + q)
    
    js_div = 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)
    return js_div
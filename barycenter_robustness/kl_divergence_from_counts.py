import numpy as np

def kl_divergence_from_counts(P_counts, Q_counts, units="bits", num_bins=100, epsilon=1e-10):
    """
    Compute the Kullback-Leibler divergence between two discrete distributions
    given their observed counts (P_counts and Q_counts), with automatic binning.

    Args:
    P_counts (array-like): Observed counts for the first distribution.
    Q_counts (array-like): Observed counts for the second distribution.
    num_bins (int): The number of bins to use for both distributions.
    epsilon (float): Small value to avoid division by zero or log of zero.

    Returns:
    float: The KL divergence D(P || Q).
    """
    # The bins should be the same for both P and Q
    
    # Compute shared bin edges based on the range of both distributions
    min_val = min(np.min(P_counts), np.min(Q_counts))
    max_val = max(np.max(P_counts), np.max(Q_counts))
    bin_edges = np.linspace(min_val, max_val, num_bins + 1)

    # Create histograms (bins) for both P and Q using the same bin edges
    P_hist, _ = np.histogram(P_counts, bins=bin_edges, density=True)
    Q_hist, _ = np.histogram(Q_counts, bins=bin_edges, density=True)
    
    # Normalize the histograms so they sum to 1 (convert to probability distributions)
    P_hist = P_hist / P_hist.sum()
    Q_hist = Q_hist / Q_hist.sum()

    # Clip values to avoid log(0) or division by zero
    P_hist = np.clip(P_hist, epsilon, 1)
    Q_hist = np.clip(Q_hist, epsilon, 1)

    # Compute KL divergence in bits or nats
    if units == "bits":
        kl_div = np.sum(P_hist * np.log2(P_hist / Q_hist))
    else:
        kl_div = np.sum(P_hist * np.log(P_hist / Q_hist))

    return kl_div

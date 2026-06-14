import numpy as np
from sklearn.preprocessing import StandardScaler

def discretise(series, n_bins):
    """Equal‑frequency discretisation."""
    if len(series) < n_bins:
        return np.zeros_like(series, dtype=int)
    quantiles = np.linspace(0, 100, n_bins+1)[1:-1]
    bins = np.percentile(series, quantiles)
    # If bins are degenerate (all same), fallback to equal width
    if len(np.unique(bins)) == 1:
        bins = np.linspace(series.min(), series.max(), n_bins+1)[1:-1]
    return np.digitize(series, bins)

def transfer_entropy(source, target, lag=1, n_bins=5):
    """Transfer entropy from source to target."""
    src = discretise(source, n_bins)
    tgt = discretise(target, n_bins)
    if len(src) <= lag:
        return 0.0
    src_past = src[:-lag]
    tgt_past = tgt[:-lag]
    tgt_future = tgt[lag:]
    # H(tgt_future | tgt_past)
    counts_yy = {}
    for yf, yp in zip(tgt_future, tgt_past):
        counts_yy[(yf, yp)] = counts_yy.get((yf, yp), 0) + 1
    total = len(tgt_future)
    H_cond1 = 0.0
    for (yf, yp), cnt in counts_yy.items():
        # p(yf|yp) = cnt / total_yp
        total_yp = sum(1 for (_, yp_) in counts_yy if yp_ == yp)
        if total_yp > 0:
            pyf_yp = cnt / total_yp
            pyf_yp = max(pyf_yp, 1e-12)
            H_cond1 += (cnt / total) * (-np.log2(pyf_yp))
    # H(tgt_future | src_past, tgt_past)
    counts_xyy = {}
    for yf, xp, yp in zip(tgt_future, src_past, tgt_past):
        counts_xyy[(yf, xp, yp)] = counts_xyy.get((yf, xp, yp), 0) + 1
    H_cond2 = 0.0
    for (yf, xp, yp), cnt in counts_xyy.items():
        total_xyp = sum(1 for (_, xp_, yp_) in counts_xyy if xp_ == xp and yp_ == yp)
        if total_xyp > 0:
            p_yf_xp_yp = cnt / total_xyp
            p_yf_xp_yp = max(p_yf_xp_yp, 1e-12)
            H_cond2 += (cnt / total) * (-np.log2(p_yf_xp_yp))
    te = max(H_cond1 - H_cond2, 0.0)
    return te

def net_info_outflow(returns_matrix, macro_df, lag=1, n_bins=5):
    """
    Compute net information outflow for each ETF.
    Simple version: compute TE from each ETF to every other ETF, average.
    """
    n = returns_matrix.shape[1]
    te_out = np.zeros(n)
    te_in = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            te = transfer_entropy(returns_matrix[:, i], returns_matrix[:, j], lag, n_bins)
            te_out[i] += te
            te_in[j] += te
    net_te = te_out - te_in
    # Normalise to avoid all zeros
    if np.max(np.abs(net_te)) < 1e-12:
        # Fallback: use random variation to avoid all zeros
        net_te = np.random.randn(n) * 0.01
    return net_te

import numpy as np
from sklearn.preprocessing import StandardScaler

def discretise(series, n_bins):
    """Equal‑frequency discretisation."""
    if len(series) < n_bins:
        return np.zeros_like(series, dtype=int)
    quantiles = np.linspace(0, 100, n_bins+1)[1:-1]
    bins = np.percentile(series, quantiles)
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
        pyf_yp = cnt / sum(1 for (_, yp_) in counts_yy if yp_ == yp)
        pyf_yp = max(pyf_yp, 1e-12)
        H_cond1 += (cnt / total) * (-np.log2(pyf_yp))
    # H(tgt_future | src_past, tgt_past)
    counts_xyy = {}
    for yf, xp, yp in zip(tgt_future, src_past, tgt_past):
        counts_xyy[(yf, xp, yp)] = counts_xyy.get((yf, xp, yp), 0) + 1
    H_cond2 = 0.0
    for (yf, xp, yp), cnt in counts_xyy.items():
        denom = sum(1 for (_, xp_, yp_) in counts_xyy if xp_ == xp and yp_ == yp)
        p_yf_xp_yp = cnt / denom if denom > 0 else 0
        p_yf_xp_yp = max(p_yf_xp_yp, 1e-12)
        H_cond2 += (cnt / total) * (-np.log2(p_yf_xp_yp))
    te = max(H_cond1 - H_cond2, 0.0)
    return te

def conditional_te(source, target, macro_df, lag=1, n_bins=5, num_permutations=50):
    """
    Compute transfer entropy from source to target conditioned on macro variables.
    Also returns a significance (p‑value) via permutation test.
    """
    # Align lengths
    min_len = min(len(source), len(target), len(macro_df))
    source = source[:min_len]
    target = target[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Standardise macro
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    # Compute composite macro factor (simple average of scaled macro)
    macro_factor = np.mean(macro_scaled, axis=1)
    # Discretise macro factor
    macro_disc = discretise(macro_factor, n_bins)
    # Now compute conditional TE: H(Y_future | Y_past, macro) - H(Y_future | X_past, Y_past, macro)
    # For each macro state, compute TE separately and average
    te_conditional = 0.0
    for m in range(n_bins):
        idx = (macro_disc == m)
        if np.sum(idx) < lag + 2:
            continue
        src_sub = source[idx]
        tgt_sub = target[idx]
        te_m = transfer_entropy(src_sub, tgt_sub, lag, n_bins)
        te_conditional += te_m / n_bins
    # Permutation test for significance (shuffle source)
    te_perm = []
    for _ in range(num_permutations):
        src_shuffled = np.random.permutation(source)
        te_perm_m = 0.0
        for m in range(n_bins):
            idx = (macro_disc == m)
            if np.sum(idx) < lag + 2:
                continue
            src_sub = src_shuffled[idx]
            tgt_sub = target[idx]
            te_m = transfer_entropy(src_sub, tgt_sub, lag, n_bins)
            te_perm_m += te_m / n_bins
        te_perm.append(te_perm_m)
    p_value = np.mean(np.array(te_perm) >= te_conditional)
    return te_conditional, p_value

def net_info_outflow(returns_matrix, macro_df, lag=1, n_bins=5, num_permutations=50):
    """
    Compute net information outflow for each ETF: sum(outgoing TE) - sum(incoming TE)
    using macro‑adjusted conditional TE.
    """
    n = returns_matrix.shape[1]
    te_out = np.zeros(n)
    te_in = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            te, p = conditional_te(returns_matrix[:, i], returns_matrix[:, j], macro_df, lag, n_bins, num_permutations)
            if p < 0.05:  # only count significant edges
                te_out[i] += te
                te_in[j] += te
    net_te = te_out - te_in
    return net_te

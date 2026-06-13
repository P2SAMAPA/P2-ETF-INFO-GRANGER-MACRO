# Information‑Theoretic Granger Causality with Macro‑Adjusted Significance

Computes transfer entropy (non‑linear Granger causality) between ETFs, conditioning on macro variables to remove spurious connections. A permutation test provides significance. The per‑ETF score is net information outflow (outgoing minus incoming TE) – a measure of whether an ETF leads (positive) or follows (negative) others.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Transfer entropy with equal‑frequency discretisation
- Macro conditioning via composite macro factor
- Permutation test for significance (p < 0.05)
- Score = net information outflow (higher = leader)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-info-granger-macro-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (slower due to permutations; reduce `NUM_PERMUTATIONS` for speed)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- Positive net outflow → ETF leads others (potential alpha source).
- Negative net outflow → ETF follows others.

## Requirements

See `requirements.txt`.

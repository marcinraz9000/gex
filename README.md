# Deribit Options Dashboard (Streamlit)

This repo contains a Streamlit app that builds BTC/ETH options analytics:
- Net GEX by strike (+ cumulative)
- Dealer GEX heatmap (Price × IV)
- Greeks by strike
- IV smiles (RR25/BF25)
- **Strike × Expiry matrix** for OI & GEX with auto-filtering and monthly/quarterly grouping
- Presets: KF / Monthlies / Full + user-saved presets

## Local run

```bash
pip install -r requirements.txt
python -m streamlit run deribit_dashboard_matrix_presets_user.py
```

or use the cloud-safe variant:
```bash
python -m streamlit run deribit_dashboard_matrix_presets_user_cloud.py
```

## Deploy to Streamlit Community Cloud

1. **Create a GitHub repo** and push these files:
   - `deribit_dashboard_matrix_presets_user_cloud.py` (recommended)
   - `requirements.txt`
   - `README.md`

2. Go to **share.streamlit.io**, click **New app**, pick your repo, **main branch**, and set **Main file** to:
   - `deribit_dashboard_matrix_presets_user_cloud.py`

3. Click **Deploy**.

### Notes
- No API keys required. Using Deribit public endpoints.
- Cloud storage is ephemeral. Presets are stored at `~/.streamlit/options_presets.json`.

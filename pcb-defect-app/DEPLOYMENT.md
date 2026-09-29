# Deploying to Streamlit Community Cloud

1. Push this folder to a public GitHub repo (`models/*.pt` is gitignored).
2. share.streamlit.io → New app → main file `app.py`, Python 3.11 (Advanced settings).
   `packages.txt` installs `libgl1` for OpenCV.
3. **Getting `best.pt` onto the server** (the app runs in simulated mode without it):
   - `startup.sh` downloads it from `$MODEL_URL`, **but Streamlit Cloud does not execute it**.
     Use it locally/other hosts, or
   - commit the weights with Git LFS (GitHub blocks plain files >100 MB; a ~128 MB `best.pt`
     needs LFS or a smaller export), removing `models/*.pt` from `.gitignore`, or
   - add a download-on-first-run step in `pcbdet/model_utils.py` reading a URL from `st.secrets`.
4. Free tier is CPU-only with ~1 GB RAM; expect ~8–15 s per image and keep batches small.

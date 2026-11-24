# DeepWeeds Streamlit Model Explorer

## 1. Project structure
- `app/app.py` — Streamlit pages and UI
- `src/preprocessing.py` — SVM and CNN inference preprocessing
- `models/` — trained model checkpoints and SVM scaler
- `data/` — split files, labels, saved evaluation reports and sample images

## 2. Setup (Windows PowerShell)
Open PowerShell in this project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 3. Run
```powershell
streamlit run app/app.py
```
Streamlit will print a local URL, usually `http://localhost:8501`.

## 4. Model files
The app expects:
- `models/best_svm.pkl`
- `models/combined_scaler.pkl`
- `models/best_cnn_baseline.keras`
- `models/best_cnn_weighted.keras`

Do not rename one CNN checkpoint as the other. Both supplied checkpoints are included separately.

## 5. Notes
- Upload JPG/JPEG/PNG directly from your own device using the uploader.
- Six supplied sample images are bundled in `data/sample_images/`.
- Performance metrics are loaded from saved evaluation CSV files, not recomputed in the app.
- SVM preprocessing follows the notebook's RGB histogram + GLCM approach. CNN preprocessing resizes RGB images to 256x256 and scales pixels to [0,1].
- CNN softmax values are scores, not guaranteed calibrated probabilities. The SVM does not expose probabilities because it was trained with `probability=False`.
- If TensorFlow installation fails on Windows, use a supported Python/TensorFlow environment (the included requirements are pinned for reproducibility, but TensorFlow platform compatibility varies).

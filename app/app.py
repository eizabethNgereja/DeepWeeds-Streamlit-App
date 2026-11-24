from pathlib import Path
import sys
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import cv2
from PIL import Image
from skimage.feature import graycomatrix, graycoprops
import tensorflow as tf
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.preprocessing import extract_svm_features, preprocess_cnn_image

CLASS_NAMES = [
    "Chinee apple", "Lantana", "Parkinsonia", "Parthenium",
    "Prickly acacia", "Rubber vine", "Siam weed", "Snake weed", "Negative"
]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"

st.set_page_config(page_title="DeepWeeds | Model Explorer", page_icon="🌿", layout="wide")

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp { background: linear-gradient(145deg,#071b16 0%,#0b2820 48%,#102f25 100%); color:#f1f5ef; }
[data-testid="stSidebar"] { background:#091f19; border-right:1px solid #285443; }
h1,h2,h3,p,label,span,div { color:inherit; }
h1 { color:#f3f7f1; letter-spacing:-.035em; }
h2,h3 { color:#e4eee5; }
.smallcap { color:#9cc9a9; font-size:.72rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.lead { color:#c3d4c8; font-size:1rem; line-height:1.6; }
.metric-card { background:#12382b; border:1px solid #2c6049; border-radius:14px; padding:18px 20px; min-height:112px; }
.metric-label { color:#b8d2c0; font-size:.76rem; text-transform:uppercase; letter-spacing:.1em; }
.metric-value { color:#f7faf5; font-size:1.75rem; font-weight:750; margin-top:7px; }
.panel { background:rgba(14,48,36,.78); border:1px solid #2b5945; border-radius:16px; padding:18px; }
div[data-testid="stFileUploader"] { background:#102d23; border:1px dashed #6aa581; border-radius:12px; padding:10px; }
.stButton>button { background:#9acb8a; color:#10251b; border:0; border-radius:9px; font-weight:700; }
.stDownloadButton>button { background:#9acb8a; color:#10251b; border:0; border-radius:9px; font-weight:700; }
[data-testid="stDataFrame"] { background:#f5f7f2; }
hr { border-color:#315a47; }
</style>
""", unsafe_allow_html=True)

def metric_card(label, value, note=""):
    st.markdown(f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div style="color:#b8d2c0;font-size:.8rem">{note}</div></div>', unsafe_allow_html=True)

@st.cache_resource
def load_svm():
    model = joblib.load(MODEL_DIR / "best_svm.pkl")
    scaler = joblib.load(MODEL_DIR / "combined_scaler.pkl")
    return model, scaler

@st.cache_resource
def load_cnn(filename):
    return tf.keras.models.load_model(MODEL_DIR / filename, compile=False)

@st.cache_data
def read_csv(name):
    return pd.read_csv(DATA_DIR / name)

def run_prediction(image_rgb, choice):
    results = {}
    if choice in ("Tuned SVM", "Compare all"):
        svm, scaler = load_svm()
        features = extract_svm_features(image_rgb)
        scaled = scaler.transform(features.reshape(1, -1))
        pred = int(svm.predict(scaled)[0])
        results["Tuned SVM"] = {"pred": pred, "scores": None}
    cnn_files = {
        "Baseline CNN": "best_cnn_baseline.keras",
        "Class-weighted CNN": "best_cnn_weighted.keras"
    }
    selected = list(cnn_files) if choice == "Compare all" else ([choice] if choice in cnn_files else [])
    for label in selected:
        model = load_cnn(cnn_files[label])
        batch = preprocess_cnn_image(image_rgb)[None, ...]
        scores = model.predict(batch, verbose=0)[0]
        results[label] = {"pred": int(np.argmax(scores)), "scores": scores}
    return results

def show_result(label, result):
    name = CLASS_NAMES[result["pred"]]
    if name == "Negative":
        st.success(f"{label}: No weed detected")
    else:
        st.success(f"{label}: {name}")
    if result["scores"] is not None:
        scores = result["scores"]
        st.caption("CNN softmax scores (not calibrated probabilities)")
        score_df = pd.DataFrame({"Class": CLASS_NAMES, "Score": scores}).sort_values("Score", ascending=False)
        st.bar_chart(score_df.set_index("Class")["Score"])
        st.write("Top 3:", " · ".join(f'{r.Class} ({r.Score:.1%})' for r in score_df.head(3).itertuples()))
    else:
        st.info("The SVM was trained with probability=False. It returns a predicted class, not a probability or confidence score.")

st.sidebar.markdown('<div class="smallcap">DeepWeeds / Field study</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigate", ["Overview", "Predict", "Performance", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Image classification · 9 classes · RGB input")

if page == "Overview":
    st.markdown('<div class="smallcap">Project dashboard</div>', unsafe_allow_html=True)
    st.title("DeepWeeds Model Explorer")
    st.markdown('<p class="lead">A practical interface for exploring classical machine learning and convolutional neural networks on the imbalanced DeepWeeds dataset.</p>', unsafe_allow_html=True)
    labels = read_csv("labels.csv")
    train = read_csv("train_subset0.csv")
    val = read_csv("val_subset0.csv")
    test = read_csv("test_subset0.csv")
    a,b,c,d = st.columns(4)
    for col,lab,valx,note in [(a,"Total images",f"{len(labels):,}","Dataset metadata"),(b,"Image dimensions","256 × 256","RGB images"),(c,"Classes","9","8 weed classes + Negative"),(d,"Negative class",f'{(labels["Species"]=="Negative").sum():,}',"Largest class")]:
        with col: metric_card(lab,valx,note)
    st.markdown("### Official split")
    x,y,z = st.columns(3)
    for col,lab,valx in [(x,"Training",len(train)),(y,"Validation",len(val)),(z,"Testing",len(test))]:
        with col: metric_card(lab,f"{valx:,}","Images")
    st.markdown("### Class distribution")
    counts = labels["Species"].value_counts().reindex(CLASS_NAMES).fillna(0).astype(int)
    st.bar_chart(counts)
    ratio = counts.max()/counts.min()
    st.markdown(f'<div class="panel"><b>Why imbalance matters</b><br><span class="lead">The largest class is {ratio:.1f} times the smallest class. A model can achieve a reasonable overall accuracy while still missing less-represented weed species, so macro-F1, per-class recall and the confusion matrix are important alongside accuracy.</span></div>', unsafe_allow_html=True)

elif page == "Predict":
    st.markdown('<div class="smallcap">Interactive inference</div>', unsafe_allow_html=True)
    st.title("Predict an image")
    st.markdown('<p class="lead">Upload a field image or choose one of the six supplied examples. The same image is passed through the selected model pipeline.</p>', unsafe_allow_html=True)
    model_choice = st.selectbox("Model", ["Tuned SVM", "Baseline CNN", "Class-weighted CNN", "Compare all"])
    samples_dir = DATA_DIR/"sample_images"
    sample_names = sorted([p.name for p in samples_dir.glob("*.jpg")])
    source = st.radio("Image source", ["Upload from my device", "Try a sample image"], horizontal=True)
    image_rgb = None
    if source == "Upload from my device":
        uploaded = st.file_uploader("Choose a JPG or PNG image", type=["jpg","jpeg","png"])
        if uploaded:
            try:
                image_rgb = np.array(Image.open(uploaded).convert("RGB"))
            except Exception as exc:
                st.error(f"Could not read this image. Please upload a valid JPG or PNG. Details: {exc}")
    else:
        if sample_names:
            chosen = st.selectbox("Sample image", sample_names)
            image_rgb = np.array(Image.open(samples_dir/chosen).convert("RGB"))
            st.caption(f"Sample file: {chosen}")
        else:
            st.warning("No sample images are installed.")
    if image_rgb is not None:
        st.image(image_rgb, caption="Input image", width=360)
        if st.button("Run prediction", type="primary"):
            try:
                with st.spinner("Running model inference..."):
                    output = run_prediction(image_rgb, model_choice)
                cols = st.columns(len(output))
                for col,(label,result) in zip(cols, output.items()):
                    with col:
                        st.markdown(f"#### {label}")
                        show_result(label,result)
            except FileNotFoundError as exc:
                st.error(f"Required model file is missing: {exc}")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")

elif page == "Performance":
    st.markdown('<div class="smallcap">Held-out test evaluation</div>', unsafe_allow_html=True)
    st.title("Model performance")
    st.markdown('<p class="lead">Metrics below are loaded from the saved test-result CSV files produced by the evaluation workflow. Macro-F1 is emphasized because it weights all classes equally.</p>', unsafe_allow_html=True)
    comparison = read_csv("final_model_comparison.csv")
    comparison = comparison.sort_values("Macro F1", ascending=False)
    st.dataframe(comparison.style.format({c:"{:.3f}" for c in comparison.columns if c!="Model"}), use_container_width=True, hide_index=True)
    best = comparison.iloc[0]
    st.markdown(f'<div class="panel"><div class="smallcap">Highest macro-F1 in saved test results</div><h2>{best["Model"]}</h2><div style="font-size:2rem;font-weight:800;color:#d6efc9">{best["Macro F1"]:.3f}</div><span class="lead">Accuracy: {best["Accuracy"]:.1%} · Balanced accuracy: {best["Balanced Accuracy"]:.1%}</span></div>', unsafe_allow_html=True)
    reports = {
        "Baseline SVM":"baseline_svm_test_report.csv",
        "Class-weighted SVM":"class-weighted_svm_test_report.csv",
        "Tuned SVM":"tuned_svm_test_report.csv",
        "Baseline CNN":"baseline_cnn_test_report.csv",
        "Class-weighted CNN":"class-weighted_cnn_test_report.csv"
    }
    chosen_report = st.selectbox("Per-class report", list(reports))
    report = read_csv(reports[chosen_report]).rename(columns={"Unnamed: 0":"Class"})
    st.dataframe(report, use_container_width=True, hide_index=True)
    st.markdown("### Confusion matrices")
    matrices = [
        ("Tuned SVM","cm_cell43_0.png"),
        ("Baseline SVM","cm_cell44_0.png"),
        ("Class-weighted SVM","cm_cell44_1.png"),
        ("Baseline CNN","cm_cell44_2.png"),
        ("Class-weighted CNN","cm_cell44_3.png")
    ]
    for start in range(0,len(matrices),2):
        cols=st.columns(2)
        for col,(name,fn) in zip(cols,matrices[start:start+2]):
            with col:
                st.markdown(f"**{name}**")
                p=RESULTS_DIR/fn
                if p.exists(): st.image(str(p), use_container_width=True)
                else: st.caption(f"Matrix not found: {fn}")
    errors = read_csv("weed_detection_comparison.csv")
    st.markdown("### Weed / Negative-class errors")
    st.dataframe(errors, use_container_width=True, hide_index=True)
    st.download_button("Download model comparison CSV", comparison.to_csv(index=False).encode(), "model_comparison.csv", "text/csv")

else:
    st.markdown('<div class="smallcap">Project information</div>', unsafe_allow_html=True)
    st.title("About this project")
    st.markdown("""
    ### Dataset
    DeepWeeds contains 17,509 field images across eight weed species and a Negative class. Images are processed as 256 × 256 RGB inputs for the CNN.

    ### Models
    - **SVM:** 48 normalized RGB histogram features (16 bins per channel) and four GLCM texture features, standardized using the training-set scaler; RBF SVM.
    - **CNN:** Three convolutional blocks (32, 64 and 128 filters), max pooling, global average pooling, a 128-unit dense layer, dropout and a nine-class softmax output.
    - **Class weighting:** Used to increase the influence of under-represented classes during training.

    ### Important interpretation notes
    - The SVM checkpoint was trained with `probability=False`; no SVM confidence score is displayed.
    - CNN softmax outputs are model scores and are not necessarily calibrated probabilities.
    - A single uploaded image is an inference demonstration, not evidence of field-level reliability.
    - Predictions may be affected by lighting, background, image quality and class imbalance. This tool is an academic prototype, not a replacement for expert identification.

    ### Dataset reference
    Olsen, A., et al. (2019). DeepWeeds: A multiclass weed species image dataset for deep learning. *Scientific Reports, 9*, 2058. https://doi.org/10.1038/s41598-018-38343-3
    """)

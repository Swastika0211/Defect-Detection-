"""
Industrial Defect Detection System
====================================
Dataset  : Casting Product Quality Inspection (Kaggle)
Model    : MobileNetV2 Transfer Learning (TensorFlow/Keras)
Frontend : Streamlit

Run locally:
    streamlit run app.py

Deploy:
    Push to GitHub → deploy on share.streamlit.io
"""

import os
import time
import io
import requests

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm_mpl
import streamlit as st
import tensorflow as tf
from PIL import Image

# ─── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title = "Industrial Defect Detector",
    page_icon  = "🔍",
    layout     = "wide",
    initial_sidebar_state = "expanded"
)

# ─── Constants ───────────────────────────────────────────────────────────────
IMG_SIZE    = (224, 224)
IMG_SHAPE   = (224, 224, 3)
MODEL_PATH  = "models/casting_defect_model.h5"
# Hugging Face Hub model URL (upload your model there for cloud deployment)
MODEL_HF_URL = os.getenv("MODEL_URL", "")   # set in Streamlit secrets

CLASSES     = {0: "Defective ❌", 1: "Good ✅"}
COLORS      = {0: "#E24B4A",      1: "#1D9E75"}
CONF_LEVELS = {
    "high":   (0.90, "High confidence — reliable prediction"),
    "medium": (0.75, "Moderate confidence — consider re-inspection"),
    "low":    (0.00, "Low confidence — manual inspection required"),
}

# ─── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e0e0e0;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 28px; }
    .metric-card p  { margin: 4px 0 0; color: #666; font-size: 13px; }
    .result-good    { color: #1D9E75; font-size: 28px; font-weight: bold; }
    .result-bad     { color: #E24B4A; font-size: 28px; font-weight: bold; }
    .info-box {
        background: #e8f4fd;
        border-left: 4px solid #378ADD;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Model Download (for Streamlit Cloud) ───────────────────────────────────
def download_model_if_needed():
    """Downloads model from HuggingFace Hub if not present locally."""
    if os.path.exists(MODEL_PATH):
        return True
    if not MODEL_HF_URL:
        return False
    os.makedirs("models", exist_ok=True)
    with st.spinner("Downloading model from cloud storage (~50 MB)..."):
        try:
            r = requests.get(MODEL_HF_URL, stream=True, timeout=120)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
            return True
        except Exception as e:
            st.error(f"Model download failed: {e}")
            return False

# ─── Model Loading ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    if not download_model_if_needed():
        st.error("Model file not found. Please check MODEL_PATH or MODEL_URL.")
        st.stop()
    return tf.keras.models.load_model(MODEL_PATH)

# ─── Preprocessing ───────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    """Resize → RGB → normalize → add batch dim."""
    img = image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict_image(model, image: Image.Image, threshold: float = 0.5) -> dict:
    """Run inference and return structured result dict."""
    arr  = preprocess(image)
    prob = float(model.predict(arr, verbose=0)[0][0])

    class_idx  = 1 if prob > threshold else 0
    confidence = prob if class_idx == 1 else (1.0 - prob)

    return {
        "class_idx"  : class_idx,
        "label"      : CLASSES[class_idx],
        "color"      : COLORS[class_idx],
        "confidence" : confidence,
        "good_prob"  : prob,
        "defect_prob": 1.0 - prob,
        "raw_score"  : prob,
    }

# ─── Grad-CAM ────────────────────────────────────────────────────────────────
def get_gradcam(model, img_array: np.ndarray) -> np.ndarray:
    """
    Generates a Grad-CAM heatmap showing which image regions
    drove the model's prediction.
    """
    try:
        # Find last conv layer in MobileNetV2
        last_conv = next(
            l for l in reversed(model.layers)
            if isinstance(l, tf.keras.layers.Conv2D)
        ).name
    except StopIteration:
        return None

    grad_model = tf.keras.Model(
        model.inputs,
        [model.get_layer(last_conv).output, model.output]
    )
    img_tensor = tf.cast(img_array, tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        pred_idx = tf.argmax(preds[0])
        loss     = preds[:, pred_idx]

    grads   = tape.gradient(loss, conv_out)
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis])
    heatmap = tf.maximum(heatmap, 0)
    mx      = tf.reduce_max(heatmap)
    heatmap = (heatmap / (mx + 1e-8)).numpy()

    heatmap_resized = cv2.resize(heatmap, IMG_SIZE)
    heatmap_uint8   = np.uint8(255 * heatmap_resized)
    heatmap_color   = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb     = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return heatmap_rgb

def overlay_heatmap(original: np.ndarray, heatmap: np.ndarray, alpha=0.45) -> np.ndarray:
    """Blend Grad-CAM heatmap with original image."""
    if heatmap is None:
        return original
    overlay = alpha * (heatmap / 255.0) + (1 - alpha) * (original / 255.0)
    return np.clip(overlay * 255, 0, 255).astype(np.uint8)

def render_gradcam_figure(image: Image.Image, model) -> plt.Figure:
    """Returns a matplotlib figure with original / heatmap / overlay."""
    arr     = np.array(image.convert("RGB").resize(IMG_SIZE))
    inp     = preprocess(image)
    heatmap = get_gradcam(model, inp)
    overlay = overlay_heatmap(arr, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("none")
    for ax, img, title in zip(axes,
                               [arr, heatmap, overlay],
                               ["Original", "Grad-CAM heatmap", "Overlay"]):
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(title, fontsize=11, fontweight="bold")
    fig.suptitle("Model Attention Map — where it looks for defects",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

# ─── Confidence Badge ────────────────────────────────────────────────────────
def confidence_badge(confidence: float) -> str:
    if confidence >= 0.90:
        return "🟢 High confidence"
    elif confidence >= 0.75:
        return "🟡 Moderate confidence"
    else:
        return "🔴 Low confidence — manual check recommended"

# ─── Sidebar ─────────────────────────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/factory.png", width=70)
        st.title("⚙️ Settings")

        threshold = st.slider(
            "Decision threshold",
            min_value=0.10, max_value=0.90, value=0.50, step=0.05,
            help="Lower = more sensitive to defects (fewer missed, more false alarms)"
        )
        show_gradcam = st.toggle("Show Grad-CAM heatmap", value=True,
                                  help="Visualize where the model focuses")

        st.divider()
        st.subheader("📊 Model Info")
        st.markdown("""
        | Parameter | Value |
        |-----------|-------|
        | Architecture | MobileNetV2 |
        | Input size | 224 × 224 |
        | Dataset | Casting (Kaggle) |
        | Classes | Good / Defective |
        | Strategy | Transfer learning |
        """)

        st.divider()
        st.subheader("📁 Dataset")
        st.markdown("""
        **Casting Product Quality Inspection**
        - ~7,000 real industrial images
        - Submersible pump impellers
        - Binary: ok_front / def_front
        - [View on Kaggle](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product)
        """)

        st.divider()
        st.caption("Built with TensorFlow · OpenCV · Streamlit")

    return threshold, show_gradcam

# ─── Tab 1: Single Image Upload ──────────────────────────────────────────────
def tab_single_image(model, threshold, show_gradcam):
    st.subheader("Upload a product image for inspection")
    st.markdown('<div class="info-box">Supported formats: JPG, JPEG, PNG, BMP · '
                'Optimal input: 300×300 px casting images</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Drop an image here or click to browse",
        type=["jpg", "jpeg", "png", "bmp"],
        key="single_upload"
    )

    if not uploaded:
        # Show example placeholder
        st.markdown("---")
        st.info("Upload a casting product image above to begin inspection.")
        return

    image = Image.open(uploaded)

    with st.spinner("🔍 Analysing image..."):
        time.sleep(0.2)
        result = predict_image(model, image, threshold)

    # ── Layout: image | result | probabilities
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.image(image, caption="Uploaded image", use_column_width=True)

    with col2:
        st.markdown("### Inspection Result")
        css_class = "result-good" if result["class_idx"] == 1 else "result-bad"
        st.markdown(f'<p class="{css_class}">{result["label"]}</p>',
                    unsafe_allow_html=True)

        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.progress(result["confidence"])
        st.markdown(f"**{confidence_badge(result['confidence'])}**")

        st.markdown("---")
        action = (
            "✅ Pass — item can proceed to next stage."
            if result["class_idx"] == 1
            else "🚨 Fail — remove item from production line."
        )
        st.markdown(f"**Recommended action:** {action}")

    with col3:
        st.markdown("### Probability breakdown")
        st.metric("Good probability",      f"{result['good_prob']:.1%}")
        st.metric("Defective probability", f"{result['defect_prob']:.1%}")

        st.markdown("**Good:**")
        st.progress(float(result["good_prob"]))
        st.markdown("**Defective:**")
        st.progress(float(result["defect_prob"]))

        st.markdown(f"*Decision threshold: {threshold:.0%}*")

    # ── Grad-CAM
    if show_gradcam:
        st.markdown("---")
        st.subheader("🔬 Grad-CAM — Model attention map")
        with st.spinner("Generating heatmap..."):
            fig = render_gradcam_figure(image, model)
        st.pyplot(fig, use_container_width=True)
        st.caption("Red/yellow regions = areas that most influenced the decision. "
                   "Use this to verify the model is looking at the product, not background.")

# ─── Tab 2: Camera Capture ───────────────────────────────────────────────────
def tab_camera(model, threshold, show_gradcam):
    st.subheader("Capture from your camera")
    st.markdown('<div class="info-box">Point your camera at a casting part and '
                'click "Take photo" to inspect it instantly.</div>',
                unsafe_allow_html=True)

    camera_img = st.camera_input("Take a photo")

    if not camera_img:
        return

    image = Image.open(camera_img)
    with st.spinner("Analysing..."):
        result = predict_image(model, image, threshold)

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Captured image", use_column_width=True)
    with col2:
        st.markdown("### Live inspection result")
        css_class = "result-good" if result["class_idx"] == 1 else "result-bad"
        st.markdown(f'<p class="{css_class}">{result["label"]}</p>',
                    unsafe_allow_html=True)
        st.metric("Confidence", f"{result['confidence']:.1%}")
        st.progress(result["confidence"])
        st.markdown(confidence_badge(result["confidence"]))

    if show_gradcam:
        st.markdown("---")
        with st.spinner("Generating heatmap..."):
            fig = render_gradcam_figure(image, model)
        st.pyplot(fig, use_container_width=True)

# ─── Tab 3: Batch Analysis ───────────────────────────────────────────────────
def tab_batch(model, threshold):
    st.subheader("Batch quality inspection")
    st.markdown('<div class="info-box">Upload multiple product images at once. '
                'Results are downloadable as CSV.</div>', unsafe_allow_html=True)

    files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if not files:
        return

    # ── Summary metrics placeholder
    st.markdown(f"**{len(files)} images queued for inspection**")

    prog  = st.progress(0, text="Starting batch inspection...")
    rows  = []
    total = len(files)

    # Process each image
    status_container = st.empty()
    for i, f in enumerate(files, 1):
        try:
            img    = Image.open(f)
            result = predict_image(model, img, threshold)
            rows.append({
                "Filename"    : f.name,
                "Result"      : "Good" if result["class_idx"] == 1 else "Defective",
                "Confidence"  : f"{result['confidence']:.1%}",
                "Good prob"   : f"{result['good_prob']:.1%}",
                "Defect prob" : f"{result['defect_prob']:.1%}",
                "Status"      : "PASS" if result["class_idx"] == 1 else "FAIL",
            })
        except Exception as e:
            rows.append({"Filename": f.name, "Result": "Error", "Status": str(e)})
        prog.progress(i / total, text=f"Inspecting {i}/{total}: {f.name}")

    prog.empty()

    df           = pd.DataFrame(rows)
    good_count   = (df["Status"] == "PASS").sum()
    defect_count = (df["Status"] == "FAIL").sum()
    error_count  = total - good_count - defect_count
    pass_rate    = good_count / total * 100

    # ── KPI row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total inspected", total)
    c2.metric("Passed", good_count,   delta=f"{pass_rate:.1f}%")
    c3.metric("Failed", defect_count, delta=f"-{defect_count/total*100:.1f}%",
               delta_color="inverse")
    c4.metric("Pass rate", f"{pass_rate:.1f}%")

    # ── Pie chart
    if good_count + defect_count > 0:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.pie([good_count, defect_count],
               labels=["Good", "Defective"],
               colors=["#1D9E75", "#E24B4A"],
               autopct="%1.1f%%", startangle=90,
               wedgeprops={"edgecolor": "white", "linewidth": 2})
        ax.set_title("Batch result breakdown", fontweight="bold")
        st.pyplot(fig, use_container_width=False)

    # ── Results table
    st.markdown("### Inspection results")
    st.dataframe(
        df.style.map(
            lambda v: "background-color: #d4edda" if v == "PASS"
                      else "background-color: #f8d7da" if v == "FAIL"
                      else "",
            subset=["Status"]
        ),
        use_container_width=True,
        height=400
    )

    # ── Download
    csv = df.to_csv(index=False)
    st.download_button(
        "⬇️ Download full report (CSV)",
        data=csv,
        file_name="defect_inspection_report.csv",
        mime="text/csv",
        use_container_width=True
    )

# ─── Tab 4: Model Info ───────────────────────────────────────────────────────
def tab_model_info(model):
    st.subheader("Model architecture & training details")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### Architecture
        ```
        Input (224×224×3)
             ↓
        MobileNetV2 backbone
        (pretrained on ImageNet)
             ↓
        GlobalAveragePooling2D
             ↓
        Dense(512, relu)
        BatchNormalization
        Dropout(0.4)
             ↓
        Dense(128, relu)
        Dropout(0.3)
             ↓
        Dense(1, sigmoid)  → probability
        ```

        ### Training strategy
        **Phase 1** — Frozen base (20 epochs, LR=1e-3)
        Trains only the custom classifier head.
        Lets the head adapt to the new domain.

        **Phase 2** — Fine-tune last 30 layers (30 epochs, LR=1e-4)
        Unfreezes and fine-tunes upper MobileNetV2 layers.
        Lower LR prevents destroying pretrained weights.
        """)

    with col2:
        st.markdown("""
        ### Key hyperparameters
        | Parameter | Value |
        |-----------|-------|
        | Input size | 224 × 224 × 3 |
        | Optimizer | Adam |
        | Loss | Binary crossentropy |
        | Phase 1 LR | 1e-3 |
        | Phase 2 LR | 1e-4 |
        | Batch size | 32 |
        | Val split | 15% |
        | Early stopping | patience=8 |

        ### Augmentation (train only)
        - Rotation ±20°
        - Horizontal & vertical flip
        - Zoom ±20%
        - Brightness ×0.8–1.2
        - Width/height shift ±15%
        - Shear ±10°

        ### Class imbalance handling
        Computed class weights passed to `model.fit`.
        This prevents the model from learning
        "predict good always" on a skewed dataset.
        """)

    # Live model summary
    with st.expander("View model layer summary"):
        layers_data = []
        for layer in model.layers:
            params = layer.count_params()
            layers_data.append({
                "Layer": layer.name,
                "Type": layer.__class__.__name__,
                "Output shape": str(layer.output_shape),
                "Parameters": f"{params:,}"
            })
        st.dataframe(pd.DataFrame(layers_data), use_container_width=True, height=300)

    total_params = model.count_params()
    st.metric("Total parameters", f"{total_params:,}")

# ─── Main App ────────────────────────────────────────────────────────────────
def main():
    # Header
    st.markdown("""
    <div style="text-align:center; padding: 10px 0 20px;">
        <h1 style="font-size:2.2rem; margin-bottom:4px;">
            🏭 Industrial Defect Detection System
        </h1>
        <p style="color:#555; font-size:1.05rem; margin:0;">
            AI-powered quality control · MobileNetV2 Transfer Learning · Real-time inference
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar
    threshold, show_gradcam = render_sidebar()

    # Load model
    with st.spinner("Loading AI model..."):
        model = load_model()

    # Top KPI bar
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.markdown('<div class="metric-card"><h3>MobileNetV2</h3><p>Architecture</p></div>',
                unsafe_allow_html=True)
    k2.markdown('<div class="metric-card"><h3>224×224</h3><p>Input size</p></div>',
                unsafe_allow_html=True)
    k3.markdown('<div class="metric-card"><h3>~95%</h3><p>Test accuracy</p></div>',
                unsafe_allow_html=True)
    k4.markdown('<div class="metric-card"><h3>~3.5M</h3><p>Parameters</p></div>',
                unsafe_allow_html=True)
    k5.markdown('<div class="metric-card"><h3>&lt;200ms</h3><p>Inference time</p></div>',
                unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📷 Upload image",
        "📸 Camera capture",
        "📦 Batch inspection",
        "📈 Model info"
    ])

    with tab1:
        tab_single_image(model, threshold, show_gradcam)
    with tab2:
        tab_camera(model, threshold, show_gradcam)
    with tab3:
        tab_batch(model, threshold)
    with tab4:
        tab_model_info(model)


if __name__ == "__main__":
    main()

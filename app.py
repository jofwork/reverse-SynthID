"""
reverse-SynthID Web Interface
Compatible with Streamlit 1.56+
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "src" / "extraction"))

st.set_page_config(
    page_title="SynthID Tool",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .result-box {
        background: #f0f9ff;
        border-left: 4px solid #0ea5e9;
        padding: 1rem 1.2rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .stButton>button { width: 100%; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading codebook…")
def load_bypass(codebook_path: str):
    from synthid_bypass import SynthIDBypass, SpectralCodebook
    cb = SpectralCodebook()
    cb.load(codebook_path)
    bypass = SynthIDBypass()
    return bypass, cb


@st.cache_resource(show_spinner="Loading detector…")
def load_detector(codebook_path: str):
    from robust_extractor import RobustSynthIDExtractor
    extractor = RobustSynthIDExtractor()
    extractor.load_codebook(codebook_path)
    return extractor


def pil_to_rgb_array(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def array_to_pil(arr: np.ndarray) -> Image.Image:
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def img_to_bytes(img: Image.Image, fmt="PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def make_zip(results: list) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in results:
            zf.writestr(name, data)
    return buf.getvalue()


def get_result_image(result):
    for attr in ['cleaned_image', 'image', 'output', 'bypassed', 'result']:
        val = getattr(result, attr, None)
        if val is not None:
            return val
    raise AttributeError(
        f"Cannot find image in result. Available: {[a for a in dir(result) if not a.startswith('_')]}"
    )


# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown("---")

    mode = st.radio("Mode", ["🛡️ Remove Watermark", "🔍 Detect Watermark"], index=0)

    st.markdown("---")
    st.subheader("Codebook paths")

    bypass_cb_path = st.text_input(
        "Bypass codebook (.npz)",
        value="artifacts/spectral_codebook_v3.npz",
    )
    detect_cb_path = st.text_input(
        "Detection codebook (.pkl)",
        value="artifacts/codebook/robust_codebook.pkl",
    )

    if mode == "🛡️ Remove Watermark":
        st.markdown("---")
        st.subheader("Bypass options")
        strength = st.select_slider(
            "Strength",
            options=["gentle", "moderate", "aggressive", "maximum"],
            value="aggressive",
        )
        output_format = st.selectbox("Output format", ["PNG", "JPEG"], index=0)

    st.markdown("---")
    st.caption("reverse-SynthID · research use only")


# Main
st.title("🔍 SynthID Watermark Tool")
st.caption("Upload Gemini-generated images to detect or remove SynthID watermarks.")

uploaded = st.file_uploader(
    "Drop images here (PNG / JPEG)",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True,
)

if not uploaded:
    st.info("Upload one or more images to get started.", icon="⬆️")
    st.stop()

run_btn = st.button(
    f"▶ Run {'Removal' if 'Remove' in mode else 'Detection'} on {len(uploaded)} image(s)",
    type="primary",
)

if not run_btn:
    st.stop()

# Load model
try:
    if "Remove" in mode:
        bypass, codebook = load_bypass(bypass_cb_path)
    else:
        detector = load_detector(detect_cb_path)
except Exception as e:
    st.error(f"**Could not load codebook:** {e}\n\nCheck the path in the sidebar.")
    st.stop()

# Process
results_for_zip = []
progress = st.progress(0, text="Processing…")

for i, f in enumerate(uploaded):
    progress.progress((i + 1) / len(uploaded), text=f"Processing {f.name}…")
    pil_img = Image.open(f)
    rgb = pil_to_rgb_array(pil_img)
    stem = Path(f.name).stem

    with st.container():
        if "Remove" in mode:
            try:
                result = bypass.bypass_v3(rgb, codebook, strength=strength)
                out_arr = get_result_image(result)
                out_pil = array_to_pil(out_arr)
                out_bytes = img_to_bytes(out_pil, fmt=output_format)
                ext = output_format.lower()

                col1, col2 = st.columns(2)
                with col1:
                    st.image(pil_img, caption=f"Original — {f.name}")
                with col2:
                    st.image(out_pil, caption="Processed")

                psnr = getattr(result, 'psnr', None)
                details = getattr(result, 'details', {}) or {}
                profile_res = details.get('profile_resolution', 'N/A')
                exact = details.get('exact_match', False)

                st.markdown(f"""
<div class="result-box">
<b>{f.name}</b><br>
{'PSNR: <b>' + f'{psnr:.1f} dB</b> &nbsp;|&nbsp;' if psnr else ''}
Resolution: <b>{profile_res}</b> &nbsp;|&nbsp;
Exact codebook match: <b>{'✅' if exact else '⚠️ fallback'}</b>
</div>
""", unsafe_allow_html=True)

                st.download_button(
                    label=f"⬇ Download {stem}_clean.{ext}",
                    data=out_bytes,
                    file_name=f"{stem}_clean.{ext}",
                    mime=f"image/{ext}",
                    key=f"dl_{i}",
                )
                results_for_zip.append((f"{stem}_clean.{ext}", out_bytes))

            except Exception as e:
                st.error(f"**{f.name}** — processing failed: {e}")

        else:
            try:
                det = detector.detect_array(rgb)
                verdict = "🟢 Watermarked" if det.is_watermarked else "⚪ Not detected"

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(pil_img, caption=f.name)
                with col2:
                    st.markdown(f"""
<div class="result-box">
<b>{f.name}</b><br>
Verdict: <b>{verdict}</b><br>
Confidence: <b>{det.confidence:.1%}</b>
</div>
""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"**{f.name}** — detection failed: {e}")

        st.divider()

progress.empty()

if "Remove" in mode and len(results_for_zip) > 1:
    st.download_button(
        label=f"⬇ Download all {len(results_for_zip)} images as ZIP",
        data=make_zip(results_for_zip),
        file_name="synthid_cleaned.zip",
        mime="application/zip",
    )

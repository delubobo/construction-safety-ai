"""
Construction Safety AI — Streamlit Application
Upload a construction site photo → YOLOv8 detects PPE violations →
annotated image + violation table + downloadable PDF report + trend chart.
"""

import sys
import os
from pathlib import Path

# Allow imports from project root (needed when running via `streamlit run app/main.py`)
_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Load .env so ROBOFLOW_API_KEY is available before the detector imports
_env_file = _ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        if "=" in _line and not _line.startswith("#"):
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

import streamlit as st
from PIL import Image
import plotly.graph_objects as go
import pandas as pd

from model.detector import PPEDetector
from model.report_builder import build_report
from app.utils.image_annotator import annotate_image, pil_to_bytes
from app import database as db

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Construction Safety AI",
    page_icon="🦺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .metric-card {
        background: #F2F2F7;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        text-align: center;
    }
    .violation-high   { color: #FF3B30; font-weight: 700; }
    .violation-medium { color: #FF9500; font-weight: 700; }
    .violation-low    { color: #FFCC00; font-weight: 700; }
    .compliant        { color: #34C759; font-weight: 700; }
</style>
""", unsafe_allow_html=True)


# ── Cached model loader ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_detector() -> PPEDetector:
    return PPEDetector()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🦺 Safety AI")
    st.markdown("---")

    st.subheader("Report Settings")
    site_name = st.text_input("Site Name", value="AWS Data Center — Lubbock, TX")
    project_name = st.text_input("Project Name", value="AWS Data Center Expansion")
    inspector_name = st.text_input("Inspector Name", value="Safety Officer")

    st.markdown("---")
    st.subheader("Detection Settings")
    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Lower = more detections, more false positives.",
    )

    st.markdown("---")
    st.caption("Construction Safety AI · YOLOv8 PPE Detection")


# ── Main layout ───────────────────────────────────────────────────────────────
st.title("Construction Site PPE Violation Detector")
st.markdown(
    "Upload a construction site photo to automatically detect PPE violations "
    "using computer vision. Results are logged and a downloadable PDF report is generated."
)

tab_analyze, tab_history = st.tabs(["📷 Analyze Photo", "📈 Violation Trend"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Analyze Photo
# ══════════════════════════════════════════════════════════════════════════════
with tab_analyze:
    uploaded = st.file_uploader(
        "Upload a construction site photo",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supports JPG, PNG, WEBP. Larger images may take a few seconds.",
    )

    if uploaded is None:
        st.info("Upload a photo above to begin analysis.")
        st.stop()

    # Load image
    pil_image = Image.open(uploaded).convert("RGB")

    col_orig, col_annotated = st.columns(2, gap="medium")
    with col_orig:
        st.subheader("Original Image")
        st.image(pil_image, use_container_width=True)

    # ── Run detection ─────────────────────────────────────────────────────────
    with st.spinner("Analyzing image…"):
        detector = load_detector()
        detector.confidence = confidence  # always sync from slider

        import tempfile, io
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            pil_image.save(tmp_path, format="JPEG", quality=95)

        detections = detector.detect(tmp_path)
        os.unlink(tmp_path)

    violations = [d for d in detections if d["is_violation"]]
    safe_dets  = [d for d in detections if not d["is_violation"]]

    # ── Annotate ──────────────────────────────────────────────────────────────
    annotated = annotate_image(pil_image, detections)

    with col_annotated:
        st.subheader("Annotated Image")
        st.image(annotated, use_container_width=True)

    # ── KPI metrics ───────────────────────────────────────────────────────────
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Detections", len(detections))
    m2.metric("Violations", len(violations), delta=None)
    m3.metric("Compliant PPE", len(safe_dets))
    compliance_rate = (
        round((len(safe_dets) / len(detections)) * 100) if detections else 100
    )
    m4.metric("Compliance Rate", f"{compliance_rate}%")

    # ── Violation table ───────────────────────────────────────────────────────
    st.markdown("### Detection Results")

    if violations:
        st.error(f"⚠️ {len(violations)} PPE violation{'s' if len(violations) != 1 else ''} detected")

        viol_rows = []
        for d in violations:
            viol_rows.append({
                "Violation": d["label"],
                "Severity": d["severity"],
                "Confidence": f"{d['confidence']:.0%}",
                "Corrective Action": d["corrective_action"][:120] + "…"
                    if len(d["corrective_action"]) > 120 else d["corrective_action"],
            })
        st.dataframe(
            pd.DataFrame(viol_rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.success("✅ No PPE violations detected in this image.")

    if safe_dets:
        with st.expander(f"Compliant PPE detected ({len(safe_dets)})"):
            safe_rows = [
                {"Class": d["class"], "Confidence": f"{d['confidence']:.0%}"}
                for d in safe_dets
            ]
            st.dataframe(pd.DataFrame(safe_rows), hide_index=True)

    # ── Log to DB ─────────────────────────────────────────────────────────────
    db.log_session(
        image_filename=uploaded.name,
        all_detections=detections,
    )

    # ── PDF Download ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Download Safety Report")

    pdf_bytes = build_report(
        annotated_image=annotated,
        detections=detections,
        site_name=site_name,
        project_name=project_name,
        inspector_name=inspector_name,
        image_filename=uploaded.name,
    )

    from datetime import datetime
    report_filename = f"safety_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    st.download_button(
        label="📄 Download PDF Safety Report",
        data=pdf_bytes,
        file_name=report_filename,
        mime="application/pdf",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Violation Trend
# ══════════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("Violation Trend Over Time")

    sessions = db.get_all_sessions()

    if not sessions:
        st.info("No detection sessions logged yet. Run an analysis first.")
    else:
        df = pd.DataFrame(sessions)
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["violation_count"],
            mode="lines+markers",
            name="Violations",
            line=dict(color="#FF3B30", width=2),
            marker=dict(size=8),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Violations: %{y}<br>"
                "<extra></extra>"
            ),
        ))
        fig.add_trace(go.Scatter(
            x=df["timestamp"],
            y=df["total_detections"],
            mode="lines+markers",
            name="Total Detections",
            line=dict(color="#007AFF", width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate=(
                "<b>%{x}</b><br>"
                "Total: %{y}<br>"
                "<extra></extra>"
            ),
        ))
        fig.update_layout(
            xaxis_title="Session Time",
            yaxis_title="Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=0, r=0, t=10, b=0),
            height=350,
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Session Log")
        display_df = df[["timestamp", "image_filename", "violation_count", "total_detections"]].copy()
        display_df.columns = ["Timestamp", "Image", "Violations", "Total Detections"]
        display_df["Timestamp"] = display_df["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

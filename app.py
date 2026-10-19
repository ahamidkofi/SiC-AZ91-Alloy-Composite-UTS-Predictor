from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# ---------- Load model (once) ----------
MODEL_DIR = Path(__file__).parent / "models"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "rf_uts_pipeline.joblib")
    columns = joblib.load(MODEL_DIR / "feature_columns.joblib")
    return model, columns


model, columns = load_model()

GROUPS = ["Stir casting", "Deformed", "Other casting", "Homogenized"]

# ---------- Page setup & styling ----------
st.set_page_config(page_title="ML SiC/AZ91 UTS Predictor", page_icon="🔩", layout="centered")

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.5rem; max-width: 960px; }
    h1 { font-weight: 800; line-height: 1.15; }
    div.stButton > button {
        width: 100%; border-radius: 12px; padding: 0.7rem 1rem;
        font-weight: 700; font-size: 1.05rem;
    }
    .result-card {
        border-radius: 16px; padding: 1.5rem 1.2rem; text-align: center; margin-top: 1rem;
        background: linear-gradient(135deg, rgba(56,189,248,0.18), rgba(99,102,241,0.18));
        border: 1px solid rgba(99,102,241,0.35);
    }
    .result-label { font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.75; }
    .result-value { font-size: 3.2rem; font-weight: 800; line-height: 1.1; }
    .result-unit  { font-size: 1.1rem; opacity: 0.75; }
    .footer { text-align: center; opacity: 0.8; font-size: 0.9rem; padding: 0.5rem 0 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.title("Machine Learning SiC/AZ91 Composite")
st.subheader("Ultimate Tensile Strength Predictor")
st.write(
    "This research prototype uses machine learning to predict the ultimate tensile "
    "strength (UTS) of SiC micro-particle reinforced AZ91 magnesium composites for "
    "application in light-vehicle and aerospace industries."
)

st.info(
    "**AZ91:** a lightweight magnesium alloy containing about 9% aluminium and 1% zinc, "
    "valued in light-vehicle and aerospace parts where low weight matters.\n\n"
    "**SiC (Silicon Carbide):** a very hard ceramic added to the alloy as micro-sized "
    "particles to reinforce it and raise its strength.\n\n"
    "**UTS (Ultimate Tensile Strength):** the maximum stress a material can withstand "
    "while being stretched before it breaks."
)

# ---------- Inputs ----------
st.subheader("Composite Details")
with st.container(border=True):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sic_vol = st.number_input(
            "SiC volume (%)", min_value=0.0, max_value=30.0, value=10.0, step=0.5,
            help="Percentage of the composite volume made up of SiC particles.",
        )
    with col2:
        sic_size = st.number_input(
            "SiC particle size (µm)", min_value=0.1, max_value=110.0, value=5.0, step=0.5,
            help="Average size of the SiC reinforcement particles.",
        )
    with col3:
        grain = st.number_input(
            "Grain size (µm)", min_value=0.1, max_value=120.0, value=20.0, step=0.5,
            help="Average grain size of the magnesium matrix.",
        )
    with col4:
        group = st.selectbox(
            "Processing group", GROUPS,
            help=(
                "Stir casting; Deformed (hot extrusion, ECAP, rolling); "
                "Other casting (semi-solid stirring, centrifugal casting, simulated, CCDFed); "
                "Homogenized (T6 to T9 heat treatments)."
            ),
        )

    predict = st.button("Predict UTS", type="primary")

# ---------- Predict ----------
if predict:
    row = {c: 0 for c in columns}
    row["SiC_vol_perc"] = sic_vol
    row["SiC_size_um"] = sic_size
    row["av_grain_um"] = grain
    row[f"Treatment_group_{group}"] = 1
    new_sample = pd.DataFrame([row])[columns]

    prediction = model.predict(new_sample)[0]

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted Ultimate Tensile Strength</div>
            <div class="result-value">{prediction:.0f} <span class="result-unit">MPa</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Note when inputs fall outside the densest part of the experimental data
    if sic_vol > 20 or grain > 15 or sic_size > 50:
        st.info(
            "These inputs lie beyond the range most densely covered by the experimental "
            "dataset (SiC up to 20 vol%, grain size up to 15 µm, particle size up to 50 µm). "
            "The estimate extends the learned trends and is best confirmed experimentally."
        )

st.caption("Random Forest regression model trained on 83 experimental data points.")

# ---------- Footer ----------
st.divider()
st.markdown(
    '<div class="footer">Developed and deployed by <b>Hamid Kofi Abdul</b></div>',
    unsafe_allow_html=True,
)
import pandas as pd
import streamlit as st
import numpy as np
import joblib
import altair as alt

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Real Estate Price Prediction",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- CUSTOM CSS FOR AESTHETICS ---
st.markdown("""
<style>
    /* Center align headers */
    .stHeadingContainer {
        text-align: center;
    }
    /* Add subtle borders to containers */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column;"] > [data-testid="stVerticalBlock"] {
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 1rem;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD PREPROCESSOR ---
# Using the requested joblib loading method. 
# Added try/except to prevent crash if file is missing during testing.
preprocessor = None
try:
    preprocessor = joblib.load("../models/pipeline.joblib")
except FileNotFoundError:
    st.warning("pipeline.joblib' not found. Using mock preprocessing logic for demonstration.")
except Exception as e:
    st.error(f"Error loading pipeline: {e}")

# --- HEADER & DESCRIPTION ---
st.title("Real Estate Price Prediction System")

# Description based on Project P176B104
st.markdown("""
This system is designed to forecast property prices using machine learning algorithms. 
It utilizes data from the Real Estate Southern Spain 2024 dataset, analyzing attributes such as location, 
dimensions (interior/exterior area), and amenities to generate accurate valuations.
""")

st.divider()

# --- INPUT SECTION (CENTERED) ---
st.subheader("Property Configuration")

# Organizing inputs into 3 balanced columns for better center alignment
col_dims, col_wellness, col_tech = st.columns(3, gap="medium")

with col_dims:
    st.markdown("**Dimensions & Layout**")
    
    # 1. Bathrooms (Synced Slider + Text Input)
    st.caption("Bathrooms")
    col_bath_slider, col_bath_text = st.columns([2, 1])
    with col_bath_slider:
        bath_slider = st.slider("Bath Slider", 1, 10, 2, key="bath_slider", label_visibility="collapsed")
    with col_bath_text:
        bathrooms = st.number_input("Bath Value", 1, 10, bath_slider, step=1, label_visibility="collapsed")
    
    # 2. Bedrooms (Synced Slider + Text Input)
    st.caption("Bedrooms")
    col_bed_slider, col_bed_text = st.columns([2, 1])
    with col_bed_slider:
        bed_slider = st.slider("Bed Slider", 1, 15, 3, key="bed_slider", label_visibility="collapsed")
    with col_bed_text:
        bedrooms = st.number_input("Bed Value", 1, 15, bed_slider, step=1, label_visibility="collapsed")

    # 3. Indoor Surface (Plus/Minus Input with larger step)
    indoor_surface = st.number_input(
        "Indoor Surface (sqm)", 
        min_value=20.0, 
        max_value=100000000.0, 
        value=120.0, 
        step=5.0,
        help="Total indoor area in square meters"
    )


with col_wellness:
    st.markdown("**Wellness & Recreation**")
    communal_pool = st.checkbox("Communal Pool", value=True)
    private_pool = st.checkbox("Private Pool", value=False)
    jacuzzi = st.checkbox("Jacuzzi", value=False)
    gym = st.checkbox("Gym", value=False)
    barbeque = st.checkbox("Barbeque", value=False)

with col_tech:
    st.markdown("**Tech, Security & Exterior**")
    luxury = st.checkbox("Luxury Property", value=False)
    domotics = st.checkbox("Domotics (Smart Home)", value=False)
    heating = st.checkbox("U/F Heating / Climate", value=False)
    alarm_system = st.checkbox("Alarm System", value=False)
    communal_garden = st.checkbox("Communal Garden", value=True)
    private_garden = st.checkbox("Private Garden", value=False)
    games_room = st.checkbox("Games Room", value=False)


# --- DATA PREPARATION ---

# Create raw dataframe
data = {
    "bathrooms": [bathrooms],
    "indoor_surface": [indoor_surface],
    "jacuzzi": [1 if jacuzzi else 0],
    "communal_pool": [1 if communal_pool else 0],
    "luxury": [1 if luxury else 0],
    "domotics": [1 if domotics else 0],
    "u/f_heating_climate_control": [1 if heating else 0],
    "games_room": [1 if games_room else 0],
    "bedrooms": [bedrooms],
    "communal_garden": [1 if communal_garden else 0],
    "private_pool": [1 if private_pool else 0],
    "alarm_system_security": [1 if alarm_system else 0],
    "gym": [1 if gym else 0],
    "private_garden": [1 if private_garden else 0],
    "barbeque": [1 if barbeque else 0],
}

df_orig = pd.DataFrame(data)

# Preprocessing Logic
def preprocess_input(df, pipe=None):
    # If the joblib pipeline was loaded successfully, use it:
    if pipe is not None:
        try:
            return pipe.transform(df)
        except Exception:
            # Fallback if pipeline expects different columns than provided
            pass

    # Fallback/Mock Preprocessing
    df_proc = df.copy()
    if "indoor_surface" in df_proc.columns:
        df_proc["indoor_surface"] = np.log1p(df_proc["indoor_surface"])
    return df_proc

df_preprocessed = preprocess_input(df_orig, preprocessor)


# --- MOCK PREDICTIONS ---
# (Logic preserved exactly as requested)

base_log_price = 11.5
# Handle cases where preprocessor returns numpy array or dataframe
try:
    # If dataframe
    surf_val = df_preprocessed["indoor_surface"].iloc[0]
except:
    # If numpy array (likely from joblib pipeline)
    # Assuming indoor_surface is at index 1 based on input dict order
    surf_val = df_preprocessed[0][1] 

weight_surface = surf_val * 0.4
weight_luxury = 0.5 if luxury else 0
weight_pool = 0.2 if private_pool else 0

# 1. Linear Regression
pred_log_lr = base_log_price + weight_surface + weight_luxury + weight_pool + np.random.normal(0, 0.05)
pred_lr = np.expm1(pred_log_lr)

# 2. Neural Network
pred_log_nn = base_log_price + (weight_surface * 1.05) + (weight_luxury * 1.2) + weight_pool + 0.1
pred_nn = np.expm1(pred_log_nn)

# 3. Gradient Boosting
pred_log_gb = base_log_price + (weight_surface * 0.98) + (weight_luxury * 0.9) + (weight_pool * 1.1)
pred_gb = np.expm1(pred_log_gb)

# Average Calculation
avg_price = (pred_lr + pred_nn + pred_gb) / 3

df_results = pd.DataFrame({
    "Model": ["Linear Regression", "Neural Network", "Gradient Boosting"],
    "Price": [pred_lr, pred_nn, pred_gb],
    "Log Prediction": [pred_log_lr, pred_log_nn, pred_log_gb]
})


# --- DISPLAY RESULTS ---
st.divider()
st.subheader("Valuation Analysis")

# 1. Main Result: Average Prediction
# Displayed prominently in the center
st.markdown(
    f"""
    <div style="text-align: center; padding: 20px; background-color: rgba(60, 179, 113, 0.1); border-radius: 10px; margin-bottom: 20px;">
        <h3 style="margin:0;">Ensemble Prediction (Average)</h3>
        <h1 style="color: #4CAF50; margin:0;">€ {avg_price:,.2f}</h1>
    </div>
    """, 
    unsafe_allow_html=True
)

# 2. Individual Model Cards
c1, c2, c3 = st.columns(3)

with c1:
    with st.container():
        st.markdown("**Linear Regression**")
        st.metric(label="Price", value=f"€ {pred_lr:,.0f}")
        st.progress(min(pred_lr/avg_price, 1.0)) # Visual indicator relative to average

with c2:
    with st.container():
        st.markdown("**Neural Network**")
        st.metric(label="Price", value=f"€ {pred_nn:,.0f}")
        st.progress(min(pred_nn/avg_price, 1.0))

with c3:
    with st.container():
        st.markdown("**Gradient Boosting**")
        st.metric(label="Price", value=f"€ {pred_gb:,.0f}")
        st.progress(min(pred_gb/avg_price, 1.0))

st.markdown("###") # Spacer

# 3. Visualization and Data
col_chart, col_data = st.columns([2, 1])

with col_chart:
    st.markdown("**Model Variance Comparison**")
    # Using Altair for better label control (straight labels)
    chart = alt.Chart(df_results).mark_bar().encode(
        x=alt.X('Model', axis=alt.Axis(labelAngle=0)), # Ensure straight labels
        y='Price',
        color=alt.Color('Model', legend=None),
        tooltip=['Model', 'Price']
    ).properties(
        height=300
    )
    st.altair_chart(chart, use_container_width=True)

with col_data:
    st.markdown("**Technical Data**")
    with st.expander("View Inputs", expanded=True):
        st.dataframe(df_orig.T, use_container_width=True, column_config={0: "Value"})
    
    with st.expander("View Logs"):
        st.dataframe(df_results[["Model", "Log Prediction"]], hide_index=True)
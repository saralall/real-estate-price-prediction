import pandas as pd
import streamlit as st
import numpy as np
import joblib
import altair as alt
import os

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
    /* Metric styling to make them pop */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- LOAD ARTIFACTS (MODELS & PIPELINE) ---
@st.cache_resource
def load_artifacts():
    """
    Loads the machine learning models and preprocessing pipeline.
    Checks typical paths (current dir and ../models/).
    """
    artifacts = {}
    files = {
        "pipeline": "pipeline.joblib",
        "lr": "lr.joblib",
        "nn": "nn.joblib",
        "xgb": "xgb.joblib"
    }
    
    for key, filename in files.items():
        path = None
        # Check current directory
        if os.path.exists(filename):
            path = filename
        # Check ../models/ directory
        elif os.path.exists(f"../models/{filename}"):
            path = f"../models/{filename}"
        # Check models/ directory
        elif os.path.exists(f"models/{filename}"):
            path = f"models/{filename}"
            
        if path:
            try:
                artifacts[key] = joblib.load(path)
            except Exception as e:
                st.error(f"Failed to load {filename}: {e}")
                artifacts[key] = None
        else:
            st.warning(f"File '{filename}' not found. Prediction for this model will be disabled.")
            artifacts[key] = None
            
    return artifacts

# Load everything once
artifacts = load_artifacts()
preprocessor = artifacts.get("pipeline")

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

col_dims, col_wellness, col_tech = st.columns(3, gap="medium")

with col_dims:
    st.markdown("**Dimensions & Layout**")
    
    # 1. Bathrooms (Now Number Input)
    bathrooms = st.number_input(
        "Bathrooms", 
        min_value=1, 
        max_value=10, 
        value=2, 
        step=1,
        help="Number of bathrooms in the property"
    )
    
    # 2. Bedrooms (Now Number Input)
    bedrooms = st.number_input(
        "Bedrooms", 
        min_value=1, 
        max_value=15, 
        value=3, 
        step=1,
        help="Number of bedrooms in the property"
    )

    # 3. Indoor Surface (Now Slider)
    indoor_surface = st.slider(
        "Indoor Surface (sqm)", 
        min_value=20.0, 
        max_value=10000.0, 
        value=120.0, 
        step=5.0,
        help="Total indoor area in square meters"
    )

with col_wellness:
    st.markdown("**Wellness & Recreation**")
    communal_pool = st.checkbox("Communal Pool", value=True, help="Check if the property has access to a shared pool")
    private_pool = st.checkbox("Private Pool", value=False, help="Check if the property has a private pool")
    jacuzzi = st.checkbox("Jacuzzi", value=False, help="Check if the property includes a jacuzzi")
    gym = st.checkbox("Gym", value=False, help="Check if the property includes a gym or fitness area")
    barbeque = st.checkbox("Barbeque", value=False, help="Check if the property has a barbeque area")

with col_tech:
    st.markdown("**Tech, Security & Exterior**")
    luxury = st.checkbox("Luxury Property", value=False, help="Check if the property is classified as luxury")
    domotics = st.checkbox("Domotics", value=False, help="Check if the property has smart home automation features")
    heating = st.checkbox("U/F Heating / Climate", value=False, help="Check if the property has underfloor heating or climate control")
    alarm_system = st.checkbox("Alarm System", value=False, help="Check if the property has a security alarm system")
    communal_garden = st.checkbox("Communal Garden", value=True, help="Check if the property has access to a shared garden")
    private_garden = st.checkbox("Private Garden", value=False, help="Check if the property has a private garden")
    games_room = st.checkbox("Games Room", value=False, help="Check if the property includes a dedicated games room")


# --- DATA PREPARATION ---

# Create raw dataframe matching the training features
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

# --- REAL PREDICTIONS ---
st.divider()
st.subheader("Valuation Analysis")

# Check if pipeline exists
if preprocessor is None:
    st.error("Cannot proceed: 'pipeline.joblib' could not be loaded.")
else:
    # 1. Transform Data
    try:
        X_processed = preprocessor.transform(df_orig)
        
        # Try to extract feature names for the technical view
        try:
            feature_names_out = preprocessor.get_feature_names_out()
        except:
            feature_names_out = [f"Feature {i}" for i in range(X_processed.shape[1])]
            
    except Exception as e:
        st.error(f"Preprocessing error: {e}")
        st.stop()

    # 2. Make Predictions
    results = []
    
    # Define models to run
    models_config = [
        ("Linear Regression", artifacts.get("lr")),
        ("Neural Network", artifacts.get("nn")),
        ("Gradient Boosting", artifacts.get("xgb"))
    ]

    valid_predictions = []
    model_outputs = {}

    for name, model in models_config:
        if model:
            try:
                # Predict log price
                pred_log = model.predict(X_processed)[0]
                # Inverse transform to get actual price (expm1)
                pred_price = np.expm1(pred_log)
                
                # Store results
                valid_predictions.append(pred_price)
                model_outputs[name] = {"price": pred_price, "log": pred_log}
                results.append({"Model": name, "Price": pred_price, "Log Prediction": pred_log})
            except Exception as e:
                st.warning(f"Error running {name}: {e}")
                model_outputs[name] = {"price": 0, "log": 0}

    if not valid_predictions:
        st.error("No predictions could be generated.")
    else:
        # Calculate Ensemble Average
        avg_price = sum(valid_predictions) / len(valid_predictions)
        
        df_results = pd.DataFrame(results)

        # --- DISPLAY RESULTS ---

        # 1. Main Result: Average Prediction
        st.markdown(
            f"""
            <div style="text-align: center; padding: 20px; background-color: rgba(60, 179, 113, 0.1); border-radius: 10px; margin-bottom: 20px;">
                <h3 style="margin:0;">Ensemble Prediction (Average)</h3>
                <h1 style="color: #4CAF50; margin:0;">€ {avg_price:,.2f}</h1>
            </div>
            """, 
            unsafe_allow_html=True
        )

        # --- NEW FEATURES 1, 2, 3: METRICS ROW ---
        # 1. Valuation Range, 2. Price/m², 3. Reliability Score
        
        min_price = min(valid_predictions)
        max_price = max(valid_predictions)
        price_sqm = avg_price / indoor_surface
        
        # Reliability Calculation (Coefficient of Variation)
        std_dev = np.std(valid_predictions)
        coeff_var = std_dev / avg_price if avg_price > 0 else 0
        
        if coeff_var < 0.15:
            reliability_label = "High Agreement"
            reliability_color = "normal" # Green/Standard
        elif coeff_var < 0.30:
            reliability_label = "Moderate Variance"
            reliability_color = "off" # Gray/Neutral
        else:
            reliability_label = "Low Agreement"
            reliability_color = "inverse" # Red/Warning

        # Layout for Metrics
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.metric("Conservative Est.", f"€ {min_price:,.0f}", help="Lowest valuation among the models")
        with m2:
            st.metric("Optimistic Est.", f"€ {max_price:,.0f}", help="Highest valuation among the models")
        with m3:
            st.metric("Price per m²", f"€ {price_sqm:,.0f}", help="Average valuation divided by indoor surface")
        with m4:
            st.metric("Model Consensus", reliability_label, help="How closely the models agree with each other. Lower variance is better.", delta=f"±{coeff_var*100:.1f}%", delta_color=reliability_color)

        st.divider()

        # 2. Individual Model Cards
        c1, c2, c3 = st.columns(3)

        # Helper to display card
        def display_card(container, name):
            data = model_outputs.get(name)
            if data and data["price"] > 0:
                price = data["price"]
                ratio = min(price / avg_price, 1.0) if avg_price > 0 else 0
                container.markdown(f"**{name}**")
                container.metric(label="Price", value=f"€ {price:,.0f}")
                container.progress(ratio)
            else:
                container.markdown(f"**{name}**")
                container.caption("Not Available")

        with c1:
            with st.container():
                display_card(st, "Linear Regression")

        with c2:
            with st.container():
                display_card(st, "Neural Network")

        with c3:
            with st.container():
                display_card(st, "Gradient Boosting")

        st.markdown("###") # Spacer

        # 3. Visualization (GRAPH BIGGER)
        st.markdown("**Model Variance Comparison**")
        chart = alt.Chart(df_results).mark_bar().encode(
            x=alt.X('Model', axis=alt.Axis(labelAngle=0)), 
            y='Price',
            color=alt.Color('Model', legend=None),
            tooltip=['Model',  alt.Tooltip('Price', format=",.2f")]
        ).properties(
            height=500  # Increased from 300 to 500
        )
        st.altair_chart(chart, use_container_width=True)

        # 4. Technical Data (MOVED BELOW EVERYTHING)
        st.divider()
        st.markdown("**Technical Data**")
        
        # A. View User Inputs
        with st.expander("1. User Inputs", expanded=True):
            st.dataframe(df_orig.T, use_container_width=True, column_config={0: "Value"})
        
        # B. View Preprocessed Vector (Internal State)
        with st.expander("2. Preprocessed Vector (Model Input)"):
            try:
                df_proc = pd.DataFrame(X_processed, columns=feature_names_out)
                st.dataframe(df_proc.T, use_container_width=True, column_config={0: "Transformed Value"})
            except Exception as e:
                st.write(X_processed)
                st.caption(f"Could not format dataframe: {e}")

        # C. View Linear Regression Coefficients (Logic)
        with st.expander("3. Linear Model Weights"):
            lr_model = artifacts.get("lr")
            if lr_model and hasattr(lr_model, "coef_"):
                # Match coefficients to features
                coeffs = lr_model.coef_
                # Handle if coefs is 1D or 2D
                if coeffs.ndim > 1: coeffs = coeffs[0]
                
                df_coef = pd.DataFrame({
                    "Feature": feature_names_out,
                    "Weight": coeffs
                }).sort_values(by="Weight", ascending=False)
                
                st.dataframe(df_coef, hide_index=True, use_container_width=True)
            else:
                st.info("Linear Regression model not loaded or coefficients unavailable.")

        # D. View Predictions
        with st.expander("4. Prediction Logs"):
            st.dataframe(df_results[["Model", "Log Prediction"]], hide_index=True)
            
        # E. View Hyperparameters
        with st.expander("5. Model Hyperparameters"):
            for name, model in models_config:
                if model:
                    st.caption(f"**{name}**")
                    try:
                        st.json(model.get_params(), expanded=False)
                    except:
                        st.write("Params not available")
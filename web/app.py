import pandas as pd
import streamlit as st

st.set_page_config(page_title="Real Estate Price Prediction")

st.title("Real Estate Price Prediction")
st.markdown("""
This dashboard compares the predictions of two models based on user inputs.
Adjust the parameters in the sidebar to see how the models react differently.
""")

st.divider()

# --- SIDEBAR: INPUT FEATURES ---
st.sidebar.header("Input Parameters")
st.sidebar.write("Adjust the feature values below:")

# Feature 1: Slider (Continuous variables like Age, Temperature, Price)
feat_1 = st.sidebar.slider(
    "Feature 1 (e.g., Temperature)", min_value=0.0, max_value=1.0, value=0.5, step=0.01
)

# Feature 2: Number Input (Precise numbers like Sq Ft, Salary)
feat_2 = st.sidebar.number_input(
    "Feature 2 (e.g., Square Footage)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01,
)

# Feature 3: Select Slider (Ordinal categories: Low/Med/High)
feat_3_raw = st.sidebar.select_slider(
    "Feature 3 (e.g., Complexity)", options=["Low", "Medium", "High"]
)
# Convert string input to number for the model
feat_3_map = {"Low": 0.1, "Medium": 0.5, "High": 1.0}
feat_3 = feat_3_map[feat_3_raw]

# Feature 4: Radio Button (Mutually exclusive options)
feat_4_raw = st.sidebar.radio(
    "Feature 4 (e.g., Region)", options=["North", "South", "East"]
)
# Convert to number
feat_4_map = {"North": 1, "South": 2, "East": 3}
feat_4 = feat_4_map[feat_4_raw]

# Feature 5: Checkbox / Toggle (Binary Yes/No)
feat_5_bool = st.sidebar.toggle("Feature 5: Include Premium Package?", value=True)
feat_5 = 1.0 if feat_5_bool else 0.0

feat_6_bool = st.sidebar.checkbox("Feature 6: Expedited delivery", value=False)
feat_6 = 1.0 if feat_6_bool else 0.0

# TODO: uncomment to have a "Run Prediction" button
# if st.sidebar.button("Run Prediction", type="primary"):

df_orig = pd.DataFrame(
    {
        "f1": [feat_1],
        "f2": [feat_2],
        "f3": [feat_3],
        "f4": [feat_4],
        "f5": [feat_5],
        "f6": [feat_6],
    }
)

# TODO: preprocess data
df_preprocessed = df_orig


df_print = df_preprocessed.T.reset_index().round(4)
df_print.columns = ["Feature Name", "Value"]

# TODO: predict real values
pred_lr_log = 2
pred_nn_log = 2.5
pred_lr = round(10**pred_lr_log, 2)
pred_nn = round(10**pred_nn_log, 2)
pred_lr_log_print = round(pred_lr_log, 4)
pred_nn_log_print = round(pred_nn_log, 4)

df_chart = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Neural Network"],
        "Prediction": [pred_lr, pred_nn],
    }
)
df_chart_log = pd.DataFrame(
    {
        "Model": ["Linear Regression", "Neural Network"],
        "Log Prediction": [pred_lr_log, pred_nn_log],
    }
)

st.subheader("Prediction Results")

col1, col2 = st.columns(2)

with col1:
    st.info("Linear Regression Model")
    st.metric(label="Predicted Value", value=f"{pred_lr} €")
with col2:
    st.success("Neural Network Model")
    st.metric(label="Predicted Value", value=f"{pred_nn} €")

st.subheader("Visual Comparison")
st.bar_chart(df_chart, x="Model", y="Prediction", color="Model")

with st.expander("View Technical Details (Log-Scale Predictions)"):
    st.write("The values below are the raw outputs from the model logic")

    tech_col1, tech_col2 = st.columns(2)
    with tech_col1:
        st.metric("Linear Regression Log-Value", f"{pred_lr_log}")
    with tech_col2:
        st.metric("Neural Network Log-Value", f"{pred_nn_log}")

    # Show the comparison chart in Log scale (often easier to compare visually)
    st.write("Log-Scale Comparison Chart")
    st.bar_chart(df_chart_log, x="Model", y="Log Prediction", color="Model")

with st.expander("See Raw Input Data"):
    st.dataframe(df_print, hide_index=True)

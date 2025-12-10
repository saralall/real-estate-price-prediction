# Real Estate Price Prediction (Southern Spain 2024)

## Description
This project focuses on predicting real estate prices in Southern Spain using data from 2024. The objective was to develop a robust machine learning pipeline that accurately estimates property values based on various structural and locational features. The project encompasses data preprocessing, exploratory data analysis, the implementation of multiple regression models, and the deployment of a user-friendly web interface.

## Data Source
The dataset used for this analysis is hosted on Zenodo:
* **Link:** [https://doi.org/10.5281/zenodo.10962212](https://doi.org/10.5281/zenodo.10962212)

## Key Features & Methodology

### 1. Data Processing
* **Cleaning & Engineering:** Implementation of a rigorous preprocessing pipeline including outlier removal, handling missing values, and engineering features such as `indoor_surface` to address multicollinearity.
* **Analysis:** Correlation analysis (Pearson and Spearman) to identify key price drivers.

### 2. Machine Learning Models
Three distinct modeling approaches were evaluated to optimize prediction accuracy:
* **Linear Regression:** Established as a baseline to understand linear relationships between features and price.
* **XGBoost Regressor:** Implemented to capture non-linear patterns and interactions; identified as the best-performing model.
* **Neural Network (MLP):** A deep learning approach developed to explore complex feature hierarchies.

### 3. Deployment
* **Streamlit Web Interface:** A fully functional web application allowing users to input property characteristics and receive price predictions in real-time.

## Installation and Usage

### Prerequisites
* Python 3.8+
* [uv](https://github.com/astral-sh/uv) (An extremely fast Python package installer and resolver)

### Installation
1.  Clone the repository.

2.  Create a virtual environment and install dependencies using `uv`:
    ```bash
    # Create a virtual environment
    uv venv

    # Activate the environment
    # On macOS/Linux:
    source .venv/bin/activate
    # On Windows:
    .venv\Scripts\activate

    # Install dependencies
    uv sync
    ```

### Running the Application
To launch the Streamlit interface, you can use `uv run` or run directly within the activated environment:

```bash
# Using uv run (automatically handles environment)
uv run streamlit run app.py

# OR within the active environment
streamlit run app.py
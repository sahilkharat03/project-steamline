import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import load_and_clean_data
from feature_engineering import create_features
from model import train_model

st.title(" HHS Forecasting Dashboard")

# Load data
df = load_and_clean_data("../data.csv1")
df = create_features(df)

# Train model
y_test, preds = train_model(df)

# Plot
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(y_test.index, y_test, label="Actual")
ax.plot(y_test.index, preds, label="Predicted")
ax.legend()

st.pyplot(fig)

st.success("Model Ran Successfully ")
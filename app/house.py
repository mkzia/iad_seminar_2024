import streamlit as st
import pandas as pd
import numpy as np

st.title('House Price Prediction')

st.sidebar.selectbox('income_cat', ['A', 'B'])
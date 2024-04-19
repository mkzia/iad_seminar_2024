import streamlit as st
import pandas as pd
import numpy as np
import json
import dill
with open('rfr_v1.pkl', 'rb') as f:
    reloaded_model = dill.load(f)


st.title('Median House Value Prediction')

with open('rfr_v1_info.json') as f:
    model_info = json.load(f)
    side_bar_options = model_info.get('options')
    options = {}
    for key, value in side_bar_options.items():
        if key in ['ocean_proximity', 'income_cat']:
            options[key] = st.sidebar.selectbox(key, value)
        else:
            min_val, max_val = value
            current_value = (min_val + max_val)/2
            options[key] = st.sidebar.slider(key, min_val, max_val, value=current_value)

st.write(options)

if st.button('Predict'): 
    # Convert options to df 
    df = pd.Series(options).to_frame().T
    df["income_cat"] = pd.cut(df["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
    y_hat = reloaded_model.predict(df)
    st.write(df)
    st.write(f'The predicted median house value is: ${y_hat[0]:,}')
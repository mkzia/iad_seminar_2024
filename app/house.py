import streamlit as st
import pandas as pd
import json
import dill
with open('rfr_v1.pkl', 'rb') as f:
    reloaded_model = dill.load(f)


st.title('House Price Prediction')

with open('options.json') as f:
    side_bar_options = json.load(f)
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
    y_hat = reloaded_model.predict(df)
    st.write(df)
    st.write(f'The predicted house price is: ${y_hat[0]}')
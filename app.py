import streamlit as st

x = st.slider('Select a value')
st.write(x, 'cubed is', x * x * x)

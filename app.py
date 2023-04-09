import streamlit as st
from transformers import pipeline

st.title("Sentiment Analysis")

model_name = 'sentiment-analysis'

classifier = pipeline(model_name)
text = st.text_input('Enter sample text:', 'I really like HuggingFace and it is a great tool for deploying AI models.')

result = classifier(text)
st.write('The sentiment displayed in the sample text is', result[0]['label'], 'with a', (str((result[0]['score'])*100) + ' %'), 'certainty.')
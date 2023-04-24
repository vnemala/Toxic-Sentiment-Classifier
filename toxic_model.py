
import streamlit as st
from transformers import pipeline

st.title("Toxic Comment Analysis")

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained('toxic-comment-model')
model = AutoModelForSequenceClassification.from_pretrained('toxic-comment-model')

classifier = pipeline("text-classification",model = model, tokenizer = tokenizer)

test_df = pd.read_csv('test.csv')

list_test = test_df['comment_text'].tolist()[0:3]
results = classifier(list_test)

st.write('The sentiment displayed in the sample text is', results[0]['label'], 'with a', (str((results[0]['score'])*100) + ' %'), 'certainty.')
st.write('The sentiment displayed in the sample text is', results[1]['label'], 'with a', (str((results[1]['score'])*100) + ' %'), 'certainty.')
st.write('The sentiment displayed in the sample text is', results[2]['label'], 'with a', (str((results[2]['score'])*100) + ' %'), 'certainty.')

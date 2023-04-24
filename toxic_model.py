
import streamlit as st
from transformers import pipeline

st.title("Toxic Comment Analysis")

from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import torch
import torch.nn.functional as F

import pandas as pd
import numpy as np 

model_name = "distilbert-base-uncased-finetuned-sst-2-english"

tokenizer = AutoTokenizer.from_pretrained('toxic-comment-model')
model = AutoModelForSequenceClassification.from_pretrained('toxic-comment-model')

classifier = pipeline("text-classification",model = model, tokenizer = tokenizer)

test_df = pd.read_csv('test.csv')

list_test = test_df['comment_text'].tolist()[0,6,7]
results = classifier(list_test)

final_results = []
final_results_prob = []

final_results.append('identity_hate')
final_results.append('insult')
final_results.append('insult')
 
final_results_prob.append(results[0]['score'])
final_results_prob.append(results[1]['score'])
final_results_prob.append(results[2]['score'])
  

df = pd.DataFrame([list_test,final_results,final_results_prob],
   columns=['tweet','toxicity class', 'probability'])

st.table(df)

st.write('The sentiment displayed in the sample text is', results[0]['label'], 'with a', (str((results[0]['score'])*100) + ' %'), 'certainty.')
st.write('The sentiment displayed in the sample text is', results[1]['label'], 'with a', (str((results[1]['score'])*100) + ' %'), 'certainty.')
st.write('The sentiment displayed in the sample text is', results[2]['label'], 'with a', (str((results[2]['score'])*100) + ' %'), 'certainty.')

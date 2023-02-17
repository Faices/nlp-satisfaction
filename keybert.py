from textblob_de import TextBlobDE
import nbformat
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud 
from functions import *
from gensim.utils import simple_preprocess
from globalvars import *
from IPython.display import display
from IPython.display import Markdown as md
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from pandas_profiling import ProfileReport
import contractions
import gensim
import kaleido
import itertools
import nltk
import numpy as np
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import matplotlib
import matplotlib.pyplot as plt
import plotly.io as pio
import re
import spacy
import string
from functions import *
import csv
from keybert import KeyBERT

# nltk.download('averaged_perceptron_tagger')
# nltk.download('punkt')
# nltk.download('stopwords')

# nlp = spacy.load("de_core_news_lg")
# ## download nlp language package
# #!python -m spacy download de_core_news_lg
# # Load the German model


## Import textbased dataframe with all Surveys from 2019-2022 with text comments and prerocessed columns
filelocation = 'data/DataText'
df_text = pd.read_feather(filelocation)

df_text['text_preprocessed_tokenized_string'] = join_list_of_list(df_text['text_preprocessed_tokenized'])
kw_model = KeyBERT()
keywords = kw_model.extract_keywords(df_text['text_preprocessed_tokenized_string'])


# -*- coding: utf-8 -*-
"""Preprocessing.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1g3qaG5ZbFg_jQpAHAJTuiU6SbK3zodjT
"""




base_dir = '/content/drive/MyDrive/sarcasmdetection/'

import os

os.environ['HF_TOKEN'] = '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'
os.environ['COLAB_APP_IOPUB_DATA_RATE_LIMIT'] = '1000000000'

# Commented out IPython magic to ensure Python compatibility.
# %pip install fasttext
import fasttext.util
fasttext.util.download_model('en',if_exists='ignore') #design
ft = fasttext.load_model('cc.en.300.bin')

# Commented out IPython magic to ensure Python compatibility.
# %pip install torch
# %pip install pandas
# %pip install numpy
# %pip install matplotlib
# %pip install nltk
# %pip install gensim
# %pip install spacy

import os
import torch
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import gensim.downloader as api
import spacy

if 'en_core_web_sm' not in spacy.util.get_installed_models():
    spacy.cli.download('en_core_web_sm')
nlp = spacy.load("en_core_web_sm")

base_dir = '/content/drive/MyDrive/sarcasmdetection/'


#processs each dataset separately first -> gain insight

#get dictionary for all words in dataframe

#path
reddit = {
 'test' : base_dir + 'reddit/sarcasm_detection_shared_task_reddit_testing.jsonl',
 'train' : base_dir + 'reddit/sarcasm_detection_shared_task_reddit_training.jsonl'
}

twitter = {
    'train' : base_dir + 'twitter/sarcasm_detection_shared_task_twitter_training.jsonl',
    'test' : base_dir + 'twitter/sarcasm_detection_shared_task_twitter_testing.jsonl'
}

iac = [base_dir + 'sarcasm_v2/GEN-sarc-notsarc.csv', base_dir + 'sarcasm_v2/HYP-sarc-notsarc.csv',
       base_dir + 'sarcasm_v2/RQ-sarc-notsarc.csv']


#dataset
df_reddit = pd.DataFrame(columns=['text', 'label'])
df_twitter = pd.DataFrame(columns=['text', 'label'])
df_iac = pd.DataFrame(columns=['text', 'label'])

# Process reddit dataset
for file_name in [reddit['train'], reddit['test']]:
    with open(file_name, 'r') as file:
        for line in file:
            obj = json.loads(line)
            y = obj['label']
            context = " ".join(obj['context'])
            resp = obj['response']
            s = context + resp
            df_reddit = pd.concat([df_reddit, pd.DataFrame({'text': [s], 'label': [y]})], ignore_index=True)

# Process twitter dataset
for file_name in [twitter['train'], twitter['test']]:
    with open(file_name, 'r') as file:
        for line in file:
            obj = json.loads(line)
            y = obj['label']
            context = " ".join(obj['context'])
            resp = obj['response']
            s = context + resp
            df_twitter = pd.concat([df_twitter, pd.DataFrame({'text': [s], 'label': [y]})], ignore_index=True)

# Process iac dataset
for file_name in iac:
    df = pd.read_csv(file_name)
    df = df.iloc[:, [0, 2]]
    df.columns = ['label','text']
    df_iac = pd.concat([df_iac, df])
    df_iac = df_iac[['text','label']]



df_reddit['label'] = df_reddit['label'].replace({'NOT_SARCASM': 0, 'SARCASM': 1})

df_twitter['label'] = df_twitter['label'].replace({'NOT_SARCASM': 0, 'SARCASM': 1})

df_iac['label'] = df_iac['label'].replace({'notsarc': 0, 'sarc': 1})

from transformers import DistilBertTokenizer

#create dictionary from words
def stop_word_remove(l) :
    stop_words = set(stopwords.words('english'))
    return [elem for elem in l if elem not in stop_words]

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    # convert POS tag to WordNet format
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]

    return lemmas

#use BERT tokenizer instead
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

#encoded_input = tokenizer.encode(text, truncation=True, max_length=512)
def bert_tokenize(text) :

    return [token for token in tokenizer.tokenize(text)]

def preprocess(df) :

    #lowercase
    df['text'] = df['text'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    #url & non-white space removal
    url_pattern = re.compile(r'https?://\S+')

    df['text'] = df['text'].apply(lambda x: url_pattern.sub('',x))
    #remove non white space
    df['text'] = df['text'].replace(to_replace=r'[^\w\s]', value='', regex=True)
    #digits
    df['text'] = df['text'].replace(to_replace=r'\d', value='', regex=True)

    #tokenization
    df['text'] = df['text'].apply(lambda x : bert_tokenize(x))

    # df['text'] = df['text'].apply(lambda x : [token.lemma_ for token in nlp(x)])

    return df

s = set()
def add_to_vocab_set(word_list) :
    for word in word_list :
        s.add(word)
# print(df_reddit)

df_reddit = preprocess(df_reddit)
df_twitter = preprocess(df_twitter)
df_iac = preprocess(df_iac)

df_reddit['text'].apply(lambda x: add_to_vocab_set(x))
df_twitter['text'].apply(lambda x: add_to_vocab_set(x))
df_iac['text'].apply(lambda x: add_to_vocab_set(x))


print(df_reddit)
print(df_twitter)
print(df_iac)

"""Non tokenized"""

from transformers import DistilBertTokenizer

#create dictionary from words
def stop_word_remove(l) :
    stop_words = set(stopwords.words('english'))
    return [elem for elem in l if elem not in stop_words]

lemmatizer = WordNetLemmatizer()

def lemmatize_tokens(tokens):
    # convert POS tag to WordNet format
    def get_wordnet_pos(word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    lemmas = [lemmatizer.lemmatize(token, get_wordnet_pos(token)) for token in tokens]

    return lemmas

#use BERT tokenizer instead
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

#encoded_input = tokenizer.encode(text, truncation=True, max_length=512)
def bert_tokenize(text) :

    return [token for token in tokenizer.tokenize(text)]

def preprocess(df) :

    #lowercase
    df['text'] = df['text'].apply(lambda x: x.lower() if isinstance(x, str) else x)
    #url & non-white space removal
    url_pattern = re.compile(r'https?://\S+')

    df['text'] = df['text'].apply(lambda x: url_pattern.sub('',x))
    #remove non white space
    df['text'] = df['text'].replace(to_replace=r'[^\w\s]', value='', regex=True)
    #digits
    df['text'] = df['text'].replace(to_replace=r'\d', value='', regex=True)

    #tokenization
    df['text'] = df['text'].apply(lambda x : bert_tokenize(x))

    # df['text'] = df['text'].apply(lambda x : [token.lemma_ for token in nlp(x)])

    return df

s = set()
def add_to_vocab_set(word_list) :
    for word in word_list :
        s.add(word)
# print(df_reddit)

df_reddit = preprocess(df_reddit)
df_twitter = preprocess(df_twitter)
df_iac = preprocess(df_iac)

df_reddit['text'].apply(lambda x: add_to_vocab_set(x))
df_twitter['text'].apply(lambda x: add_to_vocab_set(x))
df_iac['text'].apply(lambda x: add_to_vocab_set(x))


print(df_reddit)
print(df_twitter)
print(df_iac)

#create an int look up map
int_dict = {"PAD": 0, "CLS": 1} #0 = pad token

co = 2
for word in s :
    int_dict[word] = co
    co+=1

print(len(int_dict))
#save int_dict

file_path = base_dir + 'int_dict.txt' #word map with integer
with open(file_path, 'w') as file:
    file.write(json.dumps(int_dict))

emb_lookup = {} #static embeddings LOOKUP -> no CLS

co = 2
for word in s :
    try :
        vector = ft.get_word_vector(word)
        emb_lookup[int_dict[word]] = list(vector.astype(float))
    except Exception as e :
        print(e)
        co+=1
        # emb_lookup[word] = [0] * 300

emb_lookup[0] = [0] * 300

file_path = base_dir + 'emb_lookup.txt'
with open(file_path, 'w') as file:
    file.write(json.dumps(emb_lookup))

print("words not found", co)
print(emb_lookup)

#save df

df_reddit.to_csv(base_dir + 'df_reddit.csv', index=False)
df_twitter.to_csv(base_dir + 'df_twitter.csv', index=False)
df_iac.to_csv(base_dir + 'df_iac.csv', index=False)


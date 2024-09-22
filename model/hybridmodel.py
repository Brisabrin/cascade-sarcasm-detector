

base_dir = '/content/drive/MyDrive/sarcasmdetection/'

import fasttext.util
# fasttext.util.download_model('en',if_exists='ignore') #design

import gdown

model_path = 'cc.en.300.bin'
gdrive_model_id = 'your_file_id'  # Replace 'your_file_id' with the actual ID of the model file on Google Drive

try:
    ft = fasttext.load_model(base_dir + model_path)
    print("FastText model already exists. Loading...")
except FileNotFoundError:
    print("FastText model not found. Downloading from Google Drive...")
    gdown.download(f'https://drive.google.com/uc?id={gdrive_model_id}', model_path, quiet=False)
    print("FastText model downloaded successfully.")
    ft = fasttext.load_model(model_path)
    print("FastText model loaded.")

import requests
from google.colab import drive

def download_file(url, filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz'

filename = 'cc.en.300.bin.gz'

# download_file(url, filename)

# drive.mount('/content/drive')

# destination_path = '/content/drive/MyDrive/cc.en.300.bin.gz'

# !mv {filename} '{destination_path}'

# Commented out IPython magic to ensure Python compatibility.
# %pip install fasttext
# %pip install gdown
import fasttext.util
import gzip
import shutil

compressed_file_path = '/content/drive/MyDrive/cc.en.300.bin.gz'

decompressed_file_path = '/content/drive/MyDrive/cc.en.300.bin'

with gzip.open(compressed_file_path, 'rb') as f_in:
    with open(decompressed_file_path, 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

# Load the decompressed FastText model
ft = fasttext.load_model(decompressed_file_path)

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""Credentials for BERT HUGGING FACE"""

import os

os.environ['HF_TOKEN'] = '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'
os.environ['COLAB_APP_IOPUB_DATA_RATE_LIMIT'] = '1000000000'

"""Fasttext static embedding import"""

# %pip install fasttext
# import fasttext.util
# fasttext.util.download_model('en',if_exists='ignore') #design
# ft = fasttext.load_model('cc.en.300.bin')

"""Multi-dataset import : Reddit, Twitter, *IAC*"""

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
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import gensim.downloader as api
import spacy
import math

# if 'en_core_web_sm' not in spacy.util.get_installed_models():
#     spacy.cli.download('en_core_web_sm')
# nlp = spacy.load("en_core_web_sm")

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

"""Text Preprocessing Stage"""

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

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

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

    # #tokenization
    # df['text'] = df['text'].apply(lambda x : bert_tokenize(x))

    return df


df_reddit = preprocess(df_reddit)
df_twitter = preprocess(df_twitter)
df_iac = preprocess(df_iac)

#find


print(df_reddit)
print(df_twitter)
print(df_iac)

"""Dataset for contextual"""

# df_reddit.to_csv(base_dir + 'df_reddit.csv', index=False)
# df_twitter.to_csv(base_dir + 'df_twitter.csv', index=False)
# df_iac.to_csv(base_dir + 'df_iac.csv', index=False)

"""Dataset for static - use simple tokenization"""

df_reddit['text_tokenized'] = df_reddit['text'].apply(lambda x: word_tokenize(x))
df_twitter['text_tokenized'] = df_twitter['text'].apply(lambda x: word_tokenize(x))
df_iac['text_tokenized'] = df_iac['text'].apply(lambda x: word_tokenize(x))


s = set()
def add_to_vocab_set(word_list) :
    for word in word_list :
        s.add(word)
#keep token ids of static embeddings
df_reddit['text_tokenized'].apply(lambda x: add_to_vocab_set(x))
df_twitter['text_tokenized'].apply(lambda x: add_to_vocab_set(x))
df_iac['text_tokenized'].apply(lambda x: add_to_vocab_set(x))

#save lookup
int_dict = {"PAD": 0} #0 = pad token

co = 1
for word in s :
    int_dict[word] = co
    co+=1

print(len(int_dict))

file_path = base_dir + 'int_dict.txt' #word map with integer
with open(file_path, 'w') as file:
    file.write(json.dumps(int_dict))

emb_lookup = {} #static embeddings LOOKUP -> no CLS
emb_lookup[0] = [0] * 300

co = 1
for word in s :
    try :
        vector = ft.get_word_vector(word)
        emb_lookup[int_dict[word]] = list(vector.astype(float))
    except Exception as e :
        print(e)
        co+=1
print(emb_lookup)

file_path = base_dir + 'emb_lookup.txt'
with open(file_path, 'w') as file:
    file.write(json.dumps(emb_lookup))

print("words not found", co)
print(emb_lookup)
df_reddit['text_tokenized'].apply(lambda x: [int_dict[i] for i in x])
df_twitter['text_tokenized'].apply(lambda x: [int_dict[i] for i in x])
df_iac['text_tokenized'].apply(lambda x: [int_dict[i] for i in x])


#save df for tokenized - static embeddings process
df_reddit.to_csv(base_dir + 'df_reddit_tok.csv', index=False)
df_twitter.to_csv(base_dir + 'df_twitter_tok.csv', index=False)
df_iac.to_csv(base_dir + 'df_iac_tok.csv', index=False)


df_reddit.to_csv(base_dir + 'df_reddit.csv', index=False)
df_twitter.to_csv(base_dir + 'df_twitter.csv', index=False)
df_iac.to_csv(base_dir + 'df_iac.csv', index=False)

#load df
import pandas as pd
import numpy as np
import ast
df_reddit = pd.read_csv(base_dir + 'df_reddit.csv')
df_twitter = pd.read_csv(base_dir + 'df_twitter.csv')
df_iac = pd.read_csv(base_dir + 'df_iac.csv')

df_reddit['text_tokenized'] = df_reddit['text_tokenized'].apply(lambda x: ast.literal_eval(x))
df_twitter['text_tokenized'] = df_twitter['text_tokenized'].apply(lambda x: ast.literal_eval(x))
df_iac['text_tokenized'] = df_iac['text_tokenized'].apply(lambda x: ast.literal_eval(x))

print(df_reddit.columns)
print(df_twitter.columns)
print(df_iac.columns)

print(df_reddit)
print(df_twitter)
print(df_iac)

sarc = 0
not_sarc = 1

not_sarc = np.sum(df_reddit['label'] == 0)
sarc = np.sum(df_reddit['label'] == 1)
print(sarc, not_sarc)
print(df_reddit.shape)

print("\n\n\n\n")
print(df_iac.iloc[5000, 0])
print(df_reddit.iloc[1000, 0])


print(str(df_twitter.iloc[500,0]))

"""Find max sequence length of each dataset"""

print(type(df_reddit.iloc[0,2]))
mean_length = df_reddit['text_tokenized'].apply(len).mean()
std_length = df_reddit['text_tokenized'].apply(len).std()

reddit_threshold = mean_length + 2 * std_length
mean_length = df_twitter['text_tokenized'].apply(len).mean()
std_length = df_twitter['text_tokenized'].apply(len).std()
twitter_threshold = mean_length + 2 * std_length

mean_length = df_iac['text_tokenized'].apply(len).mean()
std_length = df_iac['text_tokenized'].apply(len).std()
iac_threshold = mean_length + 2 * std_length

print()

df_reddit = df_reddit[df_reddit['text_tokenized'].apply(len) <= reddit_threshold]
df_twitter = df_twitter[df_twitter['text_tokenized'].apply(len) <= twitter_threshold]
df_iac = df_iac[df_iac['text_tokenized'].apply(len) <= iac_threshold]

# Recalculate max lengths for the filtered DataFrames
max_reddit = max(df_reddit['text_tokenized'].apply(len))
max_twitter = max(df_twitter['text_tokenized'].apply(len))
max_iac = max(df_iac['text_tokenized'].apply(len))

print(f"Max length of filtered reddit dataset: {max_reddit}")
print(f"Max length of filtered twitter dataset: {max_twitter}")
print(f"Max length of filtered iac dataset: {max_iac}")

max_length = max(max(max_reddit, max_twitter), max_iac)

df_reddit = df_reddit[df_reddit['text_tokenized'].apply(len) <= max_length]
df_twitter = df_twitter[df_twitter['text_tokenized'].apply(len) <= max_length]
df_iac = df_iac[df_iac['text_tokenized'].apply(len) <= max_length]

print(f"Max length of reddit dataset {max_reddit}")
print(f"Max length of twitter dataset {max_twitter}")
print(f"Max length of iac dataset {max_iac}")
print("max_length", max_length)

reddit_size = df_reddit.shape[0]
twitter_size = df_twitter.shape[0]
iac_size = df_iac.shape[0]

print("reddit", reddit_size)
print("twitter", twitter_size)
print("iac", iac_size)

from transformers import BertTokenizer
import torch
import json
import pandas as pd
import torch
from torch.utils.data import Dataset, RandomSampler, DataLoader
import matplotlib.pyplot as plt
import re
import nltk
import torch.nn as nn
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
import ast

emb_file = base_dir + 'emb_lookup.txt'
dict_file = base_dir + 'int_dict.txt'

with open(emb_file, 'r') as file :
  emb_lookup = json.load(file)

with open(dict_file, 'r') as file :
  vocab_set = json.load(file)

print(len(emb_lookup.keys()))
print(emb_lookup.keys())


class CustomTextDataset(Dataset) :
    def __init__(self, df, tokenizer,max_length, set_type='train', transform=None, target_transform=None, vocab_set=None) :
        self.df = df
        self.transform = transform
        self.target_transform = target_transform
        n = self.df.shape[0]
        self.max_length_static = max_length      #max length across train & test
        self.vocab_size = len(vocab_set) if vocab_set != None else 0
        self.tokenizer = tokenizer
        self.set_type = set_type

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx) :
          context_sample = self.df.iloc[idx, 0]#list of words
          label = self.df.iloc[idx,1]

          text_sample = self.df.iloc[idx, 2]
          # print(max_length)

          tokenized_text = tokenizer.encode_plus(context_sample, add_special_tokens=True, truncation=True, max_length=512, padding='max_length', return_tensors='pt')
          input_ids = tokenized_text['input_ids']
          attention_masks = tokenized_text['attention_mask']

          #handle padding for static embeddings

          text_sample = torch.tensor([vocab_set[word] for word in text_sample], dtype=torch.int)
          text_sample = F.pad(text_sample, (0, max(max_length - len(text_sample), 0)), value=0)
          # print("text sample shape", text_sample.size())
          # print("input_ids",input_ids.size())
          # print("attention mask", attention_masks.size())
          # if (self.set_type == 'train') :

          if (self.set_type == 'val') :
            idx += train_size

          elif self.set_type == 'test' or self.set_type == "reddit":
            idx += (train_size + val_size)
          elif self.set_type == 'twitter' :
            idx += (train_size + val_size + reddit_size)


          return text_sample, input_ids, attention_masks, idx , torch.tensor(label, dtype=torch.int)

"""Declare Hyperparameters"""

BATCH_SIZE = 32
NUM_EPOCHS = 5
SHUFFLE = True

"""Create Custom Dataset - split train, val, test and store"""

from transformers import DistilBertTokenizer


tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
# train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)


test_df = pd.concat([df_reddit, df_twitter], ignore_index=True)


train_df, val_df = train_test_split(df_iac, test_size=0.2, random_state=42)
#store the indices for LDA set retrieval

#store the dataframes
train_df.to_csv(base_dir + 'train_df.csv', index=False)
val_df.to_csv(base_dir + 'val_df.csv', index=False)
test_df.to_csv(base_dir + 'test_df.csv', index=False)

"""Getting topic modelling - LDA"""

from gensim import corpora
from gensim.models import LdaModel
import torch
from gensim.test.utils import common_texts

print(train_df)
df_topic = pd.concat([train_df, val_df,test_df], ignore_index=True)
vocab = list(vocab_set.keys())
vocab.remove("PAD")
dictionary = corpora.Dictionary()
dictionary.add_documents(common_texts)
print(type(vocab))
dictionary.add_documents([vocab])
for doc in df_topic["text_tokenized"]:
    dictionary.add_documents([doc])

corpus = [dictionary.doc2bow(doc) for doc in df_topic["text_tokenized"]]
lda_model = LdaModel(corpus, num_topics=5, id2word=dictionary)

lda_set = {}
for idx, sample_bow in enumerate(corpus):
    topic_distribution = lda_model.get_document_topics(sample_bow, minimum_probability=0)
    feature_vector = [score for _, score in topic_distribution]
    lda_set[idx] = feature_vector

lda_set_tensor = torch.tensor([lda_set[idx] for idx in sorted(lda_set.keys())])
torch.save(lda_set_tensor, base_dir + 'lda_set.pt')

# torch.save(lda_set_tensor, base_dir + 'lda_set.pt')

"""Retrieve train, val, test df"""

import ast
train_df = pd.read_csv(base_dir + 'train_df.csv')
val_df = pd.read_csv(base_dir + 'val_df.csv')
test_df = pd.read_csv(base_dir + 'test_df.csv')

train_df['text_tokenized'] = train_df['text_tokenized'].apply(lambda x: ast.literal_eval(x))
val_df['text_tokenized'] = val_df['text_tokenized'].apply(lambda x: ast.literal_eval(x))
test_df['text_tokenized'] = test_df['text_tokenized'].apply(lambda x: ast.literal_eval(x))

print(train_df.columns)
train_size = train_df.shape[0]
val_size = val_df.shape[0]
test_size = test_df.shape[0]

print("Dataset shape")
print("train size", train_df.shape)
print("val size", val_df.shape)
print("test size", test_df.shape)


train_dataset = CustomTextDataset(train_df, tokenizer, max_length, 'train')
val_dataset = CustomTextDataset(val_df, tokenizer, max_length, 'val')
test_dataset = CustomTextDataset(test_df, tokenizer,max_length,'test')

"""Create embedding matrix"""

emb_lookup = {int(key): value for key, value in emb_lookup.items()}

embedding_weights = [emb_lookup[i] for i in range(len(emb_lookup))]
print(emb_lookup)


# for i in range(len(emb_lookup)) :
#   try :
#     print(emb_lookup[i])
#   except Exception as e :
#     print(f"didn't find {e}, index {i}")

embedding_weights = torch.tensor(embedding_weights).to(device)

train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_iterator = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)

for text_sample, input_ids, attention_masks, indices, label in train_iterator:

    print(f'Text Sample: {text_sample.shape}')  # Shape of text sample tensor
    print(f'Context Sample Input IDs: {input_ids.shape}')  # Shape of input IDs of context sample
    print(f'Context Sample Attention Mask: {attention_masks.shape}')  # Shape of attention mask of context sample
    print(f'Label: {label.shape}')  # Shape of label tensor
    print(f'Indices: {indices}')  # Indices


#load LDA Tensor
lda_set = torch.load(base_dir + 'lda_set.pt').to(device)
for idx, element in enumerate(lda_set):
    lda_set[idx] = element.to(device)

"""Define Model"""

from transformers import DistilBertForSequenceClassification, AdamW
import math

STRIDE = 2
class SarcasmModel(nn.Module):
    def __init__(self, vocab_size, cnn_dim, static_emb_dim=300, max_length=208, bert_alone=False):
        super(SarcasmModel, self).__init__()

        # frozen embedding layer
        self.embedding = nn.Embedding(vocab_size, static_emb_dim)
        self.embedding.weight = nn.Parameter(embedding_weights, requires_grad=False)

        self.conv_layers = nn.ModuleList()
        self.cnn_pool_layers = nn.ModuleList()
        self.cnn_relu = nn.ReLU()
        print(max_length)

        l_i = max_length
        n_i = 1
        cnn_output_dim = static_emb_dim
        num_layers = len(cnn_dim["conv"]["k"])
        for i in range(num_layers):
            if i == 0:
                in_channels = static_emb_dim
            else:
                in_channels = cnn_dim["conv"]["out"][i - 1]

            out_channels = cnn_dim["conv"]["out"][i]
            kernel_size = cnn_dim["conv"]["k"][i]

            conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=STRIDE, padding=0)
            val = (((l_i) - kernel_size) / STRIDE) + 1
            l_i = math.floor(val)
            n_i = out_channels

            pool_size = cnn_dim["pool"]["k"][i]
            pool = nn.MaxPool1d(kernel_size=pool_size, stride=STRIDE)

            val = ((l_i - pool_size) / STRIDE) + 1
            l_i = math.floor(val)

            self.conv_layers.append(conv)
            self.cnn_pool_layers.append(pool)

        self.transformer = DistilBertForSequenceClassification.from_pretrained(base_dir + 'pretrain_iac', num_labels=2)

        # for param in self.transformer.parameters():
        #     param.requires_grad = False

        factor = 4
        cnn_out_dim = l_i * n_i
        bert_out_dim = cnn_out_dim * factor
        lda_out_dim = 4

        self.bert_reduce = nn.Linear(768, cnn_out_dim * factor, bias=True)
        self.bert_act = nn.ReLU()
        total_out_dim = cnn_out_dim + lda_out_dim + bert_out_dim

        self.dropout = nn.Dropout(p=0.1)
        self.output_layer = nn.Linear(total_out_dim, 1)
        print(total_out_dim)
        self.sigmoid_layer = nn.Sigmoid()
        self.bert_alone = bert_alone

    def forward(self, x, input_ids, attention_masks, indices):
        embedded_x = self.embedding(x)
        input_ids = input_ids.squeeze(1)
        attention_masks = attention_masks.squeeze(1)

        outputs = self.transformer(input_ids, attention_mask=attention_masks, output_hidden_states=True)
        last_hidden_states = outputs.hidden_states[-1]
        bert_output = last_hidden_states[:, 0, :]

        if not self.bert_alone:
            cnn_input = embedded_x.permute(0, 2, 1)
            for conv_layer, pool_layer in zip(self.conv_layers, self.cnn_pool_layers):
                conv_output = conv_layer(cnn_input)
                conv_output = self.cnn_relu(conv_output)
                cnn_pool_output = pool_layer(conv_output)
                cnn_input = cnn_pool_output

            cnn_flatten = torch.flatten(cnn_input, 1)
            lda_tensor = lda_set[indices]
            concat_output = torch.cat((cnn_flatten, lda_tensor), dim=1)
            bert_output = self.bert_reduce(bert_output)
            bert_output = self.bert_act(bert_output)
            pre_output = torch.cat((bert_output, concat_output), dim=1)
        else:
            pre_output = bert_output

        fc_output = self.output_layer(pre_output)
        drop_output = self.dropout(fc_output)
        output = self.sigmoid_layer(drop_output)
        return output

"""Train Model"""

import torch.optim as optim
from tqdm import tqdm
import numpy as np

# Define CNN dimensions
cnn_dim = {
    "conv": {
        "k": [3, 3, 3],
        "out": [64, 32, 16]
    },
    "pool": {
        "k": [2, 2, 2]
    }
}


vocab_size = len(vocab_set)

model = SarcasmModel(vocab_size, cnn_dim,max_length=max_length, bert_alone=False)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=5e-5)

criterion = nn.BCELoss()
NUM_EPOCHS = 8


patience = 3
min_delta = 0.001
best_val_loss = np.Inf
epochs_no_improve = 0
early_stop = False
best_model_state_dict = None

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(train_iterator, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        x, input_ids, attention_masks, indices, y = batch
        optimizer.zero_grad()

        x = x.to(device)
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)
        indices = indices.to(device)
        y = y.to(device)

        outputs = model(x, input_ids, attention_masks, indices).float()
        y = y.float().unsqueeze(1)

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        predictions = (outputs > 0.5).float()
        total_correct += (predictions == y).sum().item()
        total_samples += y.size(0)
    epoch_loss = running_loss / len(train_iterator)
    epoch_accuracy = total_correct / total_samples
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}')

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for val_batch in val_iterator:
            x_val, input_ids_val, attention_masks_val, indices_val, y_val = val_batch
            x_val = x_val.to(device)
            input_ids_val = input_ids_val.to(device)
            attention_masks_val = attention_masks_val.to(device)
            indices_val = indices_val.to(device)
            y_val = y_val.to(device)

            outputs_val = model(x_val, input_ids_val, attention_masks_val, indices_val).float()
            y_val = y_val.float().unsqueeze(1)

            loss_val = criterion(outputs_val, y_val)
            val_loss += loss_val.item() * x_val.size(0)

    val_loss /= len(val_iterator.dataset)

    print(f'Validation Loss: {val_loss}')

    if val_loss < best_val_loss - min_delta:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_state_dict = model.state_dict()
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f'Early stopping after {epoch+1} epochs.')
        early_stop = True
        break

if best_model_state_dict is not None:
    torch.save(best_model_state_dict, base_dir + 'best_model2.pth')

"""Evaluate Model"""

#Evaluation
saved_model_path = base_dir + 'best_model2.pth'
saved_model_state_dict = torch.load(saved_model_path)

model = SarcasmModel(vocab_size, cnn_dim,max_length=max_length, bert_alone=False).to(device)

# model = SarcasmModel(vocab_size, cnn_dim, bert_alone=False)

model.load_state_dict(saved_model_state_dict)
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for batch in test_iterator:
        x, input_ids, attention_masks, indices, labels = batch
        x = x.to(device)
        indices = indices.to(device)
        labels = labels.to(device).float()
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        outputs = model(x, input_ids, attention_masks, indices).squeeze()
        predicted = (outputs > 0.5).float()
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

# Calculate accuracy
accuracy = total_correct / total_samples
print(f"Accuracy: {accuracy}")

"""Separate tests for twitter and reddit"""

reddit_df = test_df[:reddit_size]
twitter_df = test_df[reddit_size:]
reddit_dataset = CustomTextDataset(reddit_df, tokenizer, max_length, 'reddit')
twitter_dataset = CustomTextDataset(twitter_df, tokenizer, max_length, 'twitter')

reddit_iterator = DataLoader(reddit_dataset, batch_size=BATCH_SIZE, shuffle=False)
twitter_iterator = DataLoader(twitter_dataset, batch_size=BATCH_SIZE, shuffle=False)

"""Reddit"""

#Evaluation
saved_model_path = base_dir + 'best_model.pth'
saved_model_state_dict = torch.load(saved_model_path)

vocab_size = len(vocab_set)
# Define CNN dimensions
cnn_dim = {
    "conv": {
        "k": [3, 3, 3],
        "out": [64, 32, 16]
    },
    "pool": {
        "k": [2, 2, 2]
    }
}


model = SarcasmModel(vocab_size, cnn_dim,max_length=max_length, bert_alone=False).to(device)

model.load_state_dict(saved_model_state_dict)
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for batch in reddit_iterator:
        x, input_ids, attention_masks, indices, labels = batch
        x = x.to(device)
        indices = indices.to(device)
        labels = labels.to(device).float()
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        outputs = model(x, input_ids, attention_masks, indices).squeeze()
        predicted = (outputs > 0.5).float()
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

# Calculate accuracy
accuracy = total_correct / total_samples
print(f"Reddit accuracy: {accuracy}")

#Evaluation
saved_model_path = base_dir + 'best_model2.pth'
saved_model_state_dict = torch.load(saved_model_path)

model = SarcasmModel(vocab_size, cnn_dim,max_length=max_length, bert_alone=False).to(device)

model.load_state_dict(saved_model_state_dict)
model.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
    for batch in twitter_iterator:
        x, input_ids, attention_masks, indices, labels = batch
        x = x.to(device)
        indices = indices.to(device)
        labels = labels.to(device).float()
        input_ids = input_ids.to(device)
        attention_masks = attention_masks.to(device)

        outputs = model(x, input_ids, attention_masks, indices).squeeze()
        predicted = (outputs > 0.5).float()
        total_correct += (predicted == labels).sum().item()
        total_samples += labels.size(0)

# Calculate accuracy
accuracy = total_correct / total_samples
print(f"Twitter accuracy: {accuracy}")
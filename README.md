# Chatbot-using-RNN
This Repository contains all assignments and submission made during the course of GDSE project Chatbot using RNN

## Setup
First, necessary libraries are imported and the input data files are located.

```python
import numpy as np
import pandas as pd
import os
import nltk
import tensorflow as tf
from keras.layers import Input, Activation, Dense, Dropout, Embedding, Permute, LSTM
from keras.layers.merge import add, concatenate, dot
from keras.models import Model, Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils.data_utils import get_file
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import tarfile
import re
from functools import reduce

# Ensure nltk tokenizer is available
nltk.download('punkt')

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
```
## Data Preparation
The dataset is downloaded, extracted, and tokenized. Functions are defined to parse and vectorize the stories and questions.

```python
path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
tar = tarfile.open(path)

def tokenize(sent):
    return nltk.word_tokenize(sent)

def parse_stories(lines):
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f):
    data = parse_stories(f.readlines())
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data]
    return data

challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))
```
## Vocabulary Creation
The vocabulary is created from the training and test stories.

```python
vocab = set()
for story, question, answer in train_stories + test_stories:
    vocab |= set(story + question + [answer])
vocab = sorted(vocab)
vocab_size = len(vocab) + 1

word_to_idx = {w: idx for idx, w in enumerate(vocab)}
idx_to_word = {idx: w for idx, w in enumerate(vocab)}

story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
```
## Data Vectorization
The stories and questions are vectorized.

```python
def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return (pad_sequences(X, maxlen=story_maxlen),
            pad_sequences(Xq, maxlen=query_maxlen), np.array(Y))

inputs_train, queries_train, answers_train = vectorize_stories(train_stories, word_to_idx, story_maxlen, query_maxlen)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories, word_to_idx, story_maxlen, query_maxlen)
```
## Model Definition
The model is defined using Keras functional API.

```python
train_epochs = 100
batch_size = 32
embed_size = 50
lstm_size = 64
dropout_rate = 0.30

input_sequence = Input((story_maxlen,))
question = Input((query_maxlen,))

# Encoders
input_encoder_m = Sequential()
input_encoder_m.add(Embedding(input_dim=vocab_size, output_dim=embed_size))
input_encoder_m.add(Dropout(dropout_rate))

input_encoder_c = Sequential()
input_encoder_c.add(Embedding(input_dim=vocab_size, output_dim=query_maxlen))
input_encoder_c.add(Dropout(dropout_rate))

question_encoder = Sequential()
question_encoder.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=query_maxlen))
question_encoder.add(Dropout(dropout_rate))

input_encoded_m = input_encoder_m(input_sequence)
input_encoded_c = input_encoder_c(input_sequence)
question_encoded = question_encoder(question)

match = dot([input_encoded_m, question_encoded], axes=-1, normalize=False)
match = Activation('softmax')(match)

response = add([match, input_encoded_c])
response = Permute((2, 1))(response)

answer = concatenate([response, question_encoded])
answer = LSTM(lstm_size)(answer)
answer = Dropout(dropout_rate)(answer)
answer = Dense(vocab_size)(answer)
answer = Activation('softmax')(answer)

model = Model([input_sequence, question], answer)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())
plot_model(model)
```

## Model Training
The model is trained on the vectorized training data and validated on the vectorized test data.

```python
history = model.fit([inputs_train, queries_train], answers_train, 
                    batch_size=batch_size, 
                    epochs=train_epochs,
                    validation_data=([inputs_test, queries_test], answers_test))

model.save('model.h5')
```
## Model Prediction
The model is tested on a few examples from the test data.

```python
for i in range(0, 10):
    current_inp = test_stories[i]
    current_story, current_query, current_answer = vectorize_stories([current_inp], word_to_idx, story_maxlen, query_maxlen)
    current_prediction = model.predict([current_story, current_query])
    current_pred = idx_to_word[np.argmax(current_prediction)]
    ques = " ".join(current_inp[1])
    print(f"Question : {ques} \nAnswer : {current_pred}")
    print("--------------------------------")
```
### Data set used : The bAbi Dialogue Dataset 

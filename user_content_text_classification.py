# %%
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import plot_model
from tensorflow.keras import Sequential
import pandas as pd
import numpy as np
import re, datetime, os

# %%
# Step 1: Data Loading
df = pd.read_csv("https://raw.githubusercontent.com/susanli2016/PyCon-Canada-2019-NLP-Tutorial/master/bbc-text.csv")
text = df['text']
target = df['category']

# %%
# Step 2: Data Inspection & Visualization
df.info()
print(df.isna().sum())          # To check NaN but doesnt appear
print(df.duplicated().sum())    # have 99 duplicated data need to be remove

print(df['text'][10])

# %%
# Step 3: Data Cleaning
temp = []
for index, txt in enumerate(text) :
   text[index] = re.sub('[^a-zA-Z]', ' ', txt)
   temp.append(len(text[index].split()))
   
print(np.median(temp))

# %%
df1 = pd.concat([text,target], axis=1)
df1 = df1.drop_duplicates()

# %%
# Step 4: Feature Selection 
text = df1['text']          # Features --> X
target = df1['category']    # Target --> y

# %%
# Step 5: Data Preprocessing
# Features (X) Preparation
# Tokenizer
num_words = 5000
tokenizer = Tokenizer(num_words=num_words, oov_token='<OOV>')
tokenizer.fit_on_texts(text)

word_index = tokenizer.word_index
train_sequences = tokenizer.texts_to_sequences(text)

# Padding + Truncatting
train_sequences = pad_sequences(train_sequences, maxlen=50, padding='post', truncating='post')

# %%
# Target label (y) preprocessing
ohe = OneHotEncoder(sparse=False)
train_target = ohe.fit_transform(target[::,None])

# %%
# train test split
X_train, X_test, y_train, y_test = train_test_split(train_sequences, train_target)

# %%
# Model Development
model = Sequential()
model.add(Embedding(num_words, 64))
model.add(Bidirectional(LSTM(64, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Bidirectional(LSTM(32, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Dense(5))
model.add(Bidirectional(LSTM(16)))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))
model.summary()

plot_model(model, show_shapes=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='acc')

# Callbacks
log_dir =  os.path.join(os.getcwd(), 'logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb = TensorBoard(log_dir=log_dir)
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, callbacks=tb)

# %%
y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print(accuracy_score(y_true, y_pred))

# %%

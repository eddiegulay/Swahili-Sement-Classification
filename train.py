# load libraries
import pandas  as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

from keras_tuner import HyperModel, RandomSearch
from keras_tuner.engine.hyperparameters import HyperParameters


# load cleaned data
path = 'assets/'
train  = pd.read_csv(path + 'cleaned_training_set.csv')
train = train.sample(frac=1, random_state=42)


# Split the dataset into train and test sets
train_data = pd.DataFrame()
test_data = pd.DataFrame()

# Select 80% of each sentiment class for training
for sentiment in [-1, 0, 1]:
    sentiment_data = train[train['sentiment'] == sentiment]
    n_samples = int(0.8 * len(sentiment_data))
    train_data = train_data.append(sentiment_data[:n_samples])
    test_data = test_data.append(sentiment_data[n_samples:])

# Shuffle the train and test sets again
train_data = train_data.sample(frac=1, random_state=42)
test_data = test_data.sample(frac=1, random_state=42)


# Initialize the tokenizer
tokenizer = Tokenizer()
# Specify the maximum number of words to be used
max_sequence_length = 64
# Fit the tokenizer on the training data
tokenizer.fit_on_texts(train_data['comment'])
vocab_size = len(tokenizer.word_index) + 1

# Convert training texts to sequences of tokens
train_sequences = tokenizer.texts_to_sequences(train_data['comment'])
# Pad training sequences to ensure uniform length
train_padded_sequences = pad_sequences(train_sequences, maxlen=max_sequence_length)

# Convert testing texts to sequences of tokens
test_sequences = tokenizer.texts_to_sequences(test_data['comment'])
# Pad testing sequences to ensure uniform length
test_padded_sequences = pad_sequences(test_sequences, maxlen=max_sequence_length)


num_classes = 2  # Number of sentiment classes
sentiment_mapping = [0, 1]

# Convert the sentiment labels to one-hot encoded vectors for training data
train_one_hot_labels = np.zeros((len(train_data), num_classes))
for i, sentiment in enumerate(train_data['sentiment']):
    index = sentiment_mapping[sentiment]
    train_one_hot_labels[i, index] = 1

# Convert the sentiment labels to one-hot encoded vectors for testing data
test_one_hot_labels = np.zeros((len(test_data), num_classes))
for i, sentiment in enumerate(test_data['sentiment']):
    index_ = sentiment_mapping[sentiment]
    test_one_hot_labels[i, index_] = 1


# Define the hypermodel
class SentimentAnalysisHyperModel(HyperModel):
    def __init__(self, vocab_size, max_sequence_length, small=False):
        self.vocab_size = vocab_size
        self.max_sequence_length = max_sequence_length
        self.small = small
    
    def build(self, hp):
        model = Sequential()
        if not self.small:
            model.add(Embedding(self.vocab_size, 136, input_length=self.max_sequence_length))
            model.add(LSTM(128, return_sequences=True))
            model.add(LSTM(102))
            model.add(Dense(hp.Int('layer_units', min_value=96, max_value=128, step=32), activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(3, activation='softmax'))
        else:
            model.add(Embedding(len(tokenizer.word_index) + 1, 128, input_length=max_sequence_length))
            model.add(Bidirectional(LSTM(hp.Int('layer_units', min_value=96, max_value=128, step=32), dropout=0.2, recurrent_dropout=0.2)))
            model.add(Dense(3, activation='softmax'))
            
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(1e-4),
            metrics=['accuracy']
        )
        return model

    def build_model(self, hp=128):
        model = Sequential()
        if not self.small:
            model.add(Embedding(self.vocab_size, 136, input_length=self.max_sequence_length))
            model.add(LSTM(128, return_sequences=True))
            model.add(LSTM(102))
            model.add(Dense(96, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(3, activation='softmax'))
        else:
            model.add(Embedding(len(tokenizer.word_index) + 1, 136, input_length=max_sequence_length))
            model.add(LSTM(hp, dropout=0.2, recurrent_dropout=0.2))
            model.add(Dense(3, activation='softmax'))
            
        model.compile(
            loss='categorical_crossentropy',
            optimizer=keras.optimizers.Adam(1e-4),
            metrics=['accuracy']
        )
        return model


# USE WHEN HYPER PARAMETERS ARE NOT WELL DEFINED
"""
# Define the hyperparameters search space
hypermodel = SentimentAnalysisHyperModel(vocab_size, max_sequence_length, False)
hyperparameters = HyperParameters()
hyperparameters.Int('layer_units', min_value=96, max_value=128, step=32)

# Perform random search hyperparameter tuning
tuner = RandomSearch(
    hypermodel,
    objective='val_accuracy',
    max_trials=5,
    directory='hyperparameter_tuning',
    project_name='swahili_sentiment_analysis_sarufi'
)

tuner.search(train_padded_sequences, train_one_hot_labels, validation_data=(test_padded_sequences, test_one_hot_labels), batch_size=4, epochs=2)

# Get the best hyperparameters
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hyperparameters)

"""

model = Sequential()
model.add(Embedding(vocab_size, 136, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(102))
model.add(Dense(96, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer=keras.optimizers.Adam(1e-4),
    metrics=['accuracy']
)


history = model.fit(train_padded_sequences, train_one_hot_labels, validation_data=(test_padded_sequences, test_one_hot_labels), batch_size=4, epochs=10)


# Save the model
best_model.save("model/hyper_sarufi_tunned_swahili_sentiment_rating.h5")

#save tokenizer
tokenizer_json = tokenizer.to_json()
with open('tokenizers/hyper_sarufi_tunned_swahili_sentiment_rating.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)
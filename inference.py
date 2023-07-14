import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import re


def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9 \-]", "", text)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) 
    text = re.sub(r"\d", "", text)
    text = text.lower()   
    text = re.sub(r"(.)\1+", r"\1", text)
    return text

def classify(text, model, tokenizer, max_sequence_length):
    text = clean_text(text)
    text_sequences = tokenizer.texts_to_sequences([text])
    text_padded_sequences = pad_sequences(text_sequences, maxlen=max_sequence_length)
    predicted_sentiment = model.predict(text_padded_sequences, verbose=0)
    sentiment_labels = {0: "Negative", 1: "Positive"}
    predicted_result = sentiment_labels[int(np.argmax(predicted_sentiment))]
    return predicted_result


def main():
    parser = argparse.ArgumentParser(description='Sentiment Classification Inference')
    parser.add_argument('text', type=str, help='Text for sentiment classification')
    args = parser.parse_args()
    model = load_model("model/hyper_sarufi_tunned_swahili_sentiment_rating.h5")
    with open('tokenizers/hyper_sarufi_tunned_swahili_sentiment_rating.json', 'r', encoding='utf-8') as f:
        tokenizer_json = f.read()
        tokenizer = tokenizer_from_json(tokenizer_json)
    max_sequence_length = 64
    predicted_sentiment = classify(args.text, model, tokenizer, max_sequence_length)
    print("Text:", args.text)
    print("Predicted Sentiment:", predicted_sentiment)

if __name__ == '__main__':
    main()

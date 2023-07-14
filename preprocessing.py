import pandas  as pd
import numpy as np
import re

path = 'assets/'
train  = pd.read_csv(path + 'neural_tech_swahili_sentiment.csv')

# Rename the columns in the DataFrame
train.rename(columns={'id':'train_id', 'text': 'comment', 'labels': 'sentiment'}, inplace=True)

train = train.drop_duplicates()
train = train.dropna()

def clean_text(text):
    # Remove special characters, punctuation, and non-Swahili characters
    text = re.sub(r"[^a-zA-Z0-9 \-]", "", text)
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    # Remove HTML tags (if available)
    text = re.sub(r"<.*?>", "", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # remove numbers
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text) 
    text = re.sub(r"\d", "", text)
    # set in lowercase
    text = text.lower()   
    
    # Remove consecutive duplicate characters (e.g., 'loooove' to 'love')
    text = re.sub(r"(.)\1+", r"\1", text)
    
    return text

# Apply text cleaning to the 'text' column
train['comment'] = train['comment'].apply(clean_text)

train.to_csv(path + 'cleaned_training_set.csv', index=False)
print(f"Cleaned dataset saved in cleaned_training_set. Load and use directly")
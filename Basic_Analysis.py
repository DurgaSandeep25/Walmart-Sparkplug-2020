import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import string
import nltk
from nltk.corpus import stopwords

def remove_punctuation(text):
    """custom function to remove the punctuation"""
    PUNCT_TO_REMOVE = string.punctuation
    return text.translate(str.maketrans('', '', PUNCT_TO_REMOVE))

def remove_stopwords(text):
    """custom function to remove the stopwords"""
    STOPWORDS = set(stopwords.words('english'))
    return " ".join([word for word in str(text).split() if word not in STOPWORDS])

def remove_emoji(text):
    
    '''
    Reference: https://gist.github.com/slowkow/7a7f61f495e3dbb7e3d767f97bd7304b
    '''
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_hashtags(text):
    return ' '.join(re.sub("(#[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())

def remove_username(text):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    

def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def preprocess_text(df):
    
    df["preprocess_text"] = df["text"].str.lower()
    df["preprocess_text"] = df["preprocess_text"].apply(lambda text: remove_urls(text))
    df["preprocess_text"] = df["preprocess_text"].apply(lambda text: remove_hashtags(text))
    df["preprocess_text"] = df["preprocess_text"].apply(lambda text: remove_username(text))
    df["preprocess_text"] = df["preprocess_text"].apply(lambda text: remove_punctuation(text))
    df["preprocess_text"] = df["preprocess_text"].apply(lambda text: remove_stopwords(text))
    df["preprocess_text"] = df["preprocess_text"].apply(lambda text: remove_emoji(text))
    
    return df





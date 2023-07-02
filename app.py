# import pickle
import gradio as gr
# import pandas as pd
# import numpy as np
# import re

# import nltk
# from nltk.stem.porter import *
# stemmer = PorterStemmer()
# from nltk.corpus import stopwords

# from keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# from keras.models import load_model

from transformers import pipeline


def preprocessor(df):
    def remove_pattern(input_txt, pattern):
        r = re.findall(pattern, input_txt)
        for i in r:
            input_txt = re.sub(i,'',input_txt)
        return input_txt
    df['Tweet'] = np.vectorize(remove_pattern)(df['Tweet'], '@[\w]*')

    # Converting the tweets comments in array by splitting them
    df['Tweet'] = df['Tweet'].apply(lambda x: x.split())

    # Converting all the words to their root words Eg:runner,running-->run
    df['Tweet'] = df['Tweet'].apply(lambda x: [stemmer.stem(i) for i in x])

    nltk.download('stopwords')
    stop=stopwords.words('english')

    # Removing all the stop words from tweets and store it in stopword_x
    stopword_x=df['Tweet'].apply(lambda x: [item for item in x if item not in stop])

    for i in range(len(stopword_x)):
        stopword_x[i] = ' '.join(stopword_x[i])
        stopword_x.head()

    X=stopword_x
    max_words=41157
    max_len=217

    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X)
    sequ = tokenizer.texts_to_sequences(X)
    tweets = pad_sequences(sequ, maxlen=max_len)
    return tweets


labels=["Neutral","Postive","Extremely Negative","Extremely Positive","Negative"]

def sentiment_analysis(tweet):
    sentiment_pipeline = pipeline("sentiment-analysis")
    pred=sentiment_pipeline(tweet)
    res=pred[0]["label"]+" & "+str(pred[0]["score"])

    return res




sample=["Today I am happy as day is sunny.","The day is bad so my mood is off.",
        "I am happy.","Not in mood to talk."]

demo = gr.Interface(fn=sentiment_analysis, inputs="text", outputs="text",examples=sample)

demo.launch()

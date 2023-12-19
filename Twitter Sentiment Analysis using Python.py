#!/usr/bin/env python
# coding: utf-8

# # Twitter Sentiment Analysis using Python

# witter is one of those social media platforms where people are free to share their opinions on any topic. Sometimes we see a strong discussion on Twitter about someone’s opinion that sometimes results in a collection of negative tweets. With that in mind, if you want to learn how to do sentiment analysis of Twitter

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import re
import nltk
import nltk

data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/twitter.csv")
print(data.head())


# In[4]:


nltk.download("stopword")
stemmer =nltk.SnowballStemmer("english")
from nltk.corpus import stopwords
import string
stopword =set(stopwords.words('english'))
def clean(text):
    text = str(text).lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)
    text = [word for word in text.split(' ') if word not in stopword]
    text=" ".join(text)
    text = [stemmer.stem(word) for word in text.split(' ')]
    text=" ".join(text)
    return text
data["tweet"]=data["tweet"].apply(clean)


# Now, the next step is to calculate the sentiment scores of these tweets and assign a label to the tweets as positive, negative, or neutral. Here is how you can calculate the sentiment scores of the tweets:
# 
# 

# In[7]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download("vader_lexicon")
sentiments = SentimentIntensityAnalyzer()
data["positive"] =[sentiments.polarity_scores(i)["pos"] for i in data['tweet']]
data["negative"] =[sentiments.polarity_scores(i)["neg"] for i in data['tweet']]
data["netural"] =[sentiments.polarity_scores(i)["neu"] for i in data['tweet']]


# Now I will only select the columns from this data that we need for the rest of the task of Twitter sentiment analysis:

# In[8]:


data =data[["tweet","positive","negative","netural"]]
print(data.head())


# In[10]:


x= sum(data["positive"])
y= sum(data["negative"])
z=sum(data["netural"])
def sentiment(a,b,c):
    if (a>b)and (a>c):
        print("positive")
    elif (b>a)  and  (b>c):
        print("negative")
    else:
        print("netural")
sentiment(x,y,z)


# In[11]:


print(x)
print(y)
print(z)


# In[ ]:





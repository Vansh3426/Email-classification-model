import pandas as pd 
import numpy as np 
import sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import spacy



df = pd.read_csv("Email-classification/spam.csv")

df["spam"] = df["Category"].apply(lambda x:1 if x== "spam" else 0 )

train = df["Message"]
target = df["spam"]

Xtrain ,Xtest , ytrain ,ytest  = train_test_split(train,target , test_size=0.2)

pipe =Pipeline([("vectorizer",CountVectorizer()),
                ("model", MultinomialNB())])


model = pipe.fit(Xtrain , ytrain)

pred = pipe.predict(Xtest)

print(classification_report(ytest,pred))


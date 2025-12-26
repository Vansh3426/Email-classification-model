import pandas as pd 
import numpy as np 
import sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report


df = pd.read_csv("Email-classification/spam.csv")

df["spam"] = df["Category"].apply(lambda x:1 if x== "spam" else 0 )

train = df["Message"]
target = df["spam"]

Xtrain ,Xtest , ytrain ,ytest  = train_test_split(train,target , test_size=0.2)

v = CountVectorizer()
Xtrain_cv = v.fit_transform(Xtrain)

model =MultinomialNB()
model.fit(Xtrain_cv,ytrain)


#### PREDICTION ON TEST DATASET ####

Xtest_cv =v.transform(Xtest)
pred = model.predict(Xtest_cv)

print(classification_report(ytest,pred))


#### EXAMPLE ####

Email=[
    "You will get 100$ in your account if you subscribe to our channel. ",
    "Hey claim your offer before it Expires, 10000$ for sign up. ",
    "Hello my dear , Hows you doing ?.",
    "Hey What's up bro , meet me on monday.",
    "Can you give me some cash around 200$."

 ]

Email_cv =v.transform(Email)
pred = model.predict(Email_cv)

print(pred)
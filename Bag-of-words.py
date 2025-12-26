import pandas as pd 
import numpy as np 
import sklearn 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split


df = pd.read_csv("Email-classification/spam.csv")

# print(df[0:5])

df["spam"] = df["Category"].apply(lambda x:1 if x== "spam" else 0 )

# print(df[0:10])

train = df.drop(columns=['spam'])
target = df["spam"]



Xtrain ,Xtest , ytrain ,ytest  = train_test_split(train,target , test_size=0.2)



# print(Xtrain[0:5])
# print(Xtest[0:5])
# print(ytrain[0:5])
# print(ytest[0:5])

# print(type(Xtrain))


v = CountVectorizer()

Xtrain_cv = v.fit_transform(Xtrain)

print(type(Xtrain_cv))


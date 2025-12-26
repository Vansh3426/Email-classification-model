import pandas as pd 
import numpy as np 
import sklearn 

df = pd.read_csv("Email-classification/spam.csv")

# print(df[0:5])

df["spam"] = df["Category"].apply(lambda x:1 if x== "spam" else 0 )

print(df[0:10])
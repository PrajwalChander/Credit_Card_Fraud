import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

credit_df=pd.read_csv("/content/creditcard.csv")

credit_df.head()

credit_df.shape

credit_df.info()

credit_df.isnull().sum()

credit_df['Class'].value_counts()

x=credit_df[credit_df.Class == 0]

y=credit_df[credit_df.Class == 1]

x

y

credit_df.groupby('Class').mean()

#need to balence the data
x1=x.sample(n=492)

x1.shape

y.shape

final_df=pd.concat([x1,y],axis=0)

final_df.shape

final_df.head()

final_df.tail()

final_df['Class'].value_counts()

final_df.groupby('Class').mean()

#Splitting the data
a=final_df.drop(columns='Class',axis=1)
b=final_df['Class']

a.head()

b.head()

a_train,a_test,b_train,b_test=train_test_split(a,b,test_size=0.2,stratify=b,random_state=2)

a_train.shape

a_test.shape

b_train.shape

b_test.shape

#USING LOGISTIC REGRESSION
model=LogisticRegression()

model.fit(a_train,b_train)

#MODEL EVALUATION->ACCURACY SCORE
a_train_prediction=model.predict(a_train)
a_train_prediction_score=accuracy_score(a_train_prediction,b_train)
a_train_prediction_score

a_test_prediction=model.predict(a_test)
a_test_prediction_score=accuracy_score(a_test_prediction,b_test)
a_test_prediction_score


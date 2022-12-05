import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.header('Train your own Machine Learning model')

df = pd.read_csv('heart.csv')
st.dataframe(df)

X = df.drop('output', axis=1)
y = df['output']

split = st.sidebar.slider('Choose the test size', 1, 99, 10)
splittrain = 100 - split
split2 = split/100

st.write("Your train size : ", splittrain)
st.write("Your test size : ", split)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=split2,random_state=0)

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

ml = st.sidebar.selectbox("Choose your Machine Learning Model :",('Decision Tree', 'Random Forest', 'Logistic Regression'))

if ml == 'Decision Tree':
  from sklearn.tree import DecisionTreeClassifier
  dtree = DecisionTreeClassifier(random_state=0)
  dtree.fit(X_train, y_train)
  y_pred = dtree.predict(X_test)
  acc = round(accuracy_score(y_test, y_pred)*100 ,2)
  y_pred = dtree.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  fig = plt.figure(figsize=(5,5))
  sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  f1 = f1_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
elif ml == 'Random Forest':
  from sklearn.ensemble import RandomForestClassifier
  rfc = RandomForestClassifier(random_state=0)
  rfc.fit(X_train, y_train)
  y_pred = rfc.predict(X_test)
  acc = round(accuracy_score(y_test, y_pred)*100 ,2)
  y_pred = rfc.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  fig = plt.figure(figsize=(5,5))
  sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  f1 = f1_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)
elif ml == 'Logistic Regression':
  from sklearn.linear_model import LogisticRegression
  lr = LogisticRegression(random_state=0)
  lr.fit(X_train, y_train)
  y_pred = lr.predict(X_test)
  acc = round(accuracy_score(y_test, y_pred)*100 ,2)
  y_pred = lr.predict(X_test)
  cm = confusion_matrix(y_test, y_pred)
  fig = plt.figure(figsize=(5,5))
  sns.heatmap(data=cm,linewidths=.5, annot=True,square = True,  cmap = 'Blues')
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')
  f1 = f1_score(y_test, y_pred)
  prec = precision_score(y_test, y_pred)
  recall = recall_score(y_test, y_pred)

st.write("**Algorithm Accuracy in (%)**")
st.info(acc)
st.write("**Precision**")
st.info(prec)
st.write("**Recall**")
st.info(recall)
st.write("**F-1 Score**")
st.info(f1)
st.write("**Confusion Matrix**")
st.write(fig)

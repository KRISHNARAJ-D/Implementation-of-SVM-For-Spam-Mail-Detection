# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.

## Program:
```
Program to implement the SVM For Spam Mail Detection..
Developed by: KRISHNARAJ D
RegisterNumber: 212222230070
```
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn import svm
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv('spam.csv', encoding='ISO-8859-1')
df.head()

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['v2'])
y = df['v1']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = svm.SVC (kernel='linear') 
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print("Accuracy: ", accuracy_score (y_test, predictions)) 
print("Classification Report: ")
print(classification_report (y_test, predictions))
```
## Output:
## Head() :
![328394104-1ee3888b-6d43-4ab3-acee-9f54c74d3891](https://github.com/KRISHNARAJ-D/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119559695/7e3cf1f8-621f-4139-8048-016641a36cda)


## Kernel Model:
![328394130-43ba6821-c447-4f84-b3c1-2144ca2f8664](https://github.com/KRISHNARAJ-D/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119559695/8d4898b0-3d68-4883-abbb-6088192e6571)

## Accuracy and Classification Report :  
![328394153-26c0265b-c0ba-4b24-a43f-d874ae69f62d](https://github.com/KRISHNARAJ-D/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119559695/caa1d25e-5523-4426-af15-e247206a4b75)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.Import the required packages and print the present data.

2.Print the placement data and salary data.

3.Find the null and duplicate values.

4.Using logistic regression find the predicted values of accuracy , confusion matrices.

5.Display the results

## Program:


### Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
#### Developed by: THAMIZH KUMARAN
#### RegisterNumber:  212223240166

```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1

x=data1.iloc[:, :-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression 
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:

# DATA HEAD
![Screenshot 2025-03-25 174928](https://github.com/user-attachments/assets/292ece27-1ad9-4c0f-adc0-2cbe40d3d4af)


# DATA1 HEAD
![Screenshot 2025-03-25 174936](https://github.com/user-attachments/assets/a54c9d1e-7ab4-4793-969d-143834b2e00f)


# ISNULL().SUM()
![Screenshot 2025-03-25 174944](https://github.com/user-attachments/assets/f1764e04-6484-4b00-a4d8-5acb8372eb77)


# DATA DUPLICATE
![Screenshot 2025-03-25 175009](https://github.com/user-attachments/assets/b128aecc-3a47-40ac-892b-f1303b828951)


# PRINT DATA
![Screenshot 2025-03-25 175019](https://github.com/user-attachments/assets/6cdb39ca-b5bc-4b8d-8ea2-173683297dea)


# STATUS
![Screenshot 2025-03-25 175028](https://github.com/user-attachments/assets/7e36ef91-d76e-4ccd-9d71-ad50c1116b97)


# Y_PRED
![Screenshot 2025-03-25 175035](https://github.com/user-attachments/assets/1459fbf0-d02e-468d-a654-efa3dd3b4921)


# MODEL
![Screenshot 2025-03-25 175044](https://github.com/user-attachments/assets/29af6166-0c8d-41e4-863b-894490761834)


# ACCURACY
![Screenshot 2025-03-25 175051](https://github.com/user-attachments/assets/ac158d16-78f2-4457-8025-17078e6ef608)


# CONFUSION MATRIX
![Screenshot 2025-03-25 175056](https://github.com/user-attachments/assets/006e8cb5-0332-46d5-bed1-c2be77b9a7c8)


# CLASSIFICATION
![Screenshot 2025-03-25 175103](https://github.com/user-attachments/assets/f98a8c75-463b-4dde-9c1a-c5a0ea3b9f0c)


# LR PREDICT
![Screenshot 2025-03-25 175123](https://github.com/user-attachments/assets/13806e50-5c73-4537-b39d-99f8faffd483)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.

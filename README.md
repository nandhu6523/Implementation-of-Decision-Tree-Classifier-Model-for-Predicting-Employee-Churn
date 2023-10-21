# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.import pandas module and import the required data set.
 
 2.Find the null values and count them.
 
 3.Count number of left values.
 
 4.From sklearn import LabelEncoder to convert string values to numerical values.
 
 5.From sklearn.model_selection import train_test_split.
 
 6.Assign the train dataset and test dataset.
 
 7.From sklearn.tree import DecisionTreeClassifier.
 
 8.Use criteria as entropy.
 
 9.From sklearn import metrics.
 
 10.Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S Nandhini
RegisterNumber: 212222220028 
*/

import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()

data.isnull().sum()

data["left"].value_counts

from sklearn.preprocessing import LabelEncoder
le= LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()

x= data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]

x.head()
y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state = 100)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)

y_pred = dt.predict(x_test)
from sklearn import metrics

accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2,9,6]])

```

## Output:
Data.head():

![Screenshot 2023-10-21 111140](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123856724/41fdb6ee-5d75-4ef3-8aa5-ddb815426016)

Data.info():

![Screenshot 2023-10-21 111153](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123856724/37c0990f-2b83-4748-966d-095bedd5247e)

isnull() and sum():

![Screenshot 2023-10-21 111211](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123856724/28d932ff-c3d8-4ad6-8714-76b6b47d3e7b)

Data value counts:

![Screenshot 2023-10-21 111256](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123856724/036fbb8b-c76f-42a9-bb18-cec4cf9da4ed)

Data.head() for salary:

![Screenshot 2023-10-21 113127](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123856724/d9131b67-d8f0-4f3e-9009-442bb5664685)

Accuracy Value:

![Screenshot 2023-10-21 111446](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123856724/984b40ea-8a29-4791-b151-982d1eec9024)

Data Prediction:

![Screenshot 2023-10-21 111519](https://github.com/nandhu6523/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/123856724/ca6a0606-4c6b-4347-99c8-39bf9ed279e5)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.

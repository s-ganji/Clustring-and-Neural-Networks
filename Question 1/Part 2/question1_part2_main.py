import random

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import sklearn.datasets
from sklearn.metrics import accuracy_score

from question1_part2 import NeuralNetwork

breast_cancer = sklearn.datasets.load_breast_cancer()

    # convert the data to pandas dataframe
data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
data["class"] = breast_cancer.target
data.head()
data.describe()

    # perform scaling on the data.
X = data.drop("class", axis=1)
Y = data["class"]
X = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X, columns=data.drop("class", axis=1).columns)

    # train test split.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

units = X_train.shape[1]
n_n = NeuralNetwork([X_train.shape[1],1])
n_n.fit(X_train,Y_train)
prediction=[]

for i in range(X_test.shape[0]):
    prediction.append(n_n.predict(X_test.iloc[i]))

    # checking the accuracy of the model
print("test accuracy:")
print(accuracy_score(prediction, Y_test))


import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from question1_part1 import Perceptron

#load the breast cancer data
breast_cancer = sklearn.datasets.load_breast_cancer()

#convert the data to pandas dataframe
data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
data["class"] = breast_cancer.target
data.head()
data.describe()

#plotting a graph to see class imbalance
data['class'].value_counts().plot(kind = "barh")
plt.xlabel("Count")
plt.ylabel("Classes")
plt.show(block= False)

#perform scaling on the data.
X = data.drop("class", axis = 1)
Y = data["class"]
X = MinMaxScaler().fit_transform(X)
X = pd.DataFrame(X, columns=data.drop("class",axis = 1).columns)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)

# Building Model
perceptron = Perceptron()
wt_matrix = perceptron.fit(X_train, Y_train, 10000, 0.3)

# Run and Evaluate
# making predictions on test data
Y_pred_test = perceptron.predict(X_test)

# checking the accuracy of the model
print("test accuracy :")
print(accuracy_score(Y_pred_test, Y_test))


# # Further optimization
# # Vary the train-test size split and see if accuracy changes.
# for i in range(1,5):
#     test_size = i/10
#     print()
#     print("test size: %f"%test_size)
# #train test split.
#     X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, stratify=Y, random_state=1)
#
#     # Building Model
#     perceptron = Perceptron()
#     wt_matrix = perceptron.fit(X_train, Y_train, 10000, 0.3)
#
#     # making predictions on test data
#     Y_pred_test = perceptron.predict(X_test)
#
#     # checking the accuracy of the model
#     print("test accuracy :")
#     print(accuracy_score(Y_pred_test, Y_test))


# # Choose larger  ’learning rates’.test on the model and visualize the change in accuracy.
# acc={}
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y, random_state=1)
# # Building Model
# perceptron = Perceptron()
# for i in range(1, 4):
#     print("lr= %f" % (i / 10))
#     wt_matrix = perceptron.fit(X_train, Y_train, 10000, i / 10)
#
#     # making predictions on test data
#     Y_pred_test = perceptron.predict(X_test)
#
#     # checking the accuracy of the model
#     acc[i]=accuracy_score(Y_pred_test, Y_test)
#     print("test accuracy :")
#     print(accuracy_score(Y_pred_test, Y_test))
#
# lists= sorted(acc.items())
# a,b = zip(*lists)
# # plot the accuracy values over epochs
# plt.figure(figsize=(16, 8))
# plt.plot(a,b)
# plt.xlabel("learning rates")
# plt.ylabel("Accuracy")
# plt.ylim([0, 1])
# plt.show(block=False)
#
#





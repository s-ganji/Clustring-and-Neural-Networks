import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


class Perceptron:
    # constructor
    def __init__(self):
        self.w = None
        self.b = None


    # predictor to predict on the data based on w
    def predict(self, X):
        Y = []
     # # Take random weights in your model and test the result.
     #    self.w = np.random.randint(low=10,high=100,size=X.shape[1])
     # because model function is predicting y for each x
        for i in range(X.shape[0]):
            if np.dot(self.w, X.iloc[i]) >= self.b:
                result = 1
            else:
                result = 0
            Y.append(result)
        return np.array(Y)

    # find best weights and b
    def fit(self, X, Y, epochs, lr):
        self.w = np.ones(X.shape[1])
        self.b = 0
        accuracy = {}
        max_accuracy = 0
        wt_matrix = []
        # for all epochs
        for i in range(epochs):
            # for each (x,y)
            for count in range(X.shape[0]):
                if np.dot(self.w, X.iloc[count] ) >= self.b:
                    y_pred = 1
                else:
                    y_pred=0
                # compare real class value and prediction and update weights till congestion
                if Y.iloc[count] == 1 and y_pred == 0:
                    self.w = self.w + lr * X.iloc[count]
                    self.b = self.b - lr * 1

                elif Y.iloc[count] == 0 and y_pred == 1:
                    self.w = self.w - lr * X.iloc[count]
                    self.b = self.b + lr * 1

            wt_matrix.append(self.w)
            accuracy[i] = accuracy_score(self.predict(X), Y)
            if (accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                chkptw = self.w
                chkptb = self.b

        # checkpoint (Save the weights and b value)
        self.w = chkptw
        self.b = chkptb
        print("max accuracy in train:")
        print(max_accuracy)

        lists= sorted(accuracy.items())
        a,b = zip(*lists)
        # plot the accuracy values over epochs
        plt.figure(figsize=(16, 8))
        plt.plot(a,b)
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        # plt.ylim([0, 1])
        plt.show(block=False)

        #return the weight matrix, that contains weights over all epochs
        return np.array(wt_matrix)



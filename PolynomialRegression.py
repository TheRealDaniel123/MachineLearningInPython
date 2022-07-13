import pandas as pd
import matplotlib.pyplot

import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as linalg
import math
from sklearn.model_selection import train_test_split

#Reads in data
data = pd.read_csv('Task1 - dataset - pol_regression.csv')

#Sets variables for x column and y column
x = data['x']
y = data['y']

#Sorts the data
sortedX = np.argsort(x)
x = x[sortedX]
y = y[sortedX]

xTrain, xTest, yTrain, yTest = train_test_split(x, y,test_size=0.3,shuffle = True) #Creates train teest split

#Creates Matrix
def getPolynomialDataMatrix(x, degree):
    #Makes an array of a given shape filled with ones
    X = np.ones(x.shape)
    
    for i in range(1,degree + 1):
        X = np.column_stack((X, x ** i)) #Makes a new array with the ones stacked with
                                        #the x column with the values to the power of the degree
    return X
    

#Test different polynomials with a varying degree
def pol_regression(features_train, y_train,degree):
    
    #Calculating for 0 degree - mean
    if degree == 0:
        zeroDegree=[]
        for i in range(features_train.size):
            zeroDegree.append(np.mean(y_train)) #Appends mean to an array
        w = zeroDegree
        
    else:   
        
        #Least squared method
        X = getPolynomialDataMatrix(features_train, degree) #Gets data matrix
    
        XX = X.transpose().dot(X)
        w = np.linalg.solve(XX, X.transpose().dot(y_train)) #Gets the inverse of XX times by the transpose of itself, Calculates alpha coefficents

    return w


def eval_pol_regression(parameters, x, y, degree):
    #New matrix
    new = getPolynomialDataMatrix(x, degree)
    if degree !=0:
        new=new.dot(parameters) #Times whole matrix by the parameters(weights)
    squaredArray = (new - y)**2
    mse = np.mean(squaredArray)
    rmse = math.sqrt(mse)
    return rmse #Returns root mean squared error


plt.figure()

#Plots Data Points
plt.plot(x,y, 'bo')
y0 = pol_regression(x,y,0)

#Plots for degree 0
plt.plot(x, y0, 'y')


w1 = pol_regression(x,y,1)
y1 = getPolynomialDataMatrix(x, 1).dot(w1)


#Plots for degree 1
plt.plot(x, y1, 'r')
#Plots for degree 2
w2 = pol_regression(x,y,2)
y2 = getPolynomialDataMatrix(x, 2).dot(w2)

#Plots for degree 2
plt.plot(x, y2, 'g')

w3 = pol_regression(x,y,3)
y3 = getPolynomialDataMatrix(x, 3).dot(w3)

#Plots for degree 3
plt.plot(x, y3, 'm')

w4 = pol_regression(x,y,6)
y4 = getPolynomialDataMatrix(x, 6).dot(w4)
#Plots for degree 6
plt.plot(x, y4, 'c')

w5 = pol_regression(x,y,10)
y5 = getPolynomialDataMatrix(x, 10).dot(w5)
#Plots for degree 10
plt.plot(x, y5, 'k')

plt.legend(("Training Points","$x$", "$x^2$","$x^3$","$x^6$","$x^10$"))


# task 1.3
#Predicting y value
trainEval = []
testEval = []
degreeEval = [0,1,2,3,6,10]
#Training
w0 = pol_regression(xTrain,yTrain,0)
w1 = pol_regression(xTrain,yTrain,1)
w2 = pol_regression(xTrain,yTrain,2)
w3 = pol_regression(xTrain,yTrain,3)
w4 = pol_regression(xTrain,yTrain,6)
w5 = pol_regression(xTrain,yTrain,10)

#Evaluating on training
y0 = eval_pol_regression(w0, xTrain, yTrain, 0)
y1 = eval_pol_regression(w1, xTrain, yTrain, 1)
y2 = eval_pol_regression(w2, xTrain, yTrain, 2)
y3 = eval_pol_regression(w3, xTrain, yTrain, 3)
y4 = eval_pol_regression(w4, xTrain, yTrain, 6)
y5 = eval_pol_regression(w5, xTrain, yTrain, 10)

#Appends training evaluation data to a list
trainEval.append(y0)
trainEval.append(y1)
trainEval.append(y2)
trainEval.append(y3)
trainEval.append(y4)
trainEval.append(y5)

#Evaluating on testing
y0 = eval_pol_regression(w0, xTest, yTest, 0)
y1 = eval_pol_regression(w1, xTest, yTest, 1)
y2 = eval_pol_regression(w2, xTest, yTest, 2)
y3 = eval_pol_regression(w3, xTest, yTest, 3)
y4 = eval_pol_regression(w4, xTest, yTest, 6)
y5 = eval_pol_regression(w5, xTest, yTest, 10)

#Appends testing evaluation data to a list
testEval.append(y0)
testEval.append(y1)
testEval.append(y2)
testEval.append(y3)
testEval.append(y4)
testEval.append(y5)

#Plots Evaluation
plt.figure()
plt.ylabel("Root Mean Squared Error")
plt.xlabel("Degree")

plt.title("Polynomial Evalutation")
plt.plot(degreeEval,testEval ,'r')
plt.plot(degreeEval,trainEval ,'b')
plt.legend(("Test","Train"),loc = 'upper right')


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import MinMaxScaler
import numpy as np

#Reads the data 
df = pd.read_csv("Task3 - dataset - HIV RVG.csv")

#Plots a box plot by the participant condition
dfBoxPlot = df[["Alpha", "Participant Condition"]]
dfBoxPlot.boxplot(by = "Participant Condition")

#Plots a density plot by the participant condition
dfDensityPlot = df[["Beta","Participant Condition"]].pivot(columns = "Participant Condition", values = "Beta")
dfDensityPlot.plot.density(by = "Participant Condition")


#Outputs summary statistics 
print(df.describe())

#Scaler to normalize the data between the values -1 and 1
scaler = MinMaxScaler(feature_range = (-1,1))

#Fits the scaler to the numerical dataset columns 
scaler = scaler.fit(df.iloc[:,0:8])
scaler = scaler.transform(df.iloc[:,0:8])

#List of different iteration values to alter the max iterations of the neural network and random forest model
neuralIterations = [100,200,300,400,500,600,700,800,900,1000]
forestIterations = [5,10]

#Lists to append the accuracy of each model
accuracyOfNeuralNetwork = []
accuracyOfRandomForest = []

xTrain, xTest, yTrain, yTest = train_test_split(df.iloc[:,0:8],df.iloc[:,8],test_size=0.1,shuffle = True) #Creates train teest split

#Loops through the neural network iterations
for i in neuralIterations:
    #Creates a neural network model with varying max iterations
    clf = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(500, 500), activation = "logistic",max_iter=i)
    #Fits the model to the training and testing set
    clf.fit(xTrain,yTrain)
    #Evaluates the accuracy of the model 
    evaluate = clf.score(xTest,yTest)
    print("Neural network iteration", i ," Accuracy: ",evaluate) #Outputs accuracy and number of iterations
    #Appends the accuracy to a list
    accuracyOfNeuralNetwork.append(evaluate)
#Loops through the random forest iterations
for i in forestIterations:
    #Creates a random forest model with varying 
    clf = RandomForestClassifier(max_depth=2, n_estimators = 1000,min_samples_leaf = i)
    #Fits the model to the training and testing set
    clf.fit(xTrain,yTrain)
    #Evaluates the accuracy of the model
    evaluate = clf.score(xTest,yTest)
    print("Random forest iteration", i ," Accuracy: ",evaluate)#Outputs accuracy and number of iterations
    #Appends the accuracy to a list
    accuracyOfRandomForest.append(evaluate)

#Plots the accuracy of the neural network against the number of iterations
plt.figure()
plt.plot(neuralIterations,accuracyOfNeuralNetwork)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")


#Plots the accuracy of the neural network against the number of iterations
plt.figure()
plt.plot(forestIterations,accuracyOfRandomForest)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")


#Cross validation
def crossValidate(clf,df,model,layers):
    
    #Cross validates 10 times 
    scores = cross_validate(clf, df.iloc[:,0:8], df.iloc[:,8],cv = 10)['test_score']
    print(model + layers +" Score: ",scores.mean()) #Outputs the mean of cross validation
    
#Creates three neural networks with hidden layers 50,500,1000
neuralClf1 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(50,50), activation = "logistic",max_iter=1000)
neuralClf2 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(500,500), activation = "logistic",max_iter=1000)
neuralClf3 = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(1000,1000), activation = "logistic",max_iter=1000)

#Creates three random forest classifiers with hidden layers 50,500,1000
ranForestClf1 = RandomForestClassifier(max_depth=2, random_state=0, n_estimators = 50,min_samples_leaf = 10)
ranForestClf2 = RandomForestClassifier(max_depth=2, random_state=0, n_estimators = 500,min_samples_leaf = 10)
ranForestClf3 = RandomForestClassifier(max_depth=2, random_state=0, n_estimators = 10000,min_samples_leaf = 10)

#Cross validates with varying models
crossValidate(neuralClf1,df,"Neural Network"," layers: 50")
crossValidate(neuralClf2,df,"Neural Network"," layers: 500")
crossValidate(neuralClf3,df,"Neural Network"," layers: 1000")

crossValidate(ranForestClf1,df,"Random Forest"," trees: 50")
crossValidate(ranForestClf2,df,"Random Forest"," trees: 500")
crossValidate(ranForestClf3,df,"Random Forest"," trees: 10000")





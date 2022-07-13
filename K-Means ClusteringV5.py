#imports the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Loads the data using pandas and returns a pandas data frame
def load_data(data):
    df = pd.read_csv(data)
    
    return df
 
#Calculates the euclidean distance between each centroid and every point
def compute_euclidean_distance(vec1,vec2):

    eucDistance = np.linalg.norm(vec1 - vec2) #Uses numpy library to calculate the euclidean distance of two points
    return eucDistance #Returns the euclidean distance

#Generates random centroids from randomly selected points in the data
def initialise_centroids(dataset,k):
    x = []
    for i in range(k): #Iterates k number of times
        randomCentroids = np.random.choice(len(dataset),4,replace=False)
        height = dataset['height'].values[randomCentroids[0]]
        tailLength = dataset['tail length'].values[randomCentroids[1]]
        legLength = dataset['leg length'].values[randomCentroids[2]]
        Nose = dataset['nose circumference'].values[randomCentroids[3]]
        x.append([height,tailLength,legLength,Nose])
    
    return np.array(x)
#Reassigns the centroids location based on the mean of the data points
def move_centroids(dataset, k, assignments):
    centroids = []
    for c in range(k):
        centroids.append(np.mean([dataset[x] for x in range(len(data)) if assignments[x] == c], axis = 0))# Find the mean of all the data points assigned to each centroid

    return centroids #Return as a list of centroids
    

def iteration_step(dataset,classification,centroids,k):
    iterationStep = 0
    
    #Iterates over each cluster
    for i in range(0,k):
        for j in range(len(classification)):
            if classification[j] == i:
                #Computes square euclidean distance and adds it to a variable
                iterationStep = iterationStep + (compute_euclidean_distance(dataset.to_numpy()[j], centroids[i])**2)
        
    
    return iterationStep                

#Assigns the points to the nearest centroid    
def assign_points(dataset,centroids,classification):
    
    #Iterates over the dataset and the centroids
    for i in dataset.to_numpy():
        eucDistance = []
        for n in range(len(centroids)):
    
            eucDistance.append(compute_euclidean_distance(centroids[n],i)) #Gets the euclidean distance of every centroid and every data point
            
        minimum = eucDistance.index(min(eucDistance)) #Assigns variable as the minimum distance
        classification.append(minimum) #Appends to a list of assignments
    return classification #Returns list of assignments
    

#Main function that runs the algorithm by using the k number of iterations and the dataset
def kmeans(dataset,k):
  
    #Declares a list for keep 
    iteration_steps = []
    
    
    #Assigns initial random centroids as a numpy array using the dataset and k number of iterations
    centroids = initialise_centroids(dataset,k)
    
    #Assigns a variable for oldcentroids so that the algorithm can see if the centroids have moved through each iteration 
    oldCentroids = initialise_centroids(dataset,k)
    
    #Compares the old and new centroids to iteratively run the algorithm whilst they are not the same
    comp = centroids == oldCentroids
    while comp.all() == False:
        classification = [] #List to keep track of centroid assignments, holds the classification for every point
     
        comp = centroids == oldCentroids #Keeps track of old centroids before changing the variable to the new centroids
        oldCentroids = centroids #Changes the old centroid variable to the new centroids variable
        
        assign_points(dataset,centroids,classification) #Assigns each point to the nearest centrooid
        
        #Appends the within cluster scatter calculated in the iteration_steps function
        iteration_steps.append(iteration_step(dataset,classification,centroids,k))
        
        #Reassigns centroids by calling the move_centroids function using the centroids assigned
        centroids = np.array(move_centroids(dataset.to_numpy(),k, classification))
    xaxis = np.arange(1,len(iteration_steps)+1)
    plt.figure()
    plt.plot(xaxis,iteration_steps)
    plt.ylabel('Within-cluster scatter W')
    plt.xlabel('Number of iterations')  
    plt.title('k=' + str(k))

            
    return centroids, classification
        


        

data = load_data("Task2 - dataset - dog_breeds.csv")



centroids, classification = kmeans(data,2)
x = np.column_stack((data.to_numpy(),classification))


#Plots the data

#height and tail length
plt.figure()
k = 2
for c in range(k):
    toPlot = x[x[:,4] == c]
    plt.scatter(toPlot[:,0],toPlot[:,1])
    plt.scatter(centroids[:,0],centroids[:,1])
    plt.ylabel("Tail Length")
    plt.xlabel("Height")

#Height and Leg Length
plt.figure()
for c in range(k):
    toPlot = x[x[:,4] == c]
    plt.scatter(toPlot[:,0],toPlot[:,2])
    plt.scatter(centroids[:,0],centroids[:,2])
    plt.ylabel("Leg Length")
    plt.xlabel("Height")

    
    



            
centroids, classification = kmeans(data,3)
x = np.column_stack((data.to_numpy(),classification))

#height and tail length
plt.figure()
k = 3
for c in range(k):
    toPlot = x[x[:,4] == c]
    plt.scatter(toPlot[:,0],toPlot[:,1])
    plt.scatter(centroids[:,0],centroids[:,1])
    plt.ylabel("Tail Length")
    plt.xlabel("Height")
    
#Height and Leg Length
plt.figure()
for c in range(k):
    toPlot = x[x[:,4] == c]
    plt.scatter(toPlot[:,0],toPlot[:,2])
    plt.scatter(centroids[:,0],centroids[:,2])
    plt.ylabel("Leg Length")
    plt.xlabel("Height")
    


    
    
    
    

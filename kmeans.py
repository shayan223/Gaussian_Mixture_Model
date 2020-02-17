import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from PIL import Image
from tqdm import tqdm 
###################################
def euclidDist(x,y):
    return (np.sum((x-y)**2))**.5
    
###################################
def kmeans(data,numOfClusters):
    
        
    K = numOfClusters
    
    rows = data.shape[0]
    columns = data.shape[1]
    
    
    centroidList = np.empty((K,data.shape[1]))
    
    #print(centerList.shape)
    
    index = np.random.choice(range(rows),K,replace=False)
    
    for i in range(K):
        centroidList[i,:] = data[index[i]]
        
    
    clusterMap = np.empty(rows)#maps points to their respective clusters/centroids
    
    prevCentList = np.empty((centroidList.shape))
    iterCount = 0
    
    while(np.array_equal(prevCentList[0], centroidList[0]) == False):
        prevCentList = np.copy(centroidList)
        
        for i in range(rows):#cluster all data points
            closest = 0#index of closest centroid
            distFromClosest = euclidDist(data[i,:],centroidList[0])#current closest distance
            
            for j in range(K):#check every centroid
                checkDist = euclidDist(data[i,:],centroidList[j])#dist from current centroid
                if(checkDist < distFromClosest):#if closer than current best, update
                    closest = j
                    distFromClosest = checkDist
                    
            clusterMap[i] = closest#update mapping for current point
            
             
        #re-calculate every centroid
        for i in range(K):#loops for every cluster
            total = np.zeros(columns)
            count = 0
            #find the average of the current cluster
            for j in range(rows):#loop through every data point
                if(clusterMap[j] == i):#if the j'th data point is in i'th cluster
                    total += data[j]
                    count += 1
            
            mean = total/count
            centroidList[i] = mean #update centroid to mean of its cluster
            
        
        iterCount += 1
        
        
        
    #find mean square error of each cluster
    mseList = []
    
    for c in range(K):#do this for each cluster c
        summation = 0
        clusterSize = 0
        for x in range(rows):#compare each point with the cluster center
            if(clusterMap[x] == c):
                summation += (euclidDist(data[x],centroidList[c])**2)
                clusterSize += 1
        mseList.append((summation/clusterSize))
        
    
    #find average mean square error across all clusters
    avgMse = (sum(mseList)/K)
    

    #find mean square seperation
    summation = 0
    for i in range(K):
        for j in range(K):
            if(np.array_equal(centroidList[i],centroidList[j]) == False):
                summation += (euclidDist(centroidList[i],centroidList[j])**2)
                
    meanSquareSep = summation/((K*(K-2))/2)
    
    returnList = [centroidList, 
                  avgMse, 
                  meanSquareSep,  
                  clusterMap] 
                  #target, 
                  #probClustList]
    return returnList    
  
###############################    
data = np.genfromtxt(fname="GMM_dataset_546.txt",delimiter="  ")
#data = pd.DataFrame(data)
#print(data)
K = 3
r = 1
kMeansList = []
bestClustering = 0#holds index of the best Kmeans clustering
for i in tqdm(range(r)):
    kMeansList.append(kmeans(data,K))
    if(kMeansList[i][1] < kMeansList[bestClustering][1]):
        bestClustering = i



finalClustering = kMeansList[bestClustering]
#print(data)
centroids = finalClustering[0]
avgMSE = finalClustering[1]
meanSqrSep = finalClustering[2]
pointToCluster = finalClustering[3]

print(pointToCluster)
print(data)
print("Average mse:", finalClustering[1])
print("Mean Square Seperation:", finalClustering[2])   




#sort and display data
data = pd.DataFrame(data)
data.columns = ['x','y']
data['cluster'] = pointToCluster

pointPlot = sb.scatterplot(data=data, x='x', y='y', hue=data['cluster'],palette="bright")

sb.scatterplot(x=centroids[:,0],y=centroids[:,1],color='black',s=150, marker='X')

pointPlot.legend_.remove()
print(data)
#######################

























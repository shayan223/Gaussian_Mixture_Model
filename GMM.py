import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from PIL import Image
from tqdm import tqdm
from scipy.stats import multivariate_normal
###################################

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

def GMM(data,numOfGaus,iterations,kmeansSeeding=None):
    #parameter initializations
    rows = data.shape[0] #number of data points
    #xVals = data[:,0]
    #yVals = data[:,1]
    columns = data.shape[1]
    dim = columns #dimensionality of our data
    
    if(kmeansSeeding is None):
        mean = np.random.uniform(-1,1,(numOfGaus,dim)) #random mean for every cluster
    else:#seed initial mean with kmeans centroid
        mean = kmeansSeeding
    
    mixingCoef = np.random.uniform(0,1,(numOfGaus,1)) #pi for every cluster
    #normalise the coeficient so it sums to one
    mixingCoef = mixingCoef/mixingCoef.sum(axis=1, keepdims=True)
    
    #create a covariance matrix for every cluster/gaussian
    covList = []
    for i in range(numOfGaus):
        #random square matrix to generate covariance using dimensionality of data
        randMatrix = np.random.uniform(0,1,(dim,dim))
        #matrix times its transpose gives us a square covariance matrix (positive semidefinite)
        covariance = np.dot(randMatrix,randMatrix.transpose())
        covList.append(covariance)
        
    

    #use initialized values to create N gaussian distributions
    gaussians = []
    for i in range(numOfGaus):
        newGauss = multivariate_normal(mean[i],covList[i])
        gaussians.append(newGauss)
        
    iterCount = 0
    prevLikelihood = -1
    curLikelihood = np.NINF
    while(prevLikelihood != curLikelihood and iterCount < iterations):
        ##########   E-step    #####################
        
        #matrix, row for every data point, column for every distribution (responsibilities)
        r = np.zeros((rows,numOfGaus))
    
        #for every cluster compute responsibilities for every point
        for distribution in range(numOfGaus):
            for point in range(rows):
                prob = mixingCoef[distribution]*(gaussians[distribution].pdf(data[point]))
                #print(prob)
                r[point][distribution] = prob
                
        #normalise r values(row wise) by deviding by the sum of each row
        r = np.divide(r,r.sum(axis=1, keepdims=True))
        
    
        ##########    M-step    #####################
        
        #change mean, covariance, and mixing coefficient based on r for each cluster
        clusterResp = np.sum(r,axis=0,keepdims=True)#sum of all of each cluster's responsibility
        flatClusterResp = clusterResp.flatten()#to solve 1 vs 0 dim conflicts
        
        #first compute new mixing coeficient 
        #(this is a vector operation, so no loop is necesary)

        newMix = clusterResp/rows
        mixingCoef = newMix
        #for some reason the above operation swaps axes, so we will swap them back
        mixingCoef = np.swapaxes(mixingCoef, 0,1)

        
        #find new mean, weighted sum of every data point in cluster
        #(weighted/multiplied by responsibility of that row-cluster pair)
        #then devide by the cluster's responsibility
    
        for distribution in range(numOfGaus):
            weightedSum = 0
            for point in range(rows):
                weightedSum += (r[point][distribution] * data[point])
            newMean = (1/flatClusterResp[distribution]) * weightedSum
            mean[distribution] = newMean
            
     
    
        #compute new covariance matrix (uses updated means)
    
    
        for distribution in range(numOfGaus):
            covSum = np.zeros((dim,dim))#matches shape of covariance matrix
            for point in range(rows):
                pointMeanDif = np.subtract(data[point], mean[distribution])
                multMatrix = np.outer(np.transpose(pointMeanDif),pointMeanDif)
                covSum += np.multiply((r[point][distribution]), multMatrix)
    
            newCov = (1/flatClusterResp[distribution]) * covSum
            covList[distribution] = newCov
    
        
        
        #update parameters for all gaussian distributions
        for i in range(numOfGaus):
            newGauss = multivariate_normal(mean[i],covList[i])
            gaussians[i] = newGauss
    
        #compute log likelihood of dataset given the new parameters
        #sum of all logs of the weighted sum of the probability of all points in each distribution
        logLikely = 0
    
        for point in range(rows):
            sumOfProb = 0
            for distribution in range(numOfGaus):
                sumOfProb += mixingCoef.flatten()[distribution] * gaussians[distribution].pdf(data[point])
            logLikely += np.log(sumOfProb)
        
        #print("Log Likelihood: ",logLikely)
        #update likelihood variables to check for convergence
        prevLikelihood = curLikelihood
        curLikelihood = logLikely
        
        #keep track how many times we've iterated
        iterCount += 1
        #print("Iteration: ", iterCount)
        #if likelihood has gone down, we have an error
        if(curLikelihood < prevLikelihood-1):
            print("ERROR: likelihood decreasing")
            return
        
        #################    End of while loop    #########################
        
    #use argmax to map each point with its most likely distribution
    distributionMapping = np.argmax(r,axis=1)
    
    #return list of information
    retList = [curLikelihood,
               gaussians,
               mean,
               covList,
               distributionMapping]
    return retList


##############################################################################
    


##############################################################################
    
data = np.genfromtxt(fname="GMM_dataset_546.txt",delimiter="  ")
#print(data)
clusters = 3
r = 1
iterations = 70
seed = kmeans(data,clusters)[0]

#take the results with the best log likelihood of r trials
topScore = np.NINF
resultIndex = 0#keeps track of which result was the best
resultList = []
for i in range(r):
    result = GMM(data, clusters,iterations,kmeansSeeding=seed)
    score = result[0]
    if(score > topScore):
        topScore = score
        resultIndex = i
    resultList.append(result)
    
bestOutcome = resultList[resultIndex]
    
print("Best likelihood: ", bestOutcome[0])
print("r: ",r)
print("Iterations: ",iterations)
print("Seeded = True")
data = pd.DataFrame(data)
data.columns = ['x','y']
data['gaussian'] = bestOutcome[4]

pointPlot = sb.scatterplot(data=data, x='x', y='y', hue=data['gaussian'],palette="bright")

#sb.scatterplot(x=data['x'],y=data[:,1],color='black',s=150, marker='X')

pointPlot.legend_.remove()














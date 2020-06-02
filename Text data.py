#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import codecs
import re
import random
import math
from scipy.special import psi
import pandas as pd 

data= pd.read_csv('text_data.txt', sep="\t",  header=0)
# wordCount
wordcountseries = data.sum(axis=1)
wordCount = wordcountseries.tolist() 
#  itemIdList, itemCountList
itemCountList = []
itemIdList = []
for i in range(0, 300):
    item = []
    itemcount = []
    for j in range(0, 100):
        if data.iat[i, j] !=0:
            item.append(j)
            itemcount.append(data.iat[i, j])
    itemIdList.append(item)
    itemCountList.append(itemcount)
# word2id,id2word
keys = list(data)
values = list(range(100))
word2id = dict(zip(keys, values))

keys = list(range(100))
values = list(data)
id2word = dict(zip(keys, values))


# In[ ]:


def maxItemNum():
    num = 0
    for d in range(0, N):
        if len(itemIdList[d]) > num:
            num = len(itemIdList[d])
    return num

def initialLdaModel():
    for z in range(0, K):
        for w in range(0, M):
            nzw[z, w] += 1.0/M + random.random()
            nz[z] += nzw[z, w]
    updateVarphi()    


def updateVarphi():
    for z in range(0, K):
        for w in range(0, M):
            if(nzw[z, w] > 0):
                varphi[z, w] = math.log(nzw[z, w]) - math.log(nz[z])
            else:
                varphi[z, w] = -100
                
def variationalInference(wordCount,itemIdList,itemCountList, d, gamma, phi):
    phisum = 0
    oldphi = np.zeros([K])
    digamma_gamma = np.zeros([K])
    
    for z in range(0, K):
        gamma[d][z] = alpha + wordCount[d] * 1.0 / K
        digamma_gamma[z] = psi(gamma[d][z])
        for w in range(0, len(itemIdList[d])):
            phi[w, z] = 1.0 / K

    for iteration in range(0, iterInference):
        for w in range(0, len(itemIdList[d])):
            phisum = 0
            for z in range(0, K):
                oldphi[z] = phi[w, z]
                phi[w, z] = digamma_gamma[z] + varphi[z, itemIdList[d][w]]
                if z > 0:
                    phisum = math.log(math.exp(phisum) + math.exp(phi[w, z]))
                else:
                    phisum = phi[w, z]
            for z in range(0, K):
                phi[w, z] = math.exp(phi[w, z] - phisum)
                gamma[d][z] =  gamma[d][z] + itemCountList[d][w]  * (phi[w, z] - oldphi[z])
                digamma_gamma[z] = psi(gamma[d][z])


# In[ ]:


N = 300
M = 100
K = 3
iterInference = 20 
iterEM = 20
alpha = 5
alphaSS = 0
varphi = np.zeros([K, M])
nzw = np.zeros([K, M])
nz = np.zeros([K])
gamma = np.zeros([N, K])
phi = np.zeros([maxItemNum(), K])


# In[ ]:


initialLdaModel()

for iteration in range(0, iterEM): 
    nz = np.zeros([K])
    nzw = np.zeros([K, M])
    alphaSS = 0
    # EStep
    for d in range(0, N):
        variationalInference(wordCount,itemIdList,itemCountList, d, gamma, phi)
        gammaSum = 0
        for z in range(0, K):
            gammaSum += gamma[d, z]
            alphaSS += psi(gamma[d, z])
        alphaSS -= K * psi(gammaSum)

        for w in range(0, len(itemIdList[d])):
            for z in range(0, K):
                nzw[z][itemIdList[d][w]] +=itemCountList[d][w]  * phi[w, z]
                nz[z] += itemCountList[d][w]  * phi[w, z]

    # MStep
    updateVarphi()


# In[ ]:


topic_doc = []
for d in range(0, N):
    theta = sum(gamma[d])
    theta_1 = gamma[d][0]/theta
    theta_2 = gamma[d][1]/theta
    theta_3 = gamma[d][2]/theta
    theta_update = [theta_1,theta_2,theta_3]
    topic_doc.append(theta_update)
df = pd.DataFrame(topic_doc) 
df.T      


# In[ ]:


true_distribution = pd.read_csv('topic_doc_true.txt', sep="\t",  header=None)
true_distri_sort = true_distribution
true_distri_sort.reindex([2,0,1])


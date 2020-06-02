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

cora = pd.read_csv('cora_data.txt', sep="\t",  header='infer')

# number of document
N = 2407
# number of distinct term
M = 2961
# number of topic
K = 8

# wordCount
wordcountseries = cora.sum(axis=1)
wordCount = wordcountseries.tolist() 

#  itemIdList, itemCountList
itemCountList = []
itemIdList = []
for i in range(0, N):
    item = []
    itemcount = []
    for j in range(0, M):
        if cora.iat[i, j] !=0:
            item.append(j)
            itemcount.append(cora.iat[i, j])
    itemIdList.append(item)
    itemCountList.append(itemcount)

# word2id,id2word
keys = list(cora)
values = list(range(M))
word2id = dict(zip(keys, values))

keys = list(range(M))
values = list(cora)
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

# update model parameters : varphi (the update of alpha is ommited)
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
    theta_update = []
    for z in range(0, K): 
        theta_1 = gamma[d][z]/sum(gamma[d])
        theta_update.append(theta_1)
    topic_doc.append(theta_update)
    
df = pd.DataFrame(topic_doc) 
df.T 


# In[ ]:


# calculate the top 10 terms of each topic
topicwords = []
maxTopicWordsNum = 10
for z in range(0, K):
	ids = varphi[z, :].argsort()
	topicword = []
	for j in ids:
		topicword.insert(0, id2word[j])
	topicwords.append(topicword[0 : min(10, len(topicword))])

df = pd.DataFrame(topicwords) 
df 


# ### Try different settings and compare the results

# In[ ]:


cora = pd.read_csv('cora_data.txt', sep="\t",  header='infer')

N = 2407
# number of distinct terms 
M = 2961
# number of topic
K = 5
wordcountseries = cora.sum(axis=1)
wordCount = wordcountseries.tolist() 
#  itemIdList, itemCountList
itemCountList = []
itemIdList = []
for i in range(0, N):
    item = []
    itemcount = []
    for j in range(0, M):
        if cora.iat[i, j] !=0:
            item.append(j)
            itemcount.append(cora.iat[i, j])
    itemIdList.append(item)
    itemCountList.append(itemcount)


# In[ ]:


# word2id,id2word
keys = list(cora)
values = list(range(M))
word2id = dict(zip(keys, values))

keys = list(range(M))
values = list(cora)
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

# update model parameters : varphi (the update of alpha is ommited)
def updateVarphi():
    for z in range(0, K):
        for w in range(0, M):
            if(nzw[z, w] > 0):
                varphi[z, w] = math.log(nzw[z, w]) - math.log(nz[z])
            else:
                varphi[z, w] = -100


# In[ ]:


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


iterInference = 20 
iterEM = 20
alpha = 0.5
alphaSS = 0
varphi = np.zeros([K, M])
nzw = np.zeros([K, M])
nz = np.zeros([K])
gamma = np.zeros([N, K])
phi = np.zeros([maxItemNum(), K])

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
    theta_update = []
    for z in range(0, K): 
        theta_1 = gamma[d][z]/sum(gamma[d])
        theta_update.append(theta_1)
    topic_doc.append(theta_update)
    
df = pd.DataFrame(topic_doc) 
df.T 


# In[ ]:


# calculate the top 10 terms of each topic
topicwords = []
maxTopicWordsNum = 10
for z in range(0, K):
	ids = varphi[z, :].argsort()
	topicword = []
	for j in ids:
		topicword.insert(0, id2word[j])
	topicwords.append(topicword[0 : min(10, len(topicword))])

topicwords
tw_df = pd.DataFrame(topicwords) 
tw_df 


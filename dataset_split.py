#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import os
import torch
from torch_geometric.data import Data, DataLoader, Dataset

# In[3]:

'''
Each sample is verified by VirusTotal, which contains sample label and its appearance time.
Each malware type and the benign datasets are split into 80% and 20% samples in temporal order 
for training and testing, respectively. 
data_MD52label is a dict of {'file md5' : 'label'}
dataset_split_md5_list contains 4 md5 list of samples for training and test
'''

data_MD52label = np.load('./data/data_MD52label.npz', allow_pickle=True)
data_MD52label = data_MD52label['data_MD52label'][()]

dataset_split_md5_list = np.load('./dataset_split_md5_list.npz', allow_pickle=True)
detection_train_md5 = dataset_split_md5_list['detection_train_md5']
detection_test_md5 = dataset_split_md5_list['detection_test_md5']
classification_train_md5 = dataset_split_md5_list['classification_train_md5']
classification_test_md5 = dataset_split_md5_list['classification_train_md5']


# In[7]:


def loadnpz(md5):
    graph = np.load('./data/graph/' + md5 + '_graph.npz', allow_pickle=True)
    node_feature = graph['node_feature']
    edge_set = graph['edge_set']
    edge_attr = graph['edge_attr']
    d = Data()
    d.x = torch.tensor(node_feature)
    d.edge_attr = torch.tensor(edge_attr)
    d.edge_index = torch.tensor(edge_set).T
    return d


# In[11]:


datalist_detection_train = []
datalist_detection_test = []
datalist_classification_train = []
datalist_classification_test = []
num = 1
for md5 in list(data_MD52label.keys()):
    print('{} / {}'.format(num, len(data_MD52label)))
    num += 1
    if data_MD52label[md5] == 'Benign':
        data = loadnpz(md5)
        data.y = torch.tensor([0])
        if md5 in detection_train_md5:
            datalist_detection_train.append(data)
        else:
            datalist_detection_test.append(data)
    else:
        data = loadnpz(md5)
        data.y = torch.tensor([1])
        if md5 in detection_train_md5:
            datalist_detection_train.append(data)
        else:
            datalist_detection_test.append(data)
    if data_MD52label[md5] == 'Adware':
        data = loadnpz(md5)
        data.y = torch.tensor([0])
        if md5 in classification_train_md5:
            datalist_classification_train.append(data)
        else:
            datalist_classification_test.append(data)
        continue
    if data_MD52label[md5] == 'Backdoor':
        data = loadnpz(md5)
        data.y = torch.tensor([1])
        if md5 in classification_train_md5:
            datalist_classification_train.append(data)
        else:
            datalist_classification_test.append(data)
        continue
    if data_MD52label[md5] == 'Downloader':
        data = loadnpz(md5)
        data.y = torch.tensor([2])
        if md5 in classification_train_md5:
            datalist_classification_train.append(data)
        else:
            datalist_classification_test.append(data)
        continue
    if data_MD52label[md5] == 'Ransom':
        data = loadnpz(md5)
        data.y = torch.tensor([3])
        if md5 in classification_train_md5:
            datalist_classification_train.append(data)
        else:
            datalist_classification_test.append(data)
        continue
    if data_MD52label[md5] == 'Spyware':
        data = loadnpz(md5)
        data.y = torch.tensor([4])
        if md5 in classification_train_md5:
            datalist_classification_train.append(data)
        else:
            datalist_classification_test.append(data)
        continue
    if data_MD52label[md5] == 'Trojan':
        data = loadnpz(md5)
        data.y = torch.tensor([5])
        if md5 in classification_train_md5:
            datalist_classification_train.append(data)
        else:
            datalist_classification_test.append(data)
        continue
    if data_MD52label[md5] == 'Virus':
        data = loadnpz(md5)
        data.y = torch.tensor([6])
        if md5 in classification_train_md5:
            datalist_classification_train.append(data)
        else:
            datalist_classification_test.append(data)
        continue
    if data_MD52label[md5] == 'Worm':
        data = loadnpz(md5)
        data.y = torch.tensor([7])
        if md5 in classification_train_md5:
            datalist_classification_train.append(data)
        else:
            datalist_classification_test.append(data)
        continue

# In[13]:


torch.save(datalist_detection_train, './data/datalist_detection_train.pt')
torch.save(datalist_detection_test, './data/datalist_detection_train.pt')
torch.save(datalist_classification_train, './data/datalist_classification_train.pt')
torch.save(datalist_classification_test, './data/datalist_classification_train.pt')

# In[ ]:

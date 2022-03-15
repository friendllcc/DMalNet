#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import os


# In[3]:


def ifNewNode(nodevec, nodevec_list):
    for n in nodevec_list:
        if all(nodevec == n):
            return False
    return True


# In[4]:


def getNodeId2NodeVec(data):
    nodeid2nodevec = {0: -np.ones((32,), dtype=np.float32)}  # Node 0 is the end node of the graph
    for i in range(len(data)):
        nodevec = data[i, :32]
        if ifNewNode(nodevec, nodeid2nodevec.values()):
            nodeid2nodevec[len(nodeid2nodevec.keys())] = nodevec
    return nodeid2nodevec


# In[5]:


def getEdgeSet(data, nodeid2nodevec):
    node_np = data[:, :32]
    item_list = []
    for i in range(len(node_np)):
        key = [k for k, v in nodeid2nodevec.items() if all(v == node_np[i])]
        item_list.append(key[0])
    edge_list = []
    for i in range(len(item_list)):
        if i == len(item_list) - 1:
            edge_list.append([item_list[i], 0])
        else:
            edge_list.append([item_list[i], item_list[i + 1]])
    return np.asarray(edge_list)


# In[6]:


def getGraph(data):
    nodeid2nodevec = getNodeId2NodeVec(data)
    node_feature = np.asarray(list(nodeid2nodevec.values()))  # node feature
    edge_set = getEdgeSet(data, nodeid2nodevec)  # edge_set
    edge_attr = data[:, 32:]  # edge feature
    return node_feature, edge_set, edge_attr


# In[21]:

file_md5_list = os.listdir('./data/sequence/')
num = 1
for md5 in file_md5_list:
    print('{} / {}'.format(num, len(file_md5_list)))
    num += 1
    data = np.load('./data/sequence/' + md5 + '.npz', allow_pickle=True)
    data = data['data']
    node_feature, edge_set, edge_attr = getGraph(data)
    np.savez('./data/graph/' + md5 + '_graph.npz', node_feature=node_feature, edge_set=edge_set, edge_attr=edge_attr)

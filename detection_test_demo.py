#!/usr/bin/env python
# coding: utf-8

# In[77]:


import sys
import os.path as osp
from random import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GraphConv, TopKPooling,GCNConv, GINEConv,TAGConv,MFConv,GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score


# In[78]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('device', device)


# In[79]:


detection_data_test = torch.load('./data/demo_detection_test.pt')


# In[80]:


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.dec1 = torch.nn.Linear(100, 64)
        self.dec2 = torch.nn.Linear(64, 32)
        
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 256)
        )
        self.gineconv1 = GINEConv(self.mlp)
        self.pool1 = TopKPooling(256, ratio=0.8)
        self.conv2 = GATConv(256, 64, heads=4)
        self.pool2 = TopKPooling(256, ratio=0.8)

        self.lin1 = torch.nn.Linear(512, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 2)
        
    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr
        
        edge_attr = F.relu(self.dec1(edge_attr))
        edge_attr = F.relu(self.dec2(edge_attr))

        x = F.relu(self.gineconv1(x=x, edge_index=edge_index, edge_attr=edge_attr))
        x, edge_index, edge_attr1, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr2, batch, _, _ = self.pool2(x, edge_index, edge_attr1, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.2, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


# In[81]:


model = torch.load('./model/detection_model.pkl')


# In[82]:


data_test_loader = DataLoader(detection_data_test, batch_size=128)


# In[83]:


def test(loader):
    model.eval()
    label_list = []
    pred_list = []
    correct = 0
    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
        
        label_list_batch = data.y.to('cpu').detach().numpy().tolist()
        pred_list_batch = pred.to('cpu').detach().numpy().tolist()
        for label_item in label_list_batch:
            label_list.append(label_item)
        for pred_item in pred_list_batch:
            pred_list.append(pred_item)
    
    y_true = np.asarray(label_list)
    y_pred = np.asarray(pred_list)
    _val_confusion_matrix = confusion_matrix(y_true, y_pred)
    _val_acc = accuracy_score(y_true, y_pred)
    _val_precision = precision_score(y_true, y_pred)
    _val_recall = recall_score(y_true, y_pred)
    _val_f1 = f1_score(y_true, y_pred)
    return _val_confusion_matrix, _val_acc, _val_precision, _val_recall, _val_f1


# In[85]:


con, acc, precision, recall, f1 = test(data_test_loader)
print('Test Acc: {:.5f}, Test Precision: {:.5f}, Test Recall: {:.5f}, Test F1: {:.5f}'.
          format(acc, precision, recall, f1))
print('FPR:', con[0,1]/(con[0,1]+con[0,0]))
print('FNR:', con[1,0]/(con[1,0]+con[1,1]))

'''
Test Acc: 0.98550, Test Precision: 0.97833, Test Recall: 0.99300, Test F1: 0.98561
FPR: 0.022
FNR: 0.007
'''

# In[ ]:





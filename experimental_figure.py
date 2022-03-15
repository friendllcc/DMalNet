#!/usr/bin/env python
# coding: utf-8

# In[44]:


import sys
import os.path as osp
from random import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, GCNConv, GINEConv, TAGConv, MFConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# In[30]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print('device', device)


# detection result: get the fpr, tpr, roc_auc for roc curve
# In[31]:


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


# In[32]:


model = torch.load('./model/detection_model.pkl')

# In[33]:


print(model)


# In[34]:


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


def test_for_roc(loader):
    model.eval()
    label_list = []
    prob_list = []
    for data in loader:
        data = data.to(device)
        pred = model(data)

        label_list_batch = data.y.to('cpu').detach().numpy().tolist()
        pred_list_batch = pred.to('cpu').detach().numpy().tolist()
        for pred_item in pred_list_batch:
            prob_list.append(2 ** pred_item[1])  # 还原log_softmax产生的概率
        for label_item in label_list_batch:
            label_list.append(label_item)

    y_true = np.asarray(label_list)
    y_prob = np.asarray(prob_list)
    fpr, tpr, threshold = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)  ### calculate AUC
    return fpr, tpr, roc_auc


# In[35]:


data_test = torch.load('./data/datalist_detection_test.pt')
data_test_loader = DataLoader(data_test, batch_size=128)

# In[36]:


con, acc, precision, recall, f1 = test(data_test_loader)
print('Test Acc: {:.5f}, Test Precision: {:.5f}, Test Recall: {:.5f}, Test F1: {:.5f}'.
      format(acc, precision, recall, f1))
print(con)
print('FPR:', con[0, 1] / (con[0, 1] + con[0, 0]))
print('FNR:', con[1, 0] / (con[1, 0] + con[1, 1]))

# In[15]:


fpr, tpr, roc_auc = test_for_roc(data_test_loader)

# In[66]:


np.savez('./detection_roc', fpr=fpr, tpr=tpr, roc_auc=roc_auc)

# In[34]:


roc = np.load('./detection_roc.npz', allow_pickle=True)
roc_auc = roc['roc_auc']
print(roc_auc)

# In[33]:


plt.figure()
lw = 2
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color='#1D80C3', lw=lw, label='Proposed Model (AUC = %0.4f)' % roc_auc)
# plt.plot([0, 0.1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 0.5])
plt.ylim([0.0, 1.05])
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend(loc="lower right", fontsize=11.5)
# plt.savefig('ROC_ablation.eps', bbox_inches='tight')
plt.show()


# classification result: get the confusion matrix

# In[37]:


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.dec1 = torch.nn.Linear(100, 64)
        self.dec2 = torch.nn.Linear(64, 32)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 512)
        )
        self.gineconv1 = GINEConv(self.mlp)
        self.pool1 = TopKPooling(512, ratio=0.8)
        self.conv2 = GATConv(512, 128, heads=4)
        self.pool2 = TopKPooling(512, ratio=0.8)

        self.lin1 = torch.nn.Linear(1024, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 8)

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

        # print(x.shape)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


# In[38]:


model = torch.load('./model/classification_model.pkl')


# In[39]:


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
    _val_precision = precision_score(y_true, y_pred, average='macro')
    _val_recall = recall_score(y_true, y_pred, average='macro')
    _val_f1 = f1_score(y_true, y_pred, average='macro')
    return _val_confusion_matrix, _val_acc, _val_precision, _val_recall, _val_f1


# In[40]:


data_test = torch.load('./data/data_washing_classification_test_rate_021_0618.pt')
data_test_loader = DataLoader(data_test, batch_size=128)

# In[41]:


con, acc, precision, recall, f1 = test(data_test_loader)

# In[42]:


print(con, acc, recall, precision, f1)

# In[63]:


index2label = {0: 'Adware', 1: 'Backdoor', 2: 'Downloader', 3: 'Ransom', 4: 'Spyware', 5: 'Trojan', 6: 'Virus',
               7: 'Worm'}
pre_label = []
true_label = []
values = []
for i in range(8):
    for j in range(8):
        pre_label.append(index2label[j])
        true_label.append(index2label[i])
        values.append((con[i, j] / sum(con[i])) * 100)

df = pd.DataFrame({'Predicted label': pre_label,
                   'True label': true_label,
                   'values': values})
df.head()

# In[64]:


pt = df.pivot_table(index='True label', columns='Predicted label', values='values')
pt.head()

# In[94]:


f, ax = plt.subplots(figsize=(8, 7))
sns.color_palette("Blues")
cmap = sns.cubehelix_palette(n_colors=8, start=1, rot=-0.9, gamma=1.5, hue=1, light=0.95, dark=0.3, as_cmap=True)
sns.heatmap(pt, vmin=0, vmax=100, cmap=cmap, linewidths=0.05, ax=ax, annot=True, annot_kws={'size': 14})
ax.set_title('Classification confusion matrix(%)', fontsize=14)
ax.set_xlabel('Predicted label', fontsize=14)
ax.set_ylabel('True label', fontsize=14)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, fontsize=14)
ax.set_yticklabels(ax.get_yticklabels(), rotation=40, fontsize=14)
f.savefig('sns_heatmap_normal.eps', bbox_inches='tight')

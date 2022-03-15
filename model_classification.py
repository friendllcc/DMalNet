#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path as osp
from random import random
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import GraphConv, TopKPooling, GCNConv, GINEConv, TAGConv, MFConv, GATConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm

# In[2]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device)

# In[3]:


data_train = torch.load('./data/datalist_classification_train.pt')
data_train_loader = DataLoader(data_train, batch_size=128)
data_test = torch.load('./data/datalist_classification_test.pt')
data_test_loader = DataLoader(data_test, batch_size=128)

# Calculate the weight to reduce the influence of data imbalance

# In[5]:


Adware = 0
Backdoor = 0
Downloader = 0
Ransom = 0
Spyware = 0
Trojan = 0
Virus = 0
Worm = 0

for data in data_train:
    label = data.y.item()
    if label == 0:
        Adware += 1
    elif label == 1:
        Backdoor += 1
    elif label == 2:
        Downloader += 1
    elif label == 3:
        Ransom += 1
    elif label == 4:
        Spyware += 1
    elif label == 5:
        Trojan += 1
    elif label == 6:
        Virus += 1
    elif label == 7:
        Worm += 1

# In[6]:


Adware_weight = 1
Backdoor_weight = Adware / Backdoor
Downloader_weight = Adware / Downloader
Ransom_weight = Adware / Ransom
Spyware_weight = Adware / Spyware
Trojan_weight = Adware / Trojan
Virus_weight = Adware / Virus
Worm_weight = Adware / Worm

# In[7]:


class_weight = torch.FloatTensor([Adware_weight,
                                  Backdoor_weight,
                                  Downloader_weight,
                                  Ransom_weight,
                                  Spyware_weight,
                                  Trojan_weight,
                                  Virus_weight,
                                  Worm_weight]).cuda()


# In[43]:

# graph learning model architecture
# proposed model
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


# In[44]:


model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_scores = {'acc': [], 'precision': [], 'recall': [], 'f1': []}
test_scores = {'acc': [], 'precision': [], 'recall': [], 'f1': []}


# In[45]:

# train & test
def train(epoch, loader):
    model.train()

    loss_all = 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y, weight=class_weight)
        loss.backward()
        loss_all += data.num_graphs * loss.item()
        optimizer.step()
    return loss_all / len(data_train)


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
    # _val_confusion_matrix = confusion_matrix(y_true, y_pred)
    _val_acc = accuracy_score(y_true, y_pred)
    _val_precision = precision_score(y_true, y_pred, average='macro')
    _val_recall = recall_score(y_true, y_pred, average='macro')
    _val_f1 = f1_score(y_true, y_pred, average='macro')
    return _val_acc, _val_precision, _val_recall, _val_f1


# In[46]:


print(model)

# In[47]:

for epoch in tqdm(range(1, 401)):
    # train
    loss = train(epoch, data_train_loader)
    acc, precision, recall, f1 = test(data_train_loader)
    train_scores['acc'].append(acc)
    train_scores['precision'].append(precision)
    train_scores['recall'].append(recall)
    train_scores['f1'].append(f1)
    print(
        'Epoch: {:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Train Precision: {:.5f}, Train Recall: {:.5f}, Train F1: {:.5f}'.
            format(epoch, loss, acc, precision, recall, f1))
    # test
    acc, precision, recall, f1 = test(data_test_loader)
    test_scores['acc'].append(acc)
    test_scores['precision'].append(precision)
    test_scores['recall'].append(recall)
    test_scores['f1'].append(f1)
    print('Epoch: {:03d}, Loss: {:.5f}, Test Acc: {:.5f}, Test Precision: {:.5f}, Test Recall: {:.5f}, Test F1: {:.5f}'.
          format(epoch, loss, acc, precision, recall, f1))

    # torch.save(model, './model/classification_model.pkl')


# ablation study

# In[ ]:

# model without GAT + gPool
class Net_Without_GATgPool(torch.nn.Module):
    def __init__(self):
        super(Net_Without_GATgPool, self).__init__()

        self.dec1 = torch.nn.Linear(100, 64)
        self.dec2 = torch.nn.Linear(64, 32)

        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(32, 128),
            torch.nn.ReLU(True),
            torch.nn.Linear(128, 512)
        )
        self.gineconv1 = GINEConv(self.mlp)
        self.pool1 = TopKPooling(512, ratio=0.8)

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

        x = x1

        # print(x.shape)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x


# In[ ]:

# model without GINE + gPool
class Net_Without_GINEgPool(torch.nn.Module):
    def __init__(self):
        super(Net_Without_GINEgPool, self).__init__()

        self.conv2 = GATConv(32, 128, heads=4)
        self.pool2 = TopKPooling(512, ratio=0.8)

        self.lin1 = torch.nn.Linear(1024, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 8)

    def forward(self, data):
        x, edge_index, batch, edge_attr = data.x, data.edge_index, data.batch, data.edge_attr

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, edge_attr2, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x2

        # print(x.shape)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.4, training=self.training)
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x

import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
import time

import datasets
from datasets import ChangeDetectionDataset
from models import ChangeDetectionCNN
from sklearn.model_selection import train_test_split


# ======================================================
# 划分训练集和测试集
img_pairs,labels=datasets.list_img_paths()
train_img_pairs, test_img_pairs, train_labels, test_labels = train_test_split(img_pairs, labels, test_size=0.2)
# 创建数据集
train_dataset = ChangeDetectionDataset(train_img_pairs, train_labels)
test_dataset = ChangeDetectionDataset(test_img_pairs, test_labels)
# 创建数据加载器
batch_size = 2
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#===================================================
# 超参数设置
epochs = 100
learning_rate = 0.001
# 创建模型、损失函数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChangeDetectionCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
#===================================================
# 训练循环
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    epoch_start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        print(outputs.view(-1,outputs.shape[2], outputs.shape[3]).shape)
        loss = criterion(outputs.view(-1, outputs.shape[2], outputs.shape[3]), targets.view(-1, targets.shape[1], targets.shape[2]).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    epoch_loss = running_loss / len(train_loader)
    epoch_time = time.time() - epoch_start_time
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}, Time: {epoch_time:.4f}s")

print("Training finished.")
# 测试模型
model.eval()
correct = 0
total = 0
best_accuarcy=0
with torch.no_grad():
    
    for img_pairs, labels in test_loader:
        img_pairs,targets=img_pairs.to(device),targets.to(device)
        outputs = model(img_pairs)
      
        preds = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
        total += targets.numel()
        correct += (preds == targets).sum().item()
    accuarcy=100 * correct / total
    print(f'测试集准确率: {100 * correct / total}%')
    
# 保存模型
torch.save(model.state_dict(), 'change_detection_cnn.pth')


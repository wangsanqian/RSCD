from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from models import ChangeDetectionCNN
from torchvision import transforms
#==========================================================
# 数据预处理
def preprocess_image(img):
    img = img.astype(np.float32) / 255.0
    return img
# transform=transforms.Compose(
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# )


#===========================================================
# 自定义数据集类
class ChangeDetectionDataset(Dataset):
    def __init__(self, img_pairs, label_paths, transform=None):
        # image_pairs是列表
        self.img_pairs = img_pairs
        self.label_paths = label_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_pairs)

    def __getitem__(self, idx):
        img_pair_paths = self.img_pairs[idx]
        img1 = preprocess_image(cv2.imread(img_pair_paths[0]))
        img2 = preprocess_image(cv2.imread(img_pair_paths[1]))
        img_pair = np.concatenate((img1, img2), axis=-1)
        img_pair=img_pair.transpose(2,0,1)
        
        label_img_path = self.label_paths[idx]
        label = cv2.imread(label_img_path, cv2.IMREAD_GRAYSCALE)
        label = (label > 128).astype(np.uint8)  # 二值化

        if self.transform:
            img_pair = self.transform(img_pair)

        return img_pair, label

# ======================================================
# 加载数据LEVIR-CD数据
def list_img_paths():

    img_pairs = []  # 存储遥感影像对的路径
    labels = []  # 存储遥感影像对的标签
    # 填充img_pairs和labels
    folder_A = "./test_data/A"
    folder_B = "./test_data/B"
    labels_folder = "./test_data/label"

    img_pairs = []
    labels = []

    # 从两个文件夹中获取遥感影像对路径
    for img_file_A in os.listdir(folder_A):
        img_path_A = os.path.join(folder_A, img_file_A)
        img_path_B = os.path.join(folder_B, img_file_A)
        img_pairs.append((img_path_A, img_path_B))

        # 读取标签影像
        label_img_path = os.path.join(labels_folder, img_file_A)
    
        labels.append(label_img_path)

    assert len(img_pairs) == len(labels), "Image pairs and labels should have the same length."
    return img_pairs,labels


if __name__=='__main__':
    img_pairs,labels=list_img_paths()
    train_img_pairs, test_img_pairs, train_labels, test_labels = train_test_split(img_pairs, labels, test_size=0.2)
   # 创建数据集
    train_dataset = ChangeDetectionDataset(train_img_pairs, train_labels)
    test_dataset = ChangeDetectionDataset(test_img_pairs, test_labels)
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
    train=iter(train_loader)
    tr=iter(train_loader)
    inputs=next(tr)
    print(inputs[0].shape)
    model=ChangeDetectionCNN()
    outputs = model(inputs[0])
    print(outputs.view(-1, outputs.shape[2], outputs.shape[3]).shape)
    print(inputs[1].view(-1, outputs.shape[2], outputs.shape[3]).shape)

    
   
    
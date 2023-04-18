import  matplotlib.pyplot as plt
import cv2
import os
import numpy as np

folder_A = "./test_data/A"
folder_B = "./test_data/B"
labels_folder = "./test_data/label"

img_pairs = []
labels = []
for img_file_A in os.listdir(folder_A):
    img_path_A = os.path.join(folder_A, img_file_A)
    img_path_B = os.path.join(folder_B, img_file_A)
    label_img_path = os.path.join(labels_folder, img_file_A)
    img_pairs.append((img_path_A, img_path_B))
    labels.append(label_img_path)


def vis(idx=0):
    img_A=cv2.imread(img_pairs[idx][0])
    img_A=cv2.cvtColor(img_A,cv2.COLOR_BGR2RGB)

    img_B=cv2.imread(img_pairs[idx][1])
    img_B=cv2.cvtColor(img_B,cv2.COLOR_BGR2RGB)

    # label
    label=cv2.imread(labels[idx],cv2.IMREAD_GRAYSCALE)
    #label=cv2.cvtColor(label,cv2.COLOR_BGR2RGB)
    label = (label > 128).astype(np.uint8)  # 二值化

    return img_A,img_B,label

if __name__=='__main__':
    img_A,img_B,label=vis()
    fig,ax=plt.subplots(2,3,figsize=(8,5))
   
    ax[0][0].imshow(img_A)
    ax[0][1].imshow(img_B)
    ax[0][2].imshow(label)

    img_A2,img_B2,label2=vis(1)

    ax[1][0].imshow(img_A2)
    ax[1][1].imshow(img_B2)
    ax[1][2].imshow(label2)

    
    plt.show()


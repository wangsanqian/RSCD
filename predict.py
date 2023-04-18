import torch
import numpy as np
from models import ChangeDetectionCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def predict(model, img_pair):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(img_pair, dtype=torch.float).unsqueeze(0).to(device)
        outputs = model(inputs)
        preds = torch.where(outputs > 0.5, torch.ones_like(outputs), torch.zeros_like(outputs))
        preds = preds.squeeze().cpu().numpy()
    return preds
if __name__=='__main__':
    import cv2

    img1_path = './test_data/A/train_1.png'
    img2_path = './test_data/B/train_1.png'

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    # img1 = cv2.resize(img1, (1024, 1024))
    # img2 = cv2.resize(img2, (1024, 1024))

    img_pair = np.concatenate([img1, img2], axis=2).transpose(2, 0, 1)
    # 加载模型
    model=ChangeDetectionCNN()
    model=model.to(device)
    model.load_state_dict(torch.load('change_detection_cnn.pth'))
    
    change_detection_result = predict(model, img_pair)

    # 可视化变化检测结果
    import matplotlib.pyplot as plt

    plt.imshow(change_detection_result, cmap='gray')
    plt.show()

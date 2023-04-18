import torch
import torch.nn as nn
import torch.nn.functional as F



# 定义CNN模型
class ChangeDetectionCNN(nn.Module):
    def __init__(self):
        super(ChangeDetectionCNN, self).__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.deconv1(x))
        x = self.deconv2(x)
        x=torch.sigmoid(x)
        return x



if __name__=="__main__":
    from torchsummary import summary
    model = ChangeDetectionCNN()
    summary(model,input_size=[(6,1024,1024)],batch_size=2)
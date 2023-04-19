import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch.optim as optim


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 1, 10)
        self.conv2 = nn.Conv2d(1, 1, 10, stride=5)
        self.fc1 = nn.Linear(289, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
        


net = Net()

Data = np.genfromtxt('Table.txt')

Angles = torch.tensor(Data[:,1], dtype=torch.float32).reshape(30,-1)
Images = np.zeros((30,1,100,100))
for i in range(30):
    
    img = Image.open("Train/"+str(i+1)+".jpg")
    Mat = np.asarray(img)[:,:,0].reshape(1,100,100)
    Images[i] = Mat

    

TrainImages = torch.tensor(Images, dtype=torch.float32)


res =  net(TrainImages)

print(res)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)

for epoch in range(500):  # loop over the dataset multiple times

    outputs = net(TrainImages)
    optimizer.zero_grad()

        
    
    loss = criterion(outputs, Angles)
    loss.backward()
    optimizer.step()

        
    
        
    print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}')
        
print('Finished Training')


res =  net(TrainImages)

print(res)


torch.save(net, 'Autonet')


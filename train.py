import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import torchvision.transforms as T
import numpy as np
import math
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels= 1, 
                               kernel_size=5, stride = 5)
        self.pool1 = nn.MaxPool2d(kernel_size=5)
        
        
        
        self.fc1 = nn.Linear(16, 100)
        self.fc2 = nn.Linear(100, 25)
        self.fc3 = nn.Linear(25, 1)
        
    def forward(self, x):
        x = x.resize(1,1,100,100)
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
      
        x = torch.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        
net = Net()

to_tensor = T.ToTensor()

X=list()

for i in range(30):
    img = Image.open("Train/"+str(i+1)+".jpg")
    img = ImageOps.grayscale(img)
    img = to_tensor(img)
    X.append(img)
    
    
    
X = torch.stack(X)

Y = np.genfromtxt('Table.txt')[:,1]
Y = torch.from_numpy(Y).float()



img = Image.open("Train/1.jpg").transpose(Image.FLIP_TOP_BOTTOM)


criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
optimizer2 = torch.optim.Adam(net.parameters(), lr=0.0001)

for epoch in range(1000):  
    
    for i in range(30):
        output = net(X[i])
       
        optimizer.zero_grad()

        loss = criterion(output,Y[i])
        loss.backward()
        if epoch < 800:
            optimizer.step()
            
        else:
            optimizer2.step()
            
        print(f'[{epoch + 1}, {i + 1:5d}] loss: {loss}')
    '''  
    x1 = [50,50]
    y1 = [0, 100]
    plt.imshow(img)
    res = net(X[0]).item()
    res = math.radians(res)
    x = np.array([50, 50+40*np.cos(res)])
    y = np.array([0, 40*np.sin(res)])
    plt.scatter(50+40*np.cos(res), 40*np.sin(res), color = "red")
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.plot(x1,y1, color='green', linestyle='dashdot')
    plt.plot(x,y, color = "red", label = f'epoch: {epoch + 1}')
    plt.legend()
    plt.show()
    '''
    
    
print('Finished Training')

torch.save(net, 'model.pth')


for i in range(30):
    img = Image.open("Train/"+str(i+1)+".jpg").transpose(Image.FLIP_TOP_BOTTOM)
    img = ImageOps.grayscale(img)
    x1 = [50,50]
    y1 = [0, 100]
    plt.imshow(img)
    res = net(X[i]).item()
    res = math.radians(res)
    x = np.array([50, 50+40*np.cos(res)])
    y = np.array([0, 40*np.sin(res)])
    plt.scatter(50+40*np.cos(res), 40*np.sin(res), color = "red")
    plt.xlim([0,100])
    plt.ylim([0,100])
    plt.plot(x1,y1, color='orange', linestyle='dashdot')
    plt.plot(x,y, color = "red")
    plt.legend()
    plt.show()




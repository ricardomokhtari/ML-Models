# attempt of writing a 3 layer CNN from scratch in PyTorch

# imports
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets

compose = transforms.Compose([
            transforms.ToTensor(),
            # if using RGB: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            transforms.Normalize([0.5],[0.5])
        ])
# define global parameters
batch = 4
n_epochs = 2

# if using own dataset uncomment the next lines
# trainset = torchvision.datasets.ImageFolder(root='folder/', transform=transform)
# testset = torchvision.datasets.ImageFolder(root='folder/', transform=transform)

# create iterable training and test sets
trainset = datasets.MNIST(root="./data",train=True,transform=compose,download=True)
trainloader = torch.utils.data.DataLoader(dataset=trainset,batch_size=batch,shuffle=True,
                                          num_workers=2)

testset = datasets.MNIST(root="./data",train=False,transform=compose,download=True)
testloader = torch.utils.data.DataLoader(dataset=testset,batch_size=batch,shuffle=False,
                                          num_workers=2)

# define the classes in the data
classes = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine')

# define the network
class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        # first convolutional layer
        # if using RGB change first param to 3
        self.conv1 = nn.Conv2d(1,16,3)
        # max pooling
        self.pool = nn.MaxPool2d(2,2)
        # second convolutional layer
        self.conv2 = nn.Conv2d(16,64,3)
        # fully connected layer 1
        self.fc1 = nn.Linear(5*5*64,120)
        # fully connected layer 2
        self.fc2 = nn.Linear(120, 84)
        # output layer
        self.fc3 = nn.Linear(84, 10)
    
    # forward propagation through network
    def forward(self, x):
        # convolution + relu + max pool
        x = self.pool(F.relu(self.conv1(x)))
        # convolution + relu + max pool
        x = self.pool(F.relu(self.conv2(x)))
        # reshape data for FC layers
        x = x.view(-1,5*5*64)
        # fc layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # output layers
        x = self.fc3(x)
        # return output
        return x

# initialise CNN
cnn = CNN()

# define a loss function and optimizer
loss_criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.9)

# training loop

for epoch in range(n_epochs):
    
    running_loss = 0.0
    
    for counter, data in enumerate(trainloader, 0):
        
        # unpack the training data into labels and images
        images, labels = data
        
        # zero the gradients
        optimizer.zero_grad()
        
        # forward propagation
        output = cnn(images)
        # calculate loss
        loss = loss_criterion(output, labels)
        # backpropagate loss
        loss.backward()
        # update weights
        optimizer.step()
        # record loss
        running_loss += loss.item()
        
        # print statistics
        running_loss += loss.item()
        if counter % 15000 == 14999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, counter + 1, running_loss / 15000))
            running_loss = 0.0

print("finished training")

# test accuracy on the whole dataset
correct = 0
total = 0

with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = cnn(images)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()
            
print("accuracy = " + str(correct/total))



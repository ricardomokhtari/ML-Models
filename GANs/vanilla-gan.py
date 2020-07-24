import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from utils import Logger

# set up function for downloading MNIST
def mnist_data():
    compose = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.5], [0.5])
        ])
    # where dataset will be stored
    out_dir = './dataset'
    return datasets.MNIST(root=out_dir, train=True, transform=compose, download=True)

# Load data
data = mnist_data()

# Create loader with data, so that we can iterate over it
data_loader = torch.utils.data.DataLoader(data, batch_size=100, shuffle=True)
# Num batches
num_batches = len(data_loader)

# Discriminator - input is flattened image, returns scalar between 0 and 1
# input size is 28x28=784
# output size is 1 (true or false node)
# network has 3 hidden layers, followed by ReLUs & dropout layer

class DiscriminatorNet(nn.Module):
    """
    A three hidden-layer discriminative neural network
    """
    def __init__(self):
        super(DiscriminatorNet, self).__init__()
        n_features = 784
        n_out = 1
        
        # first hidden layer, out_channels=1024
        self.hidden0 = nn.Sequential( 
            nn.Linear(n_features, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        # second hidden layer, out_channels=512
        self.hidden1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        # third hidden layer, out_channels=256
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        # output layer, out_channels=1
        self.out = nn.Sequential(
            torch.nn.Linear(256, n_out),
            torch.nn.Sigmoid()
        )
    
    # forward propagation
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

# create instance of discriminator
discriminator = DiscriminatorNet()

# reshaping functions
# reshape image to vector (flatten)
def images_to_vectors(images):
    return images.view(images.size(0), 784)

# reshape vector to image
def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)

# Generator - input is random vector of size 100, output is vector size 784 (flat image)
# 3 hidden layers
# output layer has tanh activation function because images are normalised
# between -1 and 1

class GeneratorNet(torch.nn.Module):
    """
    A three hidden-layer generative neural network
    """
    def __init__(self):
        super(GeneratorNet, self).__init__()
        n_features = 100
        n_out = 784
        
        # first hidden layer
        self.hidden0 = nn.Sequential(
            nn.Linear(n_features, 256),
            nn.LeakyReLU(0.2)
        )
        # second hidden layer
        self.hidden1 = nn.Sequential(            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2)
        )
        # third hidden layer
        self.hidden2 = nn.Sequential(
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2)
        )
        # out layer
        self.out = nn.Sequential(
            nn.Linear(1024, n_out),
            nn.Tanh()
        )

    # forward propagation
    def forward(self, x):
        x = self.hidden0(x)
        x = self.hidden1(x)
        x = self.hidden2(x)
        x = self.out(x)
        return x

# create instance of a generator
generator = GeneratorNet()

# function that creates noise (input for G)
def noise(size):
    '''
    Generates a 1-d vector of gaussian sampled random values
    '''
    n = Variable(torch.randn(size, 100))
    return n

# define Adam optimiser for the networks
d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

# define loss function (BCE Loss)
loss = nn.BCELoss()

# real images assigned value of 1
def ones_target(size):
    '''
    Tensor containing ones, with shape = size
    '''
    data = Variable(torch.ones(size, 1))
    return data

# fake images assigned values of 0
def zeros_target(size):
    '''
    Tensor containing zeros, with shape = size
    '''
    data = Variable(torch.zeros(size, 1))
    return data

# full function for training the discriminator
def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    
    # 1.1 Train on Real Data
    # forward propagation of the real image
    prediction_real = discriminator(real_data)
    # Calculate gradient
    error_real = loss(prediction_real, ones_target(N) )
    # backpropagation
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate gradient
    error_fake = loss(prediction_fake, zeros_target(N))
    # backpropagation
    error_fake.backward()
    
    # 1.3 Update weights with gradients
    optimizer.step()
    
    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake

# function for training generator
def train_generator(optimizer, fake_data):
    N = fake_data.size(0)  

    # reset gradients  
    optimizer.zero_grad()

    # sample noise and generate fake data
    prediction = discriminator(fake_data)

    # calculate error + backpropagation
    error = loss(prediction, ones_target(N))
    error.backward()

    # update weights
    optimizer.step()

    # return error
    return error

num_test_samples = 16
test_noise = noise(num_test_samples)

# Create logger instance
logger = Logger(model_name='VGAN', data_name='MNIST')

# total no. of epochs
num_epochs = 50

for epoch in range(num_epochs):
    for n_batch, (real_batch,_) in enumerate(data_loader):
        N = real_batch.size(0)        

        # 1. Train Discriminator
        real_data = Variable(images_to_vectors(real_batch))

        # Generate fake data and detach 
        # (so gradients are not calculated for generator)
        fake_data = generator(noise(N)).detach()

        # Train D
        d_error, d_pred_real, d_pred_fake = \
              train_discriminator(d_optimizer, real_data, fake_data)

        # 2. Train Generator        
        # Generate fake data
        fake_data = generator(noise(N))

        # Train G
        g_error = train_generator(g_optimizer, fake_data)  

        # Log batch error
        logger.log(d_error, g_error, epoch, n_batch, num_batches)  

        # Display Progress every few batches
        if (n_batch) % 100 == 0: 
            test_images = vectors_to_images(generator(test_noise))
            test_images = test_images.data            
            logger.log_images(
                test_images, num_test_samples, 
                epoch, n_batch, num_batches
            );

            # Display status Logs
            logger.display_status(
                epoch, num_epochs, n_batch, num_batches,
                d_error, g_error, d_pred_real, d_pred_fake
            )
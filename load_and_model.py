from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms 
import os
import PIL
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.model_selection import train_test_split
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim



 
### Globals ###
videos_root = os.path.join(os.getcwd(), 'PEDFE_trim')
annotation_file = os.path.join(videos_root, 'annotations.txt')

# label # to label word
labels_dict = {0: "posed", 1: "genuine"}


# Returns train_loader, validation_loader
def get_loaders(dataset, batch_size, validation_split, num_workers=2, pin_memory=True):
    # https://stackoverflow.com/questions/50544730/how-do-i-split-a-custom-dataset-into-training-and-test-datasets
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))


    # shuffle dataset
    np.random.seed(42)
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # https://pytorch.org/docs/stable/data.html
    # convert to more workable from
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=train_sampler, num_workers=num_workers, pin_memory=pin_memory)
    
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler, num_workers=num_workers, pin_memory=pin_memory)


    return train_loader, validation_loader    


    
if __name__ == "__main__":


    #### Get dataset from images ###
    
    # put images into Pytorch workable form
    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
      
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), --> Maybe include this
    ])

    # break video clip down into five different segments (each is a list of frames), take 1 frame from each segment
    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=1, # was set to 5 before (i.e. 5 frames per video taken) but presents problems because a 5D tensor is created (cant do Conv2D natively)
        frames_per_segment=1,
        imagefile_template='img_{:06d}.jpg',
        transform=preprocess,
        test_mode=False
    )


    #### Split data into train and validation and put into loaders ####

    batch_size = 3
    validation_split = .2

    train_loader, validation_loader = get_loaders(dataset, batch_size, validation_split)

    ### Screwing around ####

    # NOTE: each VideoFrameDataset object is a list of samples where each sample is
    # a list of frames (tensor) from a video clip --> sample[0]
    # label (genuine, posed) for that clip --> sample[1]
    sample = (dataset[202])


    # shuffle and split into testing and training
    # random.shuffle(dataset)
    # training_part = .80 # fraction of dataset that is for training
    # testing_start = int(len(dataset) * .80)
    # training_data = dataset[:testing_start]
    # testing_data = dataset[testing_start:]

    frames = sample[0]  # list of PIL images
    label = sample[1]   # integer label,   # NOTE: posed=0, genuine=1s

    print("num batches = ", len(train_loader))
    
        
    #### Define Model to Train ####

    # Currently using a basic one ripped from pytorch.org
    # https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html


    # model
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(112896, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            if x.dim() == 5:
                B, EXTRA, C, H, W = x.shape
            x = x.view(-1, C, H, W)

            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1) # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    net = Net()

    # loss function
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



    ### TRAIN ####

    print("Train Model on training data...")
    for epoch in range(3):
         running_loss = 0
         for i, batch_data in enumerate(train_loader, 0):

            inputs, labels = batch_data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 20 == 19:    # print every 20 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 20:.3f}')
                running_loss = 0.0

    print('Finished Training')




    # save trained model
    PATH = './basic_model.pth'
    torch.save(net.state_dict(), PATH)

    # Testing (with validation data)
    
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in validation_loader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')

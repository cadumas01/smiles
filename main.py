from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms 
import os
import PIL
import sys

from load import get_loaders
from train import *
# =============================================================================
import torch
from tqdm import tqdm

# Testing (with validation data)
def test(model, validation_loader, device):
    correct = 0
    total = 0

    model.eval()
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in tqdm(validation_loader):
            print("testing data = ", data[1])
            images, labels = data

            images = images.to(device)
            #labels = labels.to(device)

            # calculate outputs by running images through the network
            outputs = model(images).to(device)
            labels = labels.to(device)


            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            predicted = predicted.to(device)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            torch.cuda.empty_cache()
    print(
        f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')


# =============================================================================
from models import *

import matplotlib.pyplot as plt
import numpy as np

### Globals ###
videos_root = os.path.join(os.getcwd(), 'PEDFE_trim')
annotation_file = os.path.join(videos_root, 'annotations.txt')

import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:<enter-size-here>"

# label # to label word
labels_dict = {0: "posed", 1: "genuine"}
num_classes = 2

batch_size = 2
validation_split = .2
num_frames = 10
num_training_epochs = 90


# runs everything
if __name__ == "__main__":
    
  
    if len(sys.argv) > 2:
        print("usage: python3 main.py [retrain]")
        print("add the argument 'retrain' to retrain the model, otherwise saved previous neural network will be loaded")
        exit()

    dev = "cpu"
    if torch.cuda.is_available(): 
        print("using cuda")
        dev = "cuda:0" 
    
    
    
    device = torch.device(dev) 

    #### Get dataset from images ###
    
    # put images into Pytorch workable form
    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
      
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # break video clip down into five different segments (each is a list of frames), take 1 frame from each segment
    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments= num_frames, # was set to 5 before (i.e. 5 frames per video taken) but presents problems because a 5D tensor is created (cant do Conv2D natively)
        frames_per_segment=1,
        imagefile_template='img_{:06d}.jpg',
        transform=preprocess,
        test_mode=False
    )

    #### Split data into train and validation and put into loaders ####
    print("Loading data")

    # Number of workers for the dataloader
    num_workers = 0 if device.type == 'cuda' else 2
    # Whether to put fetched data tensors to pinned memory
    # pin_memory = True if device.type == 'cuda' else False
    pin_memory = False

    train_loader, validation_loader = get_loaders(dataset, batch_size, validation_split, num_workers=num_workers, pin_memory=pin_memory)

    sample_video = (train_loader.dataset.video_list[0])

    print("sample video: ", sample_video.__dict__)

    
    # Define model
    print("Defining model...")
    model = CNN_LSTM(num_frames=num_frames)

    # put model on gpu (or cpu if no cuda)
    model = model.to(device)

    # train model (if applicable)
    if 'retrain' in sys.argv:
        print("Training model...")
        training_losses = train2(model, train_loader, num_training_epochs, device)    

        iterations = np.arange(training_losses.size)

        print('loss : ', training_losses)
        print(" x tick: ", iterations)
        plt.plot(training_losses)
        plt.title("Training Loss over Epoch + Batches")
        plt.ylabel("Loss")
        plt.xlabel("Iterations")
        plt.savefig("TrainingLoss.png")

    # save trained model
    PATH = './basic_model.pth'

    torch.cuda.empty_cache()


    torch.save(model.state_dict(), PATH)
    print(f"Saved trained model as {PATH}")

    
    # Test Data
    print("Testing model against validation data...")
    test(model, validation_loader, device)
    print("Validation complete.")
from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms 
import os
import PIL
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.model_selection import train_test_split
import random
import torch
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


 
### Globals ###
videos_root = os.path.join(os.getcwd(), 'PEDFE_trim')
annotation_file = os.path.join(videos_root, 'annotations.txt')

# label # to label word
labels_dict = {0: "posed", 1: "genuine"}


def plot_video(rows, cols, frame_list, plot_width, plot_height, title: str):
    fig = plt.figure(figsize=(plot_width, plot_height))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(rows, cols),  # creates 2x2 grid of axes
                     axes_pad=0.3,  # pad between axes in inch.
                     )

    for index, (ax, im) in enumerate(zip(grid, frame_list)):
        # Iterating over the grid returns the Axes.
        ax.imshow(im)
        ax.set_title(index)
    plt.suptitle(title)
    plt.show()


if __name__ == "__main__":

    # put images into Pytorch workable form
    preprocess = transforms.Compose([
        ImglistToTensor(),  # list of PIL images to (FRAMES x CHANNELS x HEIGHT x WIDTH) tensor
      
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), --> Maybe include this
    ])

    # break video clip down into five different segments (each is a list of frames), take 1 frame from each segment
    dataset = VideoFrameDataset(
        root_path=videos_root,
        annotationfile_path=annotation_file,
        num_segments=5,
        frames_per_segment=1,
        imagefile_template='img_{:06d}.jpg',
        transform=preprocess,
        test_mode=False
    )

    batch_size = 3
    num_workers = 2
    pin_memory = True # not sure if this is necessary

    validation_split = .2

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
    exit(0)
    
    print(" Train Model on training data...")
    for epoch in range(10):
        for batch in train_loader:

            print("batch = ", len(batch))
            
            """
            Insert Training Code Here
            """
            video_batch, label = batch

            print("Video Batch Tensor Size:", video_batch.size())
            print("Labels1 Size:", label.size())  # == batch_size
            print("Label = ", label) # a tensor containing the labels for each sample in the batch (if batch_size=3 then tensor has 3 labels)
          

        
           
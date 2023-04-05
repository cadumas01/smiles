from video_dataset import  VideoFrameDataset, ImglistToTensor
from torchvision import transforms 
import os
import PIL
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid


# IN PROCESSING SCRIPT, convert imagesd to jpg
 
videos_root = os.path.join(os.getcwd(), 'PEDFE_trim')
annotation_file = os.path.join(videos_root, 'annotations.txt')


# break video clip down into five different segments (each is a list of frames), take 1 frame from each segment
dataset = VideoFrameDataset(
    root_path=videos_root,
    annotationfile_path=annotation_file,
    num_segments=5,
    frames_per_segment=1,
    imagefile_template='img_{:06d}.jpg',
    transform=None,
    test_mode=False
)

sample = dataset[100]
frames = sample[0]  # list of PIL images
label = sample[0]   # integer label,   # NOTE: posed=0, genuine=1

print('dataset ', dataset[1])
print("sample ", sample)
print("frames" , frames)

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
                  
plot_video(rows=1, cols=5, frame_list=frames, plot_width=15., plot_height=3.,
        title='Evenly Sampled Frames, No Video Transform')
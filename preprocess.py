# quick script for removing extra files from PEDFE dataset before uploading to google drive and organizing data
# for final project
import zipfile
import os
from PIL import Image

# directory structuring and dataloading based on https://video-dataset-loading-pytorch.readthedocs.io/en/latest/



dir_path = "PEDFE_images"
unzip_dir_path = dir_path + "_unzipped"

dst_parent = "PEDFE_trim/"
annotations_path = dst_parent + "annotations.txt"

# remove old files and dirs before creating new ones
def clean():

    if os.path.isfile(annotations_path):
        os.system(f"rm {annotations_path}")
    
    if os.path.exists(dst_parent):
        os.system(f"rm -r {dst_parent}")


if __name__ == "__main__":
    clean()

    os.mkdir(dst_parent)

    # create annotations file 
    annotations = open(annotations_path, "w+")

for entry in os.listdir(dir_path):

    # f is happiness
    entry_parts = entry.split("_")

    # only work with happiness data
    if "f" in entry_parts[1]:
        
        # two part path {posed OR genuine}/{PARTICPANT_VIDEO}/{FRAMES}
        frames_dir = dst_parent

        # 0 for posed, 1 for genuine
        label = 0

        if "s" in entry_parts[1]:
            frames_dir +="posed/"
        else:
            frames_dir += "genuine/"
            label = 1
        
        frames_dir += f"{entry_parts[0]}_{entry_parts[2]}" + "/"

        # unzip each video into series of frames into a folder named after entry name
        with zipfile.ZipFile(f"{dir_path}/{entry}", "r") as zip_ref:
            zip_ref.extractall(frames_dir)

        # for each frame, write a line in notations satisying:
        # [PATH NAME] [FRAME] [LABEL]
        # NOTE: posed=0, genuine=1

        for frame in os.listdir(frames_dir):       

            frame_name_parts = frame.split("_")
            new_frame_name = (frame_name_parts[3]) 
            # simplify naming
            os.rename(frames_dir + frame  , frames_dir + new_frame_name)

            # convert each frame from .bmp to .jpg?

            # add annotation for this frame
            annotations.write(f"{frames_dir.removeprefix(dst_parent)} {new_frame_name} {label}\n")






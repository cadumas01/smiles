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

    # each entry is a video clip (a folder of frames)
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

            # for each clip, write a line in notations satisying:
            # [PATH NAME] [Start frame #] [End frame #] [CLASS INDEX]
            # NOTE: posed=0, genuine=1


            sorted_frames = (os.listdir(frames_dir))
            sorted_frames.sort()

            
            for indx in range(len(sorted_frames)):       

                frame_name = sorted_frames[indx]
                
                # rename all frames based on index (so frames are number [1, 2, ..., NUM_FRAMES] without gaps)
                new_frame_name = "img_" +f"{(indx +1):06d}" + ".bmp"

                # simplify naming
                os.rename(frames_dir + frame_name  , frames_dir + new_frame_name)

                # convert each frame from .bmp to .jpg? YES
                img = Image.open(frames_dir + new_frame_name)
                rgb_img = img.convert("RGB")
                jpg_frame_path = (frames_dir + new_frame_name).replace(".bmp", ".jpg")
                rgb_img.save(jpg_frame_path)

                # remove old image
                os.remove(frames_dir + new_frame_name)


            num_frames = len(os.listdir(frames_dir))

            # add annotation for this clip, start_frame is always 1
            annotations.write(f"{frames_dir.removeprefix(dst_parent)} 1 {num_frames} {label}\n")






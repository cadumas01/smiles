# quick script for removing extra files from PEDFE dataset before uploading to google drive
# for final project
import zipfile
import os

dir_path = "PEDFE_images"
unzip_dir_path = dir_path + "_unzipped"

for entry in os.listdir(dir_path):

    # f is happiness
    entry_parts = entry.split("_")

    # only work with happiness data
    if "f" not in entry_parts[1]:
        
        with zipfile.ZipFile(f"{dir_path}/{entry}", "r") as zip_ref:
            zip_ref.extractall(unzip_dir_path)
        
        print(f'entry = {entry}')
        exit()


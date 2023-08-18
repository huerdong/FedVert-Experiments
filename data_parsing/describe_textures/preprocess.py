import os
import pandas as pd
import shutil
import json

root_dir = "/ws/data_parsing/describe_textures/dtd"
orig_images_folder = "images"
new_images_folder = "images_new"
metadata_folder = "labels"

data_dir = os.path.join(root_dir, orig_images_folder)
new_data_dir = os.path.join(root_dir, new_images_folder)
metadata_dir = os.path.join(root_dir, metadata_folder)

# We swap train and test split to increase diversity of client samples
traintest_split = {}

split_id = 1 # We choose split 1 out of the 10 arbitrarily
md_files = ["test", "train", "val"]
for f in md_files:
    with open(os.path.join(metadata_dir, f"{f}{split_id}.txt")) as mdfile:
        data = mdfile.readlines()
        is_train = 1
        if f == "test":
            is_train = 0 # Do this swap here. Validation set goes in train set. 
        for row in data:
            traintest_split[row.strip()] = is_train

# We only care about the primary label for this 
labels_file = "labels_joint_anno.txt" 
labels = {}
with open(os.path.join(metadata_dir,f"{labels_file}")) as labelsfile:
    data = labelsfile.readlines()
    for row in data:
        spl = row.strip().split()
        label, _ = spl[0].split('/')
        labels[spl[0]] = label 

os.makedirs(os.path.join(new_data_dir,'train'), exist_ok=True)
os.makedirs(os.path.join(new_data_dir,'test'), exist_ok=True)

print(traintest_split)
#for image in image_fnames:
#    print(image)

with open("traintest_split.json", "w") as split_f:
    split_f.write(json.dumps(traintest_split, indent=4))

for image, label in labels.items():
    if traintest_split[image] == 1:
        output_type = 'train'
    else: 
        output_type = 'test'
    output_dir = os.path.join(new_data_dir, output_type, label)
    os.makedirs(output_dir, exist_ok=True)
    print('Image:: ', image)
    print('Destination:: ', output_dir)
    shutil.copy(src=os.path.join(data_dir, image), dst=os.path.join(new_data_dir, output_type, image))
            
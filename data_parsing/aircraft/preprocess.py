import os
import pandas as pd
import shutil
import json

root_dir = "/ws/data_parsing/aircraft/fgvc-aircraft-2013b/data"
orig_images_folder = "images"
new_images_folder = "images_new"
#metadata_folder = "metadata"

data_dir = os.path.join(root_dir, orig_images_folder)
new_data_dir = os.path.join(root_dir, new_images_folder)
metadata_dir = os.path.join(root_dir)

image_fnames = os.listdir(data_dir)

# We swap train and test split to increase diversity of client samples
traintest_split = {}
labels = {}

mdprefix = "images_"
mdtypeprefix = "images_variant_"
md_files = ["test", "train", "val"]

for f in md_files:
    with open(os.path.join(metadata_dir, f"{mdprefix}{f}.txt")) as mdfile:
        data = mdfile.readlines()
        is_train = 1
        if f == "test":
            is_train = 0 # Do this swap here. Validation set goes in train set. 
        for row in data:
            item = int(row)
            traintest_split[item] = is_train

    with open(os.path.join(metadata_dir,f"{mdtypeprefix}{f}.txt")) as labelsfile:
        data = labelsfile.readlines()
        for row in data:
            spl = row.strip().split()
            item = spl[0]
            label = " ".join(spl[1:])
            labels[int(item)] = label 

os.makedirs(os.path.join(new_data_dir,'train'), exist_ok=True)
os.makedirs(os.path.join(new_data_dir,'test'), exist_ok=True)

print(traintest_split)
#for image in image_fnames:
#    print(image)

with open("traintest_split.json", "w") as split_f:
    split_f.write(json.dumps(traintest_split, indent=4))

for image in image_fnames:
    index = int(image.split('.')[0])
    label = labels[index]
    if traintest_split[index] == 1:
        output_type = 'train'
    else: 
        output_type = 'test'
    output_dir = os.path.join(new_data_dir,output_type, str(label))
    os.makedirs(output_dir, exist_ok=True)
    print('Image:: ', image)
    print('Destination:: ', output_dir)
    shutil.copy(src=os.path.join(data_dir, image), dst=os.path.join(output_dir, image))
            
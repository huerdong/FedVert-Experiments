import os
import pandas as pd
from skimage import io
from PIL import Image
import pickle
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

base_dataset_path = "./CUB_200_2011/"
images_df = pd.read_csv(os.path.join(base_dataset_path, "images.txt"), sep=" ", names=["image_id", "path"])

train_test_split = pd.read_csv(os.path.join(base_dataset_path, "train_test_split.txt"), sep=" ", names=["img_id", "is_train"])

bounding_boxes = pd.read_csv(os.path.join(base_dataset_path, "bounding_boxes.txt"), sep=" ", names=["img_id", "x", "y", "width", "height"])

images_df["is_train"] = train_test_split.is_train

images_df["x"] = bounding_boxes.x

images_df["y"] = bounding_boxes.y

images_df["width"] = bounding_boxes.width

images_df["height"] = bounding_boxes.height

images_df["class_id"] = images_df.apply(lambda p : (int(p.path[:3])-1), axis=1)

images_df.to_csv(os.path.join(base_dataset_path, "birds_images.csv"))

train_df = images_df[images_df.is_train == 1].drop(labels="is_train", axis=1).drop(labels="image_id", axis=1)#.drop(images_df.columns[0], axis=1)
test_df = images_df[images_df.is_train == 0].drop(labels="is_train", axis=1).drop(labels="image_id", axis=1)

train_df.to_csv(os.path.join(base_dataset_path, "birds_train.csv"))
test_df.to_csv(os.path.join(base_dataset_path, "birds_test.csv"))
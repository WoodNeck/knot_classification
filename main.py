import os
import re
from preprocess import preprocess
from feature_extraction import extract_feature
from training import train


img_dir = "augmentation"
data_file = "names.txt"

with open(data_file, "r") as data:
    datas = data.readlines()

orig_labels = [data.rstrip('\n').split()[1] for data in datas]

img_paths = []
img_labels = []

files = os.listdir(img_dir)
for file in files:
    image_num = int(re.search("(knot)(\d+)(_0_)(\d+)(.ppm)", file).group(2)) - 1
    path = os.path.join(img_dir, file)
    img_paths.append(path)
    img_labels.append(orig_labels[image_num])

# print("Starting Preprocessing...")
# preprocess(img_dir, datas)
# print("Preprocessing Done...")

print("Extracting Feature...")
extract_feature(img_paths, img_labels)
print("Feature Extracting Done...")

print("Starting Training...")
train(datas, "data")
print("Training Done...")

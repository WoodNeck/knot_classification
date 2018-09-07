from preprocess import preprocess
from feature_extraction import extract_feature
from training import train


img_dir = "knots"
data_file = "names.txt"

with open(data_file, "r") as data:
    datas = data.readlines()

datas = [data.rstrip('\n').split() for data in datas]

# print("Starting Preprocessing...")
# preprocess(img_dir, datas)
# print("Preprocessing Done...")

print("Extracting Feature...")
# extract_feature(datas)
print("Feature Extracting Done...")

print("Starting Training...")
train(datas, "data")
print("Training Done...")

"""
# import matplotlib.pyplot as plt
plt.subplot(1, 3, 1), plt.imshow(img, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('Original')
plt.subplot(1, 3, 2), plt.imshow(preprocessed_img, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('After diffusion')
plt.subplot(1, 3, 3), plt.imshow(gradient_image, cmap='gray')
plt.xticks([]), plt.yticks([])
plt.xlabel('Gradient after diffusion')
plt.show()
"""

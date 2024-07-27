import numpy as np
import os
import pickle
from PIL import Image
from skimage import color
from dataset_utils import *
from model_utils import get_p
from quantisation import quantize_ab_image, resize_image


def preprocess_image(image):
    image = resize_image(image)
    lab_image = color.rgb2lab(image)
    l_image = lab_image[:, :, 0]
    ab_image = lab_image[:, :, 1:]
    return l_image, ab_image

def load_images(data_size):
    images = os.listdir("images")
    l_images = np.empty((data_size, 256, 256))
    ab_images = np.empty((data_size, 256, 256, 2))
    for i in range(data_size):
        img_as_matrix = np.asarray(Image.open(f"images/{images[i]}"))
        l_img, ab_img = preprocess_image(img_as_matrix)
        l_images[i] = l_img
        ab_images[i] = ab_img

        if i % 50 == 49:  # print every 50 mini-batches
            print("Loaded data nr.: " + str(i + 1))

    l_images = l_images.astype(np.int8)
    ab_images = ab_images.astype(np.int8)


    print("Data loaded.")
    return l_images, ab_images

l_img, ab_img = load_images(10000)

ab_img = quantize_ab_image(ab_img)

p = get_p(ab_img, Q=272)

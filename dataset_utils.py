from pathlib import Path
import pickle
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
from skimage.color import rgb2lab, lab2rgb, rgb2xyz, xyz2rgb
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from model_utils import ab_to_z


def convert_to_bw(image):
    return image.convert("L")


def load_img(img_path):
    img = Image.open(img_path).convert("RGB")
    img_np = np.asarray(img)
    return img_np

def resize_img(img, HW=(256, 256), resample=Image.BICUBIC):
    img = Image.fromarray(img)
    img = img.resize(HW, resample)
    return np.asarray(img)

def preprocess_img(img_rgb_orig, HW=(256, 256), resample=Image.BICUBIC, get_probs=True):
    img_rgb_rs = resize_img(img_rgb_orig, HW=HW, resample=resample)
    img_lab_orig = rgb2lab(img_rgb_orig)
    img_lab_rs = rgb2lab(img_rgb_rs)
    img_l_orig = img_lab_orig[:, :, 0]
    img_l_rs = img_lab_rs[:, :, 0]
    img_ab_rs = img_lab_rs[:, :, 1:]  # AB channels
    img_z = ab_to_z(img_ab_rs, 274)
    tens_orig_l = tf.convert_to_tensor(img_l_orig, dtype=tf.float32)[tf.newaxis, :, :, tf.newaxis]
    tens_rs_l = tf.convert_to_tensor(img_l_rs, dtype=tf.float32)[tf.newaxis, :, :, tf.newaxis]
    if get_probs:
        tens_rs_ab = tf.convert_to_tensor(img_z, dtype=tf.float32)[tf.newaxis, :, :, :]
    else:
        tens_rs_ab = tf.convert_to_tensor(img_ab_rs, dtype=tf.float32)[tf.newaxis, :, :, :]
    return tens_orig_l, tens_rs_l, tens_rs_ab

def create_dataset(image_folder, batch_size=32, validation_split=0.25, HW=(256, 256), resample=Image.BICUBIC, get_probs=True):
    image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
    np.random.shuffle(image_paths)
    split_index = int(len(image_paths) * (1 - validation_split))
    train_paths = image_paths[:split_index]
    val_paths = image_paths[split_index:]

    def load_and_preprocess(image_path):
        img = load_img(image_path.numpy().decode('utf-8'))
        return preprocess_img(img, HW, resample, get_probs)

    def tf_load_and_preprocess(image_path):
        img_l_orig, img_l_rs, img_ab_rs = tf.py_function(load_and_preprocess, [image_path], [tf.float32, tf.float32, tf.float32])
        img_l_orig.set_shape([1, None, None, 1])
        img_l_rs.set_shape([1, HW[0], HW[1], 1])
        if get_probs:
            img_ab_rs.set_shape([1, HW[0], HW[1], 274])
        else:
            img_ab_rs.set_shape([1, HW[0], HW[1], 2])
        return img_l_orig[0], img_l_rs[0], img_ab_rs[0]

    def discard_first_tensor(img_l_orig, img_l_rs, img_ab_rs):
        return img_l_rs, img_ab_rs
    

    train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
    train_dataset = train_dataset.map(lambda x: tf_load_and_preprocess(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.map(lambda x, y, z: discard_first_tensor(x, y, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_tensor_slices(val_paths)
    val_dataset = val_dataset.map(lambda x: tf_load_and_preprocess(x), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.map(lambda x, y, z: discard_first_tensor(x, y, z), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return train_dataset, train_paths, val_dataset, val_paths







def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename)).convert("RGB")
        if img is not None:
            images.append(img)
    return images


def visualize_images(images, bw_images):
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, num_images * 5))
    
    for i in range(num_images):
        axes[i, 0].imshow(bw_images[i], cmap='gray')
        axes[i, 0].set_title("Training Data")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(images[i])
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()



def adjust_color(Z, T, ab_domain):
    def temperature_scaling(Z, T):
        Z = np.exp(np.log(Z) / T) / np.sum(np.exp(np.log(Z) / T), axis=2)[:, :, np.newaxis]
        return Z
    Z = temperature_scaling(Z, T)

    Z = Z / np.sum(Z, axis=2)[:, :, np.newaxis]
    Z = Z * (3 / np.exp(T))

    ab_domain = np.array(ab_domain)
    final_ab = np.sum(Z[:, :, np.newaxis] * ab_domain[np.newaxis, np.newaxis, :, :], axis=2)
    return final_ab

def lab2rgb_with_gamut_mapping(lab_image):
    rgb_image = lab2rgb(lab_image)
    xyz_image = rgb2xyz(rgb_image)
    xyz_image = np.clip(xyz_image, 0, 1)  # Clip XYZ values to 0-1 range
    rgb_image = xyz2rgb(xyz_image)
    return rgb_image

# ---------------------------------------------
def temperature_scaling(Z, T):
    """
    Apply temperature scaling to the model output probabilities.
    """
    Z = np.exp(np.log(Z) / T)
    Z /= np.sum(Z, axis=-1, keepdims=True)
    return Z

def convert_Z_to_ab(Z, ab_domain, T=0.38):
    """
    Convert the model output Z (probability distribution) to ab values.
    
    Parameters:
    Z : np.ndarray
        The model output with shape (H, W, Q) where Q is the number of quantized bins.
    ab_domain : np.ndarray
        The quantized ab domain with shape (Q, 2).
    T : float
        Temperature parameter for scaling.
        
    Returns:
    ab_output : np.ndarray
        The computed ab values with shape (H, W, 2).
    """
    # Apply temperature scaling
    Z = temperature_scaling(Z, T)
    
    # Compute expected ab values
    ab_output = np.dot(Z, ab_domain)
    
    return ab_output

# ---------------------------------------------

def H(Z, T, ab_domain):
    T = np.array(T)
    def f_T(Z):
        print("Z shape: ", Z.shape)
        print("T shape: ", T[np.newaxis, :, :].shape)
        a = np.exp(np.log(Z) / T[np.newaxis, :, :])
        b = np.sum(np.exp(np.log(Z) / T), axis=2)[:, :, np.newaxis]
        Z = np.exp(np.log(Z) / T[np.newaxis, :, :]) / np.sum(np.exp(np.log(Z)) / T, axis=2)[:, :, np.newaxis]
        return Z
    Z = f_T(Z)

    # Minmax_scale
    # Z_std = (Z - Z.min(axis=2)[:, :, np.newaxis]) / (Z.max(axis=2) - Z.min(axis=2))[:, :, np.newaxis]
    # Z = Z_std * (1 - 0) + 0

    Z = Z / np.sum(Z, axis=2)[:, :, np.newaxis]
    Z = Z * (3/np.exp(T))      # Higher saturation

    ab_domain = np.array(ab_domain)
    final_ab = np.sum(Z[:, :, :, np.newaxis] * ab_domain[np.newaxis, np.newaxis, :, :], axis=2)
    return final_ab

def postprocess_output(l_img, Z_output_tens, ab_domain, T=0.38):
    print("Z_output shape: ", Z_output_tens.shape)
    Z_output = Z_output_tens[0]
    # Z_output = np.moveaxis(Z_output, -1, 0)[0]
    # Z_output = Z_output[0]
    print("Z_output shape changed: ", Z_output.shape)
    ab_output = convert_Z_to_ab(Z_output, ab_domain, T)
    # print("ab_output shape: ", ab_output.shape)
    lab_output = np.empty((l_img.shape[0], l_img.shape[1], 3))
    # print("lab_output shape: ", lab_output.shape)
    lab_output[:, :, 0] = l_img[:, :, 0]
    lab_output[:, :, 1:] = ab_output
    rgb_output = lab2rgb_with_gamut_mapping(lab_output)
    return Z_output, lab_output, rgb_output

def postprocess_tens(tens_orig_l, out_ab):
    HW_orig = tf.shape(tens_orig_l)[:2]
    HW = tf.shape(out_ab)[:2]
    if HW_orig[0] != HW[0] or HW_orig[1] != HW[1]:
        out_ab_orig = tf.image.resize(out_ab, HW_orig, method='bilinear')
    else:
        out_ab_orig = out_ab
    out_lab_orig = tf.concat([tens_orig_l, out_ab_orig], axis=-1)
    out_lab_orig_np = out_lab_orig.numpy()
    return lab2rgb(out_lab_orig_np)
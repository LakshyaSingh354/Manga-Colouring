import matplotlib.pyplot as plt
import numpy as np
from skimage import color

def plot_ab_channels(a, b, path='output', save_image=False):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    data = [('A', a), ('B', b)]
    for ax, (title, img) in zip(axes, data):
        ax.set_title(title)
        ax.imshow(img, vmin=-110, vmax=110, cmap='Greys')
        ax.axis('off')
    fig.tight_layout()
    if save_image:
        plt.savefig(path)
    plt.show()

def plot_l_and_ab_channels(img_l, img_ab, l_value=50):
    # Create an empty LAB image with L channel set to l_value
    h, w = img_l.shape
    img_lab = np.zeros((h, w, 3))
    img_lab[:, :, 0] = img_l  # Set L channel to the given L channel
    img_lab[:, :, 1:] = img_ab  # Set AB channels

    # Convert LAB image to RGB
    img_rgb = color.lab2rgb(img_lab)

    # Plot the images side by side
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Plot the black and white (L channel) image
    ax[0].imshow(img_l, cmap='gray')
    ax[0].axis('off')
    ax[0].set_title('L Channel (Black & White)')

    # Plot the color image
    ax[1].imshow(img_rgb)
    ax[1].axis('off')
    ax[1].set_title('Color Image')

    plt.show()



def plot_images(original_rgb, output_rgb, T=0.38, path='output', save_image=False):
    fig, axes = plt.subplots(ncols=2, figsize=(12, 4))
    data = [('Original', original_rgb), ('Result (T={})'.format(T), output_rgb)]
    for ax, (title, img) in zip(axes, data):
        ax.set_title(title)
        ax.imshow(img)
        ax.axis('off')
    fig.tight_layout()
    if save_image:
        plt.savefig(path)
    plt.show()

import os
import pickle
import numpy as np
from scipy.spatial import KDTree
import tensorflow as tf
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from colormath.color_diff import delta_e_cie2000
from colormath.color_objects import LabColor
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D, Dropout, Conv2DTranspose, Softmax, UpSampling2D # type: ignore
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
from data_processing_script import *

def get_ab_domain_from_keys(ab_key):
    ab = ab_key.split(",")
    a = int(ab[0][1:])
    b = int(ab[1][1:-1])
    return a, b

def get_ab_domain():
    ab_to_q_dict = pickle.load(open("ab_to_q_index_dict.p", "rb"))
    ab_domain_strings = list(ab_to_q_dict.keys())
    ab_domain = [list(get_ab_domain_from_keys(ab)) for ab in ab_domain_strings]
    return np.asarray(ab_domain)

def get_ab_to_q_dict():
    ab_domain = get_ab_domain()
    q_values = np.arange(0, len(ab_domain))
    ab_to_q_dict = dict(zip(map(tuple, ab_domain), q_values))

    return ab_to_q_dict

def get_q_to_ab_dict():
    ab_domain = get_ab_domain()
    q_values = np.arange(0, len(ab_domain))
    q_to_ab_dict = dict(zip(q_values, map(tuple, ab_domain)))

    return q_to_ab_dict


def ab_to_z(ab_image, Q, sigma=5):
    w, h = ab_image.shape[0], ab_image.shape[1]
    points = w * h
    ab_images_flat = np.reshape(ab_image, (points, 2))
    points_encoded_flat = np.empty((points, Q))
    points_indices = np.arange(0, points, dtype='int')[:, np.newaxis]

    nbrs = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(get_ab_domain())
    distances, indices = nbrs.kneighbors(ab_images_flat)

    gaussian_kernel = np.exp(-distances**2 / (2 * sigma**2))
    gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel, axis=1)[:, np.newaxis]

    points_encoded_flat[points_indices, indices] = gaussian_kernel
    points_encoded = np.reshape(points_encoded_flat, (w, h, Q))
    return points_encoded


def get_p(ab_image, Q=274):    
    ab_to_q_dict = get_ab_to_q_dict()
    ab_domain = get_ab_domain()
    batch_size, w, h, _ = ab_image.shape
    p = np.zeros(Q)

    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(ab_domain)

    flatten_ab_image = ab_image.reshape(-1, 2)

    def get_ab_indices(flatten_ab_image):
        indices = []
        for ab in tqdm(flatten_ab_image, desc="Finding nearest neighbors"):
            try:
                indices.append(ab_to_q_dict[tuple(ab)])
            except KeyError:
                _, nearest_index = nbrs.kneighbors([ab])
                nearest_ab = ab_domain[nearest_index[0][0]]
                indices.append(ab_to_q_dict[tuple(nearest_ab)])
        return indices

    indices = np.array(get_ab_indices(flatten_ab_image))

    for index in indices:
        p[index] += 1

    p /= batch_size * w * h
    print("p computed.")
    with open("p_{}--imgnet.p".format(batch_size), "wb") as f:
        pickle.dump(p, f)

    return p

def get_loss_weights(Z, Q, p_tilde, lam=0.5):
    # tf.print("Lambda set")
    a = 1 - lam
    # tf.print("a set")
    b = 0.5 / 274
    # tf.print("b set")
    w = ((a * p_tilde) + b) ** -1
    # tf.print("Weights computed...")
    w /= np.sum(p_tilde * w)
    # tf.print("Weights computed")
    q_star = tf.argmax(Z, axis=1)
    weights = tf.gather(w, q_star)
    # tf.print("Weights computed for real")
    # weights = weights.numpy()
    # tf.print("Weights converted to numpy")

    return tf.cast(weights, tf.float32)

def multinomial_crossentropy_loss(Z, Z_hat, batch_size=32):
    Q = len(get_ab_to_q_dict())
    p = pickle.load(open("p_5000.p", "rb"))
    p_tilde = gaussian_filter(p, sigma=5)
    # tf.print("P_tilde computed!!!")
    eps = 0.0001
    # weights = tf.py_function(get_loss_weights, [Z, Q, p_tilde], tf.float32)
    weights = get_loss_weights(Z, Q, p_tilde)
    # tf.print("Weights computed!!!")
    log = tf.math.log(Z_hat + eps)
    # tf.print("Log computed!!!")
    mul = log * Z
    summ = tf.reduce_sum(mul, 1) 
    loss = -tf.reduce_sum(weights * summ) / batch_size
    # tf.print("Loss computed!!!")

    return loss

# ---------------------------- Model ----------------------------

l2_reg = l2(1e-3)

def ColourNet():
    input_tensor = Input(shape=(256, 256, 1))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_1', 
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(input_tensor)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='conv1_2', 
            kernel_initializer="he_normal", kernel_regularizer=l2_reg, strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_1', 
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv2_2', 
            kernel_initializer="he_normal", kernel_regularizer=l2_reg, strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_1',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_2',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv3_3', 
            kernel_initializer="he_normal", strides=(2, 2))(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_1',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_2',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='conv4_3',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv5_1',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv5_2',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv5_3',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    x = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv6_1',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv6_2',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', dilation_rate=2, name='conv6_3',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv7_1',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv7_2',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='conv7_3',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    x = UpSampling2D(size=(2, 2))(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv8_1',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv8_2',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='conv8_3',
            kernel_initializer="he_normal", kernel_regularizer=l2_reg)(x)
    x = BatchNormalization()(x)

    x = Conv2D(274, (1, 1), activation='softmax', padding='same', name='pred')(x)
    outputs = UpSampling2D(size=(4, 4), interpolation='bilinear')(x)
    model = Model(inputs=input_tensor, outputs=outputs, name="ColorNet")
    return model



# ---------------------------- Metrics ----------------------------

def mse(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))

def mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred))

def psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))
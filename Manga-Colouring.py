import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard #type: ignore

from dataset_utils import create_dataset
from model_utils import *

train_ds, train_paths, val_ds, val_paths = create_dataset('images', batch_size=64)

model = ColourNet()

sgd = tf.keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True, clipnorm=5.)

model.compile(
    optimizer=sgd,
    loss=multinomial_crossentropy_loss,
    metrics=[
        'accuracy', mse, mae, psnr],
    run_eagerly=True
)

early_stopping_callback = EarlyStopping(
    monitor='loss',
    patience=30,
    mode='min',
    restore_best_weights=True
)


reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='loss', 
    factor=0.05, 
    patience=5, 
    verbose=1, 
    # min_lr=0.00001,
    mode='min'
)

tensorboard_callback = TensorBoard(
    log_dir='./logs', 
    histogram_freq=0, 
    write_graph=True, 
    write_images=True
)


history = model.fit(
    train_ds,
    epochs=1000,
    callbacks=[reduce_lr_callback]
)


model.save("model-best.keras")


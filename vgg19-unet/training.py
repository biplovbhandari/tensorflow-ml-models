# -*- coding: utf-8 -*-

import glob
import numpy as np
from pathlib import Path
import sh
import tensorflow as tf
from model import dataio, model


# specify directory as data io info
BASEDIR = Path('/Users/biplovbhandari/Works/SIG/hydrafloods/tf-vgg19-unet')
DATADIR = BASEDIR / 'data'
TRAINING_DIR = BASEDIR / 'training'
TESTING_DIR = BASEDIR / 'testing'
VALIDATION_DIR = BASEDIR / 'validation'

# specify some data structure
FEATURES = ['VH', 'VV']
LABELS = ['class']

# patch size for training
KERNEL_SIZE = 256
PATCH_SHAPE = (KERNEL_SIZE, KERNEL_SIZE)

# Sizes of the training and evaluation datasets.
# based on sizes of exported data and spliting performed earlier
# ~13542 samples
# ~70% are training, ~20% are testing, ~10% are validation
TRAIN_SIZE = 9480
TEST_SIZE = 2709
VAL_SIZE = 1354

# Specify model training parameters.
BATCH_SIZE = 32
EPOCHS = 20
BUFFER_SIZE = 3000

# get list of files for training, testing and eval
if TRAINING_DIR.exists() and TESTING_DIR.exists() and VALIDATION_DIR.exists():
    training_files = glob.glob(str(TRAINING_DIR) + '/*')
    testing_files = glob.glob(str(TRAINING_DIR) + '/*')
    validation_files = glob.glob(str(TRAINING_DIR) + '/*')
else:
    files = glob.glob(str(DATADIR) + '/*')
    DATASET_SIZE = len(files)
    train_size = int(0.7 * DATASET_SIZE)
    val_size = int(0.2 * DATASET_SIZE)
    test_size = int(0.1 * DATASET_SIZE)

    np.random.shuffle(files)
    training_files = files[:train_size]
    remaining = files[train_size:]
    np.random.shuffle(files)
    testing_files = remaining[:val_size]
    validation_files = remaining[val_size:]

    sh.mkdir(TRAINING_DIR)
    sh.mkdir(TESTING_DIR)
    sh.mkdir(VALIDATION_DIR)

    sh.cp([DATADIR / file for file in training_files], TRAINING_DIR)
    sh.cp([DATADIR / file for file in testing_files], TESTING_DIR)
    sh.cp([DATADIR / file for file in validation_files], VALIDATION_DIR)

# get training, testing, and eval TFRecordDataset
# training is batched, shuffled, transformed, and repeated
training = dataio.get_dataset(training_files, FEATURES, LABELS, PATCH_SHAPE,
                              BATCH_SIZE, buffer_size=BUFFER_SIZE, training=True).repeat()
# testing is batched by 1 and repeated
testing = dataio.get_dataset(testing_files, FEATURES, LABELS, PATCH_SHAPE, 1).repeat()
# eval is batched by 1
eval = dataio.get_dataset(validation_files, FEATURES, LABELS, PATCH_SHAPE, 1)

# get distributed strategy and apply distribute i/o and model build
strategy = tf.distribute.MirroredStrategy()

# define tensor input shape and number of classes
in_shape = PATCH_SHAPE + (len(FEATURES),)
out_classes = len(LABELS)

# build the model and compile
myModel = model.build(in_shape, out_classes, distributed_strategy=strategy)

# define callbacks during training
modelCheckpnt = callbacks.ModelCheckpoint(
    'bestModelWeights.h5', monitor='val_loss', save_best_only=True, mode='min', verbose=1, save_weights_only=True)
earlyStop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, verbose=0, mode='auto', restore_best_weights=True)
tensorboard = callbacks.TensorBoard(log_dir='./logs', write_images=True)

# fit the model
history = myModel.fit(
    x=training,
    epochs=EPOCHS,
    steps_per_epoch=(TRAIN_SIZE // BATCH_SIZE),
    validation_data=testing,
    validation_steps=TEST_SIZE,
    callbacks=[modelCheckpnt, tensorboard, earlyStop],
)

# check how the model trained
myModel.evaluate(eval)


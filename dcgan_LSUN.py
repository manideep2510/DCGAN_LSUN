from __future__ import division, print_function
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import *
from keras.optimizers import Adam
from tqdm import tqdm
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Nadam, Adam, SGD
from keras.metrics import categorical_accuracy, binary_accuracy
from keras.regularizers import l2
from scipy.misc import imresize
import tensorflow as tf
import numpy as np
import time
import random
import os
import sys
import cv2
import glob
from PIL import Image
import matplotlib.pyplot as plt

# To read the images in numerical order
import re
numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

# Initializing all the images into 4d arrays.

'''The images in the dataset are arranged into mupltiple folders of multiple subfolders and we need to read all
the images in all the subfolders into a single array. So, first we willl be reading all the files names including
their paths which are stored as a list named filelist. After that using these paths we read the images as arrays
and make a 4D array'''

a = sorted(glob.glob('/home/manideep/Downloads/lsun_bedroom/sample/data0/lsun/bedroom/*'), key=numericalSort)

b = []
for i in a:
    b.extend(sorted(glob.glob(i+'/*'), key=numericalSort))

c = []
for i in b:
    c.extend(sorted(glob.glob(i+'/*'), key=numericalSort))

filelist = []
for i in c:
    filelist.extend(sorted(glob.glob(i+'/*.jpg'), key=numericalSort))

# Reading the images and making a 4d array for training

X_train = np.array([imresize(np.asarray(Image.open(fname)), (64,64)) for fname in filelist[:1000]])

# Image shape information
img_shape = X_train.shape[1:]
latent_dim = 100

optimizer = Adam(lr=0.0002, beta_1=0.5)

# Generator

def generator():

    model = Sequential()

    model.add(Dense(1024*4*4, input_dim = latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((4,4,1024)))
    model.add(BatchNormalization(momentum=0.5))
    model.add(UpSampling2D())
    model.add(Conv2D(512, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(UpSampling2D())
    # For output layer, use tanh activation and BatchNormalization shouldn't be used as told in the paper
    model.add(Conv2D(3, kernel_size=3, padding="same", activation = 'tanh'))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

# Discriminator

def discriminator():

    model = Sequential()

    model.add(Conv2D(128, kernel_size=3, strides=1, input_shape=img_shape, padding='valid'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding='valid'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Conv2D(512, kernel_size=3, strides=1, padding='valid'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Conv2D(1024, kernel_size=3, strides=1, padding='valid'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.5))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

discriminator = discriminator()
discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Build the generator
generator = generator()

# The generator takes noise as input and generates imgs
z = Input(shape=(latent_dim,))
img = generator(z)

# For the combined model we will only train the generator
discriminator.trainable = False

# The discriminator takes generated images as input and determines validity
valid = discriminator(img)

# The combined model  (stacked generator and discriminator)
# Trains the generator to fool the discriminator
combined = Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=optimizer)

def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig("images/epoch_%d.png" % epoch)
    plt.close()

# Training both generator and discriminator

epochs = 100
batch_size=128
save_interval=5

# Rescale -1 to 1
X_train = X_train / 127.5 - 1.
#X_train = np.expand_dims(X_train, axis=3)

# Adversarial ground truths
valid = np.ones((batch_size, 1))
fake = np.zeros((batch_size, 1))

# Declaring empty lists to save the losses for plotting
d_loss_plot = []
g_loss_plot = []
acc_plot = []

for epoch in range(epochs):
        
    start = time.time()

    #  Training the Discriminator

    # Select a random half of images
    idx = np.random.randint(0, X_train.shape[0], batch_size)
    imgs = X_train[idx]

    # Sample noise and generate a batch of new images
    noise = np.random.normal(0, 1, (batch_size, latent_dim))
    gen_imgs = generator.predict(noise)

    # Train the discriminator (real classified as ones and generated as zeros)
    d_loss_real = discriminator.train_on_batch(imgs, valid)
    d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

    #  Training the Generator

    # Train the generator (wants discriminator to mistake images as real)
    g_loss = combined.train_on_batch(noise, valid)
        
    end = time.time()
    epoch_time = end-start
    
    # Saving the Discriminator and Generator losses and accuracy for plotting
    d_loss_plot.append(d_loss[0])
    g_loss_plot.append(g_loss)
    acc_plot.append(d_loss[1])

    # Plot the progress
    #print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
    print("Epoch %d/%d - %d s - D loss: %f - acc.: %.2f%% - G loss: %f" % (epoch, epochs, epoch_time, d_loss[0], 100*d_loss[1], g_loss))
9
    # If at save interval => save generated image samples
    if epoch % save_interval == 0:
        save_imgs(epoch)

# Ploting losses and accuracy

# Discriminator accuracy plot 
plt.plot(acc_plot)
plt.title('Discriminator accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()

# Loss plots
plt.plot(d_loss_plot)
plt.plot(g_loss_plot)
plt.title('Losses')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['Discriminator', 'Generator'])
plt.show()

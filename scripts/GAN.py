import os
import time
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
from numpy.random import randint
from tensorflow.keras import Input
from numpy import load, zeros, ones
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Conv2D, UpSampling2D, LeakyReLU, Activation
from tensorflow.keras.layers import Concatenate, Dropout, BatchNormalization, LeakyReLU

#credit to https://github.com/DarylFernandes99/Low-light-Image-Enhancement-using-GAN-- I 
#decided to change a lot on both discriminator and generator model structure, but this 
#was the first repo I found that had proof it was possible and I used their starting strucure
#for how to set up and changed/finetuned from there!
def define_discriminator(image_shape):
    """
      Define the discriminator model consisting of downsampling and patch classification
    """
    init = RandomNormal(stddev=0.05)

    in_src_image = Input(shape=image_shape)
    in_target_image = Input(shape=image_shape)

    merged = Concatenate()([in_src_image, in_target_image])

    d = Conv2D(64, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(merged)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(128, (4,4), strides=(2,2), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(256, (4,4), strides=(4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(512, (4,4), padding='same', kernel_initializer=init)(d)
    d = BatchNormalization()(d)
    d = LeakyReLU(alpha=0.2)(d)

    d = Conv2D(1, (4,4), padding='same', kernel_initializer=init)(d)
    patch_out = Activation('sigmoid')(d)

    model = Model([in_src_image, in_target_image], patch_out)

    opt = Adam(lr=0.001)
    model.compile(loss='binary_crossentropy', optimizer=opt, loss_weights=[0.5])
    return model


def define_generator(image_shape = (256, 256, 3)):
    """
    Define the generator model, downsampling and upsampling to created GAN generated image
    """
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    g = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
    g = BatchNormalization()(g, training=True)
    g3 = LeakyReLU(alpha=0.2)(g)
    g = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g3)
    g = BatchNormalization()(g, training=True)
    g2 = LeakyReLU(alpha=0.2)(g)

    g = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(g2)
    g = BatchNormalization()(g, training=True)
    g1 = LeakyReLU(alpha=0.2)(g)


    g = UpSampling2D((4, 4))(g1)
    g = Conv2D(64, (1, 1), kernel_initializer=init)(g)
    g = Concatenate()([g, g2])
    g = BatchNormalization()(g, training=True)
    g = LeakyReLU(alpha=0.2)(g)


    g = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(g)
    g = BatchNormalization()(g, training=True)
    out_image = Activation('tanh')(g)

    model = Model(in_image, out_image)
    return model

def define_gan(g_model, d_model, image_shape):
	in_src = Input(shape=image_shape)
	gen_out = g_model(in_src)
	dis_out = d_model([in_src, gen_out])
	model = Model(in_src, [dis_out, gen_out])

	opt = Adam(lr=0.0001, beta_1=0.5)  
	model.compile(loss=['binary_crossentropy', 'mae'], optimizer=opt, loss_weights=[1,100])
	return model

def load_real_samples(filename):
    """ grab a dark image and its matching light image and return a grouped pair in an array"""
    data = load(filename)
    X1, X2 = data['arr_0'], data['arr_1']
	# scale pixels
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X2, X1]

def generate_real_samples(dataset, n_samples, patch_shape):
    """
    grab n_samples from dataset and return the pair of dark,light images along with label 1 for real image
    """
    trainA, trainB = dataset
    ix = randint(0, trainA.shape[0], n_samples)
    X1, X2 = trainA[ix], trainB[ix]
    y = ones((n_samples, patch_shape, patch_shape, 1))
    return [X1, X2], y

def generate_fake_samples(g_model, samples, patch_shape):
    """
    create fake examples from generator and attach the zero labels as their real labels
    """
    X = g_model.predict(samples)
    y = zeros((len(X), patch_shape, patch_shape, 1))
    return X, y

def train(d_model, g_model, gan_model, dataset, n_epochs=100, n_batch=12):
    # determine the output square shape of the discriminator
    n_patch = d_model.output_shape[1]
    # unpack dataset
    trainA, trainB = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    for i in range(n_steps):
        # select a batch of real samples
        [X_realA, X_realB], y_real = generate_real_samples(dataset, n_batch, n_patch)
        # generate a batch of fake samples
        X_fakeB, y_fake = generate_fake_samples(g_model, X_realA, n_patch)
        # update discriminator for real samples
        d_loss1 = d_model.train_on_batch([X_realA, X_realB], y_real)
        # update discriminator for generated samples
        d_loss2 = d_model.train_on_batch([X_realA, X_fakeB], y_fake)
        # update the generator
        g_loss, _, _ = gan_model.train_on_batch(X_realA, [y_real, X_realB])


def main():
    """
    set up models and dataset, then train model for n_epochs
    """
    dataset = load_real_samples('../data/dataset.npz')
    image_shape = dataset[0].shape[1:]
    d_model = define_discriminator(image_shape)
    g_model = define_generator(image_shape)
    gan_model = define_gan(g_model, d_model, image_shape)

    train(d_model, g_model, gan_model, dataset, n_epochs = 20, n_batch=12)


    
if __name__ == "__main__":
    main()
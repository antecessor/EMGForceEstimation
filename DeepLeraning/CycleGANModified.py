from __future__ import print_function, division

import datetime
import os
import tensorflow as tf

import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Input, Dropout, Concatenate, Lambda, Flatten, GRU, LSTM, Dense, Reshape, Subtract, RepeatVector, Multiply, Activation, TimeDistributed
from keras.layers.advanced_activations import LeakyReLU, ThresholdedReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, UpSampling1D, Conv2DTranspose
from keras.models import Model
from keras.optimizers import Adam, Nadam, RMSprop
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
import keras.backend as K
from sklearn.decomposition import FastICA
from sklearn.preprocessing import scale
from tensorflow import convert_to_tensor, float32


class CycleGANModified:
    def __init__(self, row, col):
        # Input shape

        self.img_rows = row
        self.img_cols = col
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols)

        # Configure data loader
        self.dataset_name = 'EMG2Spikes'

        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2 ** 4)
        self.disc_patch = (256, 1)

        # Number of filters in the first layer of G and D
        self.gf = 6
        self.df = 12

        # Loss weights
        self.lambda_cycle = 4.0  # Cycle-consistency loss
        self.lambda_id = 0.01 * self.lambda_cycle  # Identity loss

        optimizer = Nadam()

        # Build and compile the discriminators
        self.d_A = self.build_discriminator()
        self.d_B = self.build_discriminator()
        self.d_A.compile(loss='mae',
                         optimizer=optimizer,
                         metrics=['accuracy'])
        self.d_B.compile(loss='mae',
                         optimizer=optimizer,
                         metrics=['accuracy'])

        # -------------------------
        # Construct Computational
        #   Graph of Generators
        # -------------------------

        # Build the generators
        self.g_AB = self.build_generator("tanh")
        self.g_BA = self.build_generator("tanh")

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)
        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[valid_A, valid_B,
                                       fake_B, fake_A,
                                       reconstr_A, reconstr_B])
        self.combined.compile(loss=['mse', 'mse',
                                    self.custom_loss(), self.custom_loss(),
                                    'mae', 'mae'],
                              loss_weights=[1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id],
                              optimizer=optimizer)

    # Define custom loss
    def custom_loss(self):

        # Create a loss function that adds the MSE loss to the mean of all squared activations of a specific layer
        def loss(y_true, y_pred):
            return K.mean(K.abs(y_true - y_pred)) - 10 * K.mean(K.log(K.mean(K.greater_equal(y_pred, 0.4),axis=1) / K.mean(K.less_equal(y_pred, 0.4),axis=1)))

        # Return a function
        return loss

    def build_generator(self, outputLayer="relu"):
        """U-Net Generator"""

        def counting(args):
            input = args
            var = tf.reduce_sum(input, axis=3, keepdims=True) / tf.reduce_max(input)
            return var

        def conv2d(layer_input, filters, f_size=3):
            """Layers used during downsampling"""
            d = Conv1D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=3, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling1D(size=2)(layer_input)
            u = Conv1D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)
            u = Concatenate()([u, skip_input])
            return u

        def multiplyMat(x):
            image, mask = x
            # mask = K.expand_dims(mask, axis=-1)  # could be K.stack([mask]*3, axis=-1) too
            CyyInverse = tf.linalg.inv(tf.matmul(image, K.permute_dimensions(mask, [0, 2, 1])) + tf.eye(256) * 10e-4)
            CyyY = tf.matmul(CyyInverse, image)
            return CyyY

        def multiply(x):
            image, mask = x
            # mask = K.expand_dims(mask, axis=-1)  # could be K.stack([mask]*3, axis=-1) too
            return mask * image

        def reshapeData1(x):
            return K.reshape(x, [-1, 256, 12, 1])

        def reshapeData2(x):
            return K.reshape(x, [-1, 256, 12])
            # return K.max(x, axis=2)

        def nonlinearFunction(x):
            return K.log(1 + x * x)

        def add(x):
            sig1, sig2 = x
            return sig1 + sig2

        # Image input
        # Image input
        d0 = Input(shape=self.img_shape)
        d1 = Conv1D(self.img_shape[1], kernel_size=5, padding='same', activation="relu")(d0)
        dFunc = Lambda(nonlinearFunction)(d0)
        Rxt = Lambda(add)([dFunc, d1])
        RxtConv = Conv1D(self.img_shape[1], kernel_size=5, padding='same', activation="relu")(Rxt)

        dCorrelation = Lambda(multiplyMat)([d0, d0])
        firing = Lambda(multiplyMat)([RxtConv, dCorrelation])
        out = Conv1D(self.img_shape[1], kernel_size=5, padding='same', activation="softmax")(firing)

        return Model(d0, out)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv1D(filters, kernel_size=f_size, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if normalization:
                d = InstanceNormalization()(d)
            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.img_shape[1])
        d3 = d_layer(d2, 4)

        validity = Conv1D(1, kernel_size=4, padding='same')(d3)

        return Model(img, validity)

    def train(self, x_train, y_train, epochs, batch_size=1, sample_interval=20):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.disc_patch)
        fake = np.zeros((batch_size,) + self.disc_patch)

        for epoch in range(epochs):
            for batch_i, imgs_A in enumerate(x_train):
                imgs_B = y_train[batch_i]
                imgs_B = np.reshape(imgs_B, (-1, x_train.shape[1], x_train.shape[2]))
                imgs_A = np.reshape(imgs_A, (-1, x_train.shape[1], x_train.shape[2]))
                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Translate images to opposite domain
                fake_B = self.g_AB.predict(imgs_A)
                fake_A = self.g_BA.predict(imgs_B)

                # Train the discriminators (original images = real / translated = Fake)
                dA_loss_real = self.d_A.train_on_batch(imgs_A, valid)
                dA_loss_fake = self.d_A.train_on_batch(fake_A, fake)
                dA_loss = 0.5 * np.add(dA_loss_real, dA_loss_fake)

                dB_loss_real = self.d_B.train_on_batch(imgs_B, valid)
                dB_loss_fake = self.d_B.train_on_batch(fake_B, fake)
                dB_loss = 0.5 * np.add(dB_loss_real, dB_loss_fake)

                # Total disciminator loss
                d_loss = 0.5 * np.add(dA_loss, dB_loss)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators
                g_loss = self.combined.train_on_batch([imgs_A, imgs_B],
                                                      [valid, valid,
                                                       imgs_B, imgs_A,
                                                       imgs_A, imgs_B])

                elapsed_time = datetime.datetime.now() - start_time

                # Plot the progress
                print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %05f, adv: %05f, recon: %05f, id: %05f] time: %s " \
                      % (epoch, epochs,
                         batch_i, 1,
                         d_loss[0], 100 * d_loss[1],
                         g_loss[0],
                         np.mean(g_loss[1:3]),
                         np.mean(g_loss[3:5]),
                         np.mean(g_loss[5:6]),
                         elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i, imgs_A, imgs_B)
        self.g_AB.save("EMG2Spike.h5", overwrite=True)
        self.g_BA.save("Spike2EMG.h5", overwrite=True)

    def sample_images(self, epoch, batch_i, imgs_A, imgs_B):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 2, 3

        # Demo (for GIF)
        # imgs_A = self.data_loader.load_img('datasets/apple2orange/testA/n07740461_1541.jpg')
        # imgs_B = self.data_loader.load_img('datasets/apple2orange/testB/n07749192_4241.jpg')

        # Translate images to the other domain
        fake_B = self.g_AB.predict(imgs_A)
        fake_A = self.g_BA.predict(imgs_B)
        # Translate back to original domain
        reconstr_A = self.g_BA.predict(fake_B)
        reconstr_B = self.g_AB.predict(fake_A)

        gen_imgs = np.concatenate([imgs_A, fake_B, reconstr_A, imgs_B, fake_A, reconstr_B])

        # Rescale images 0 - 1
        # gen_imgs = 0.5 * gen_imgs + 0.5

        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                for bias in range(12):
                    if i == 0 and j == 1:
                        gen_imgs[cnt][:, bias] = np.abs(gen_imgs[cnt][:, bias])
                    if np.max(gen_imgs[cnt][:, bias]) != 0:
                        gen_imgs[cnt][:, bias] = gen_imgs[cnt][:, bias] / np.max(gen_imgs[cnt][:, bias]) + bias
                    axs[i, j].plot(gen_imgs[cnt][:, bias] + bias)
                axs[i, j].set_title(titles[j])
                cnt += 1
        fig.savefig("images/%s/%d_%d.png" % (self.dataset_name, epoch, batch_i))
        plt.close()

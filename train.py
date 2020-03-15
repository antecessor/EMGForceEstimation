import argparse
import numpy as np
import os
import json
import glob
import cv2
import time
import datetime
import random

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
import generator
import discriminator

FLAGS = tf.app.flags.FLAGS


def do_preprocessing(input, height, width):
    input = tf.image.resize_images(input, [height, width])
    input = tf.cast(input, tf.float32)
    input = tf.subtract(input, tf.constant(128.0))
    input = tf.divide(input, tf.constant(128.0))
    return input


def get_train_val_batch(train_list, val_list):
    train_name = ops.convert_to_tensor(train_list, dtype=dtypes.string)
    val_name = ops.convert_to_tensor(val_list, dtype=dtypes.string)

    train_queue = tf.train.slice_input_producer([train_name], shuffle=True)
    val_queue = tf.train.slice_input_producer([val_name], shuffle=True)

    # training data
    file_content = tf.read_file(train_queue[0])
    train_image = tf.image.decode_jpeg(file_content, channels=3)
    train_image = do_preprocessing(train_image, FLAGS.load_size, FLAGS.load_size * 2)

    # split input and gt
    if FLAGS.which_direction == 'AtoB':
        train_input_image = train_image[:, :FLAGS.load_size, :]
        train_gt_image = train_image[:, FLAGS.load_size:, :]
    else:
        train_input_image = train_image[:, FLAGS.load_size:, :]
        train_gt_image = train_image[:, :FLAGS.load_size, :]

    # random crop and flip
    concat_image = tf.concat([train_input_image, train_gt_image], 2)
    concat_image = tf.random_crop(concat_image, [FLAGS.fine_size, FLAGS.fine_size, 6])
    concat_image = tf.image.random_flip_left_right(concat_image)
    train_input_image = concat_image[:, :, :3]
    train_gt_image = concat_image[:, :, 3:]
    train_input_image.set_shape([FLAGS.fine_size, FLAGS.fine_size, 3])
    train_gt_image.set_shape([FLAGS.fine_size, FLAGS.fine_size, 3])

    # validation data
    file_content = tf.read_file(val_queue[0])
    val_image = tf.image.decode_jpeg(file_content, channels=3)
    val_image = do_preprocessing(val_image, FLAGS.fine_size, FLAGS.fine_size * 2)

    # split input and gt
    if FLAGS.which_direction == 'AtoB':
        val_input_image = val_image[:, :FLAGS.fine_size, :]
        val_gt_image = val_image[:, FLAGS.fine_size:, :]
    else:
        val_input_image = val_image[:, FLAGS.fine_size:, :]
        val_gt_image = val_image[:, :FLAGS.fine_size, :]

    val_input_image.set_shape([FLAGS.fine_size, FLAGS.fine_size, 3])
    val_gt_image.set_shape([FLAGS.fine_size, FLAGS.fine_size, 3])

    # queue
    min_after_dequeue = 100
    capacity = min_after_dequeue + 4 * FLAGS.batch_size

    train_images = tf.train.shuffle_batch(
        [train_input_image, train_gt_image],
        batch_size=FLAGS.batch_size
        , num_threads=4
        , capacity=capacity
        , min_after_dequeue=min_after_dequeue
    )

    val_images = tf.train.batch(
        [val_input_image, val_gt_image],
        batch_size=FLAGS.batch_size
        # ,num_threads=1
    )

    return train_images, val_images


def train(argv=None):
    with tf.device("/%s:0" % (FLAGS.device)):
        # Generator
        input_image = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.fine_size, FLAGS.fine_size, 3])
        gt_image = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.fine_size, FLAGS.fine_size, 3])
        generated_image = generator.get_inference(input_image, FLAGS, reuse=False, drop_prob=0.5)

        ###### Discriminator
        # real
        real_AB = tf.concat([input_image, gt_image], 3)
        real_logits = discriminator.get_inference(real_AB, FLAGS, reuse=False)
        loss_d_real = discriminator.get_softmax_loss(logits=real_logits, labels=tf.ones_like(real_logits), FLAGS=FLAGS)

        # fake
        fake_AB = tf.concat([input_image, generated_image], 3)
        fake_logits = discriminator.get_inference(fake_AB, FLAGS, reuse=True)
        loss_d_fake = discriminator.get_softmax_loss(logits=fake_logits, labels=tf.zeros_like(fake_logits), FLAGS=FLAGS)

        # discriminator loss
        loss_d = (loss_d_real + loss_d_fake) / 2

        # generator loss
        loss_g_fake = discriminator.get_softmax_loss(logits=fake_logits, labels=tf.ones_like(fake_logits), FLAGS=FLAGS)
        loss_l1_train = generator.get_l1_loss(generated_image, gt_image, FLAGS)
        loss_g = loss_g_fake + FLAGS.lambda_ * loss_l1_train
        # loss_g =  loss_g_fake

        # split variables
        all_vars = tf.trainable_variables()
        vars_d = [k for k in all_vars if "discriminator" in k.name]
        vars_g = [k for k in all_vars if "generator" in k.name]

        # optimizer
        train_d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss_d, var_list=vars_d)
        train_g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(loss_g, var_list=vars_g)

        # validation
        val_input_image = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.fine_size, FLAGS.fine_size, 3])
        val_gt_image = tf.placeholder(tf.float32, [FLAGS.batch_size, FLAGS.fine_size, FLAGS.fine_size, 3])
        val_generated_image = generator.get_inference(val_input_image, FLAGS, reuse=True, drop_prob=0.5, is_train=False)

    # add summary
    sum_l1 = tf.summary.scalar('train l1 loss', loss_l1_train)
    sum_d_real = tf.summary.scalar('loss_d_real', loss_d_real)
    sum_d_fake = tf.summary.scalar('loss_d_fake', loss_d_fake)
    sum_g_fake = tf.summary.scalar('loss_g_fake', loss_g_fake)
    sum_i_train = tf.summary.image('train_images', tf.concat([input_image, gt_image, generated_image], 2),
                                   max_outputs=5)
    sum_i_val = tf.summary.image('validation_images',
                                 tf.concat([val_input_image, val_gt_image, val_generated_image], 2), max_outputs=5)

    # Build the summary operation based on the TF collection of Summaries.
    train_summary_op = tf.summary.merge([sum_l1, sum_d_real, sum_d_fake, sum_g_fake, sum_i_train])
    val_summary_op = sum_i_val

    #########################
    # get data
    train_list, val_list = get_train_val_list()
    total_train_size = len(train_list)
    total_val_size = len(val_list)
    print
    'train list - ', total_train_size
    print
    'validation list - ', total_val_size
    num_iteration = total_train_size / FLAGS.batch_size
    print
    'total train num iteration - ', num_iteration
    num_val_iteration = total_val_size / FLAGS.batch_size
    print
    'total val num iteration - ', num_val_iteration
    train_images, val_images = get_train_val_batch(train_list, val_list)

    # Create a saver.
    saver = tf.train.Saver(tf.global_variables())

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,
                                          gpu_options=gpu_options)) as sess:
        # initialize the variables
        sess.run(tf.global_variables_initializer())

        # load pretrained model
        if FLAGS.pretrained_model:
            point = tf.train.latest_checkpoint(FLAGS.train_dir + '_' + FLAGS.dataset)
            print
            'check point - ', point
            saver.restore(sess, point)

        # summary
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir + '_' + FLAGS.dataset, sess.graph)

        # initialize the queue threads to start to shovel data
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        count = 0
        # then train discrimnator and generator iteratively
        for i in range(FLAGS.num_epoch):

            for j in range(num_iteration):
                count += 1

                train_info = sess.run(train_images)
                train_input_images = train_info[0]
                train_gt_images = train_info[1]

                # train discriminator & get summary
                _, output_d_loss, output_summary = sess.run([train_d_optim, loss_d, train_summary_op],
                                                            feed_dict={input_image: train_input_images,
                                                                       gt_image: train_gt_images})

                # train generator several times
                for k in range(0, 1):
                    _, output_g_loss, output_fake_logits, output_real_logits = sess.run(
                        [train_g_optim, loss_g, fake_logits, real_logits],
                        feed_dict={input_image: train_input_images, gt_image: train_gt_images})

                ts = time.time()
                st = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
                print
                '%s, epoch[%d] iter[%d], d loss - %f, g loss - %f' % (st, i, j, output_d_loss, output_g_loss)

                summary_writer.add_summary(output_summary, count)

                # validation
                if count % 100 == 0:
                    # test only several images
                    for j in range(0, 10):
                        val_info = sess.run(val_images)
                        val_input_images = val_info[0]
                        val_gt_images = val_info[1]

                        # print test_input_image.shape

                        output_summary = sess.run(val_summary_op, feed_dict={val_input_image: val_input_images,
                                                                             val_gt_image: val_gt_images})
                        summary_writer.add_summary(output_summary, count)

                # Save the model checkpoint periodically.
                if count != 0 and count % 1000 == 0:
                    checkpoint_path = os.path.join(FLAGS.train_dir + '_' + FLAGS.dataset, '%s.ckpt' % (FLAGS.dataset))
                    saver.save(sess, checkpoint_path, global_step=count)

        # stop our queue threads and properly close the session
        coord.request_stop()
        coord.join(threads)
        sess.close()


def get_train_val_list():
    train_image_root = 'datasets/' + FLAGS.dataset + '/train/'
    train_list = glob.glob(train_image_root + '*.jpg')

    val_image_root = 'datasets/' + FLAGS.dataset + '/val/'
    val_list = glob.glob(val_image_root + '*.jpg')

    return train_list, val_list


if __name__ == '__main__':

    tf.app.flags.DEFINE_string('dataset', 'facades', """dataset""")
    tf.app.flags.DEFINE_integer('batch_size', 1, """The batch size to use.""")
    tf.app.flags.DEFINE_integer('load_size', 286, """load size.""")
    tf.app.flags.DEFINE_integer('fine_size', 256, """fine size.""")
    tf.app.flags.DEFINE_integer('num_epoch', 200, """num_epoch.""")
    tf.app.flags.DEFINE_string('which_direction', 'AtoB', """AtoB or BtoA""")
    tf.app.flags.DEFINE_integer('lambda_', 100, """weight for l1 loss""")
    tf.app.flags.DEFINE_string('train_dir', './result', """train_dir.""")
    tf.app.flags.DEFINE_bool('pretrained_model', False, """pretrained model""")
    tf.app.flags.DEFINE_string('device', 'cpu', """device""")

    log_files = glob.glob(FLAGS.train_dir + '_' + FLAGS.dataset + '/events*')
    for f in log_files:
        os.remove(f)

    tf.app.run(main=train)

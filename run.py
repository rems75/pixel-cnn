#!/usr/bin/env python

"""
Runs a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py
"""

import os
import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import plotting

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default='data', help='Location for the dataset')
parser.add_argument('-o', '--model_dir', type=str, default='save', help='Location for parameter checkpoints and samples')
parser.add_argument('-ld', '--log_dir', type=str, default='log', help='Location of logs/Only used for Philly')
parser.add_argument('-d', '--data_set', type=str, default='qbert', help='Can be either qbert|cifar|imagenet')
# model
parser.add_argument('-q', '--nr_resnet', type=int, default=5, help='Number of residual blocks per stage of the model')
parser.add_argument('-n', '--nr_filters', type=int, default=160, help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10, help='Number of logistic components in the mixture. Higher = more flexible model')
parser.add_argument('-z', '--resnet_nonlinearity', type=str, default='concat_elu', help='Which nonlinearity to use in the ResNet layers. One of "concat_elu", "elu", "relu" ')
parser.add_argument('-c', '--class_conditional', dest='class_conditional', action='store_true', help='Condition generative model on labels?')
parser.add_argument('-ed', '--energy_distance', dest='energy_distance', action='store_true', help='use energy distance in place of likelihood')
# optimization
parser.add_argument('-l', '--learning_rate', type=float, default=0.001, help='Base learning rate')
parser.add_argument('-e', '--lr_decay', type=float, default=0.999995, help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-b', '--batch_size', type=int, default=16, help='Batch size during training per GPU')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('-ns', '--num_samples', type=int, default=1, help='How many batches of samples to output.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
args = parser.parse_args()
plotting._print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty plotting._print args

# -----------------------------------------------------------------------------
# fix random seed for reproducibility
rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

# initialize data loaders for train/test splits
if args.data_set == 'imagenet' and args.class_conditional:
    raise("We currently don't have labels for the small imagenet data set")
if args.data_set == 'cifar':
    import data.cifar10_data as cifar10_data
    DataLoader = cifar10_data.DataLoader
elif args.data_set == 'imagenet':
    import data.imagenet_data as imagenet_data
    DataLoader = imagenet_data.DataLoader
elif args.data_set == 'qbert':
    import data.qbert_data as qbert_data
    DataLoader = qbert_data.DataLoader
else:
    raise("unsupported dataset")
data = DataLoader(args.data_dir, 'all', 1, rng=rng, shuffle=False, return_labels=True)
obs_shape = data.get_observation_size() # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

# energy distance or maximum likelihood?
if args.energy_distance:
    loss_fun = nn.energy_distance
else:
    if obs_shape[2] == 1:
        loss_fun = nn.discretized_mix_logistic_loss_greyscale
        sample_fun = nn.sample_from_discretized_mix_logistic_greyscale
        var_per_logistic = 3
    else:
        loss_fun = nn.discretized_mix_logistic_loss
        sample_fun = nn.sample_from_discretized_mix_logistic
        var_per_logistic = 10

# data place holders
xs = tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape)

# if the model is class-conditional we'll set up label placeholders + one-hot encodings 'h' to condition on
if args.class_conditional:
    num_labels = data.get_num_labels()
    y_sample = np.mod(np.arange(args.batch_size), num_labels)
    h_sample = tf.one_hot(tf.Variable(y_sample, trainable=False), num_labels)
    ys = tf.placeholder(tf.int32, shape=(args.batch_size,))
    hs = tf.one_hot(ys, num_labels)
else:
    h_sample = None
    hs = h_sample

# create the model
model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance, 'var_per_logistic': var_per_logistic }
model = tf.make_template('model', model_spec)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))
ema_params = [ema.average(p) for p in all_params]

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    # train
    out = model(xs, hs, ema=None, dropout_p=args.dropout_p, **model_opt)
    loss_gen = loss_fun(tf.stop_gradient(xs), out)

    # gradients
    grads = tf.gradients(loss_gen, all_params, colocate_gradients_with_ops=True)

    # test
    print(xs)
    print(hs)
    print(ema)
    print(model_opt)
    out = model(xs, hs, ema=None, dropout_p=0., **model_opt)
    # out = model(xs, hs, ema=ema, dropout_p=0., **model_opt)
    loss_gen_test = loss_fun(xs, out)

    # sample
    out = model(xs, h_sample, ema=ema, dropout_p=0, **model_opt)
    if args.energy_distance:
        new_x_gen.append(out[0])
    else:
        new_x_gen.append(sample_fun(out, args.nr_logistic_mix))

    # training op
    optimizer = tf.group(nn.adam_updates(all_params, tf.reduce_sum(grads, 0), lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(np.log(2.)*np.prod(obs_shape)*args.batch_size)

# sample from the model
def sample_from_model(sess):
    x_gen = np.zeros((args.batch_size,) + obs_shape, dtype=np.float32)
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(new_x_gen, {xs: x_gen})
            x_gen[:,yi,xi,:] = new_x_gen_np[:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    feed_dict = {xs: x}
    if y is not None:
        feed_dict.update({ys: y})
    return feed_dict

# //////////// perform training //////////////
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
test_bpd = []
lr = args.learning_rate
with tf.Session() as sess:
    begin = time.time()

    # init
    if epoch == 0:
        data.reset()  # rewind the iterator back to 0 to do one full epoch
        ckpt_file = args.model_dir + '/params_' + args.data_set + '.ckpt'
        plotting._print('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)

    # compute likelihood over data
    likelihoods = []
    for d in data:
        feed_dict = make_feed_dict(d)
        l = sess.run(loss_gen_test, feed_dict)
        likelihoods.append(np.exp(-l))
        print(np.exp(-l) * data.get_num_obs())
    plotting._print("Run time = %ds" % (time.time()-begin))

    # # generate samples from the model
    # sample_x = []
    # for i in range(args.num_samples):
    #     sample_x.append(sample_from_model(sess))
    # sample_x = np.concatenate(sample_x,axis=0)
    # img_tile = plotting.img_tile(sample_x[:100], aspect_ratio=1.0, border_color=1.0, stretch=True)
    # img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
    # plotting.plt.savefig(os.path.join(args.model_dir,'%s_sample%d.png' % (args.data_set, epoch)))
    # plotting.plt.close('all')
    # np.savez(os.path.join(args.model_dir,'%s_sample%d.npz' % (args.data_set, epoch)), sample_x)

    # save params
    # np.savez(args.model_dir + '/test_bpd_' + args.data_set + '.npz', test_bpd=np.array(test_bpd))

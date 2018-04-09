#!/usr/bin/env python

"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr_gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python run.py --nr_gpu 4
"""

import os
import sys
import json
import argparse
import time
import pickle

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
parser.add_argument('-u', '--init_batch_size', type=int, default=16, help='How much data to use for data-dependent initialization.')
parser.add_argument('-p', '--dropout_p', type=float, default=0.5, help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-x', '--max_epochs', type=int, default=5000, help='How many epochs to run in total?')
parser.add_argument('-g', '--nr_gpu', type=int, default=2, help='How many GPUs to distribute the training across?')
# evaluation
parser.add_argument('--polyak_decay', type=float, default=0.9995, help='Exponential decay rate of the sum of previous model iterates during Polyak averaging')
parser.add_argument('-ns', '--num_samples', type=int, default=1, help='How many batches of samples to output.')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
# action for count
parser.add_argument('--action', type=int, default=None, help='Action to compute the counts for')
parser.add_argument('--compute_pseudo_counts', action='store_true', help='Compute pseudo counts')
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
data = DataLoader(args.data_dir, 'all', args.batch_size*args.nr_gpu, rng=rng, shuffle=False, return_labels=True, action=args.action)
obs_shape = data.get_observation_size() # e.g. a tuple (32,32,3)
assert len(obs_shape) == 3, 'assumed right now'

# energy distance or maximum likelihood?
if args.energy_distance:
    loss_fun = nn.energy_distance
else:
    if obs_shape[2] == 1:
        loss_fun = lambda x, l: nn.discretized_mix_logistic_loss_greyscale(x,l,sum_all=True)
        sample_fun = nn.sample_from_discretized_mix_logistic_greyscale
        var_per_logistic = 3
    else:
        loss_fun = nn.discretized_mix_logistic_loss
        sample_fun = nn.sample_from_discretized_mix_logistic
        var_per_logistic = 10

# data place holders
x_init = tf.placeholder(tf.float32, shape=(args.init_batch_size,) + obs_shape)
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, ) + obs_shape) for i in range(args.nr_gpu)]
xs_single = tf.placeholder(tf.float32, shape=(1, ) + obs_shape)

# if the model is class-conditional we'll set up label placeholders + one-hot encodings 'h' to condition on
if args.class_conditional:
    num_labels = data.get_num_labels()
    y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
    h_init = tf.one_hot(y_init, num_labels)
    y_sample = np.split(np.mod(np.arange(args.batch_size*args.nr_gpu), num_labels), args.nr_gpu)
    h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
    ys = [tf.placeholder(tf.int32, shape=(args.batch_size,)) for i in range(args.nr_gpu)]
    hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]
    ys_single = tf.placeholder(tf.int32, shape=(1,))
    hs_single = tf.one_hot(ys_single, num_labels)
else:
    h_init = None
    h_sample = [None] * args.nr_gpu
    hs = h_sample
    hs_single = None

# create the model
model_opt = { 'nr_resnet': args.nr_resnet, 'nr_filters': args.nr_filters, 'nr_logistic_mix': args.nr_logistic_mix, 'resnet_nonlinearity': args.resnet_nonlinearity, 'energy_distance': args.energy_distance, 'var_per_logistic': var_per_logistic }
model = tf.make_template('model', model_spec)

# run once for data dependent initialization of parameters
init_pass = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
maintain_averages_op = tf.group(ema.apply(all_params))
ema_params = [ema.average(p) for p in all_params]

# get loss gradients over multiple GPUs + sampling
grads = []
loss_gen = []
loss_gen_test = []
new_x_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # Get loss for each image
        out = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        loss_gen.append(loss_fun(tf.stop_gradient(xs[i]), out))

        # gradients
        grads.append(tf.gradients(loss_gen[i], all_params, colocate_gradients_with_ops=True))
        # test
        out = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(loss_fun(xs[i], out))

        # sample
        out = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
        if args.energy_distance:
            new_x_gen.append(out[0])
        else:
            new_x_gen.append(sample_fun(out, args.nr_logistic_mix))

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        # loss_gen_test[0] += loss_gen_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    current_variables = tf.global_variables()
    optimizer = tf.group(nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)
    adam_variables = list(set(tf.global_variables()) - set(current_variables))
    print(adam_variables)

# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)

# sample from the model
def sample_from_model(sess):
    x_gen = [np.zeros((args.batch_size,) + obs_shape, dtype=np.float32) for i in range(args.nr_gpu)]
    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            new_x_gen_np = sess.run(new_x_gen, {xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver(tf.global_variables())

##### SECOND PASS TO COMPUTE GRADIENTS FOR EACH INPUT RATHER THAN SUMMED
loss_fun_2 = lambda x, l: nn.discretized_mix_logistic_loss_greyscale(x, l, sum_all=False)
sample_fun_2 = nn.sample_from_discretized_mix_logistic_greyscale
var_per_logistic = 3

# get loss gradients over multiple GPUs + sampling
grads_2 = []
loss_gen_2 = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # Get loss for each image
        out = model(xs[i], hs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        loss_gen_2.append(loss_fun_2(tf.stop_gradient(xs[i]), out))

        flat_loss = [loss_gen_2[i][j] for j in range(loss_gen_2[i].shape[0])]
        print(len(flat_loss))

        # gradients
        grads_2.extend([tf.gradients(l, all_params, colocate_gradients_with_ops=True) for l in flat_loss])
        print(len(all_params))
        print(len(grads_2))

loss_fun_3 = lambda x, l: nn.discretized_mix_logistic_loss_greyscale(x, l, sum_all=False)

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    grad_to_be_used = []
    out = model(xs_single, hs_single, ema=ema, dropout_p=0, **model_opt)
    loss_gen_3 = loss_fun_3(tf.stop_gradient(xs_single), out)
    print(loss_gen_3)

    for g in grads_2[0]:
        print(g)
        grad_to_be_used.append(tf.placeholder(dtype=tf.float32, shape=g.shape))
    # training op
    optimizer_2 = tf.group(nn.adam_updates(all_params, grad_to_be_used, lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)
    # TO DO: FIND A GOOD WAY TO UNDO THE UPDATE
    # TO DO: UNDO UPDATE ON ADAM PARAMS
    undo_optimization = None

sys.exit()

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) # input to pixelCNN is scaled from uint8 [0,255] to float in range [-1,1]
    if init:
        feed_dict = {x_init: x}
        if y is not None:
            feed_dict.update({y_init: y})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if y is not None:
            y = np.split(y, args.nr_gpu)
            feed_dict.update({ys[i]: y[i] for i in range(args.nr_gpu)})
    return feed_dict

# //////////// perform training //////////////
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)
test_bpd = []
lr = args.learning_rate
with tf.Session() as sess:
    begin = time.time()

    # init
    data.reset()  # rewind the iterator back to 0 to do one full epoch
    ckpt_file = os.path.join(args.model_dir, 'params_' + args.data_set + '.ckpt')
    plotting._print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)
    initial_weights = all_params.eval(session=sess)
    initial_adam = adam_variables.eval(session=sess)
    plotting._print('starting training')

    # compute likelihood over data
    likelihoods = []
    for d in data:
        feed_dict = make_feed_dict(d)
        l = np.array(sess.run(loss_gen_test, feed_dict))
        l = np.reshape(l,(-1))
        likelihoods.extend(np.exp(0 - l))
        # print(l, np.exp(0 - l))
    plotting._print("Run time = %ds" % (time.time()-begin))
    with open(os.path.join(args.model_dir,"likelihoods_"+str(args.action)+".pkl"), 'wb') as f:
        pickle.dump(likelihoods, f)

    # compute pseudo-counts
    if args.compute_pseudo_counts:
        rhos, rhos_prime, pseudo_counts, pseudo_counts_approx  = [], [], [], []
        for d in data:
            feed_dict = make_feed_dict(d)
            l2 = []
            l, g = np.array(sess.run([loss_gen_test, grads_2], feed_dict))
            for i, gradient_ in enumerate(g):
                _ = sess.run([optimizer, {grad_to_be_used: gradient_}])
                l_2.append(sess.run([loss_gen_3], {xs_single: d[0][i], ys_single: d[1][i]})))
            _ = np.array(sess.run(undo_optimization, {}))
            l, l_2 = np.reshape(l,(-1)), np.array(l_2)
            r, r_2 = np.exp(0 - l), np.exp(0 - l_2)
            rhos.extend(r)
            rhos_prime.extend(r_2)
            pseudo_counts.extend(r * (1 - r_2) / (r_2 - r))
            pseudo_counts_approx.extend(r / (r_2 - r))
        plotting._print("Run time = %ds" % (time.time()-begin))
        with open(os.path.join(args.model_dir,"likelihoods_"+str(args.action)+".pkl"), 'wb') as f:
            pickle.dump(rhos, f)
        with open(os.path.join(args.model_dir,"recoding_"+str(args.action)+".pkl"), 'wb') as f:
            pickle.dump(rhos_prime, f)
        with open(os.path.join(args.model_dir,"pseudo_counts_"+str(args.action)+".pkl"), 'wb') as f:
            pickle.dump(pseudo_counts, f)
        with open(os.path.join(args.model_dir,"pseudo_counts_approx_"+str(args.action)+".pkl"), 'wb') as f:
            pickle.dump(pseudo_counts_approx, f)

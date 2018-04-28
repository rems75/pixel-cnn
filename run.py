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
import sys

import numpy as np
import tensorflow as tf

from pixel_cnn_pp import nn
from pixel_cnn_pp.model import model_spec
from utils import plotting

# -----------------------------------------------------------------------------
parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--data_dir', type=str, default=os.getenv(
    'PT_DATA_DIR', 'data'), help='Location for the dataset')
parser.add_argument('-o', '--model_dir', type=str, default=os.getenv(
    'PT_OUTPUT_DIR', 'save'), help='Location for parameter checkpoints and samples')
parser.add_argument('--epoch', type=int, default=0, help='Model epoch to load from')
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
parser.add_argument('-e', '--lr_decay', type=float, default=1.0, help='Learning rate decay, applied every step of the optimization')
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
data_single = DataLoader(args.data_dir, 'all', args.nr_gpu, rng=rng, shuffle=False, return_labels=True, action=args.action)
actions_counts = dict(zip(*data.get_stat_labels()))
num_actions = data.original_labels.size
print(actions_counts, num_actions, np.sum(actions_counts.values()))
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
xs_single = [tf.placeholder(tf.float32, shape=(1, ) + obs_shape) for i in range(args.nr_gpu)]

# if the model is class-conditional we'll set up label placeholders + one-hot encodings 'h' to condition on
if args.class_conditional:
  num_labels = data.get_num_labels()
  y_init = tf.placeholder(tf.int32, shape=(args.init_batch_size,))
  h_init = tf.one_hot(y_init, num_labels)
  y_sample = np.split(np.mod(np.arange(args.batch_size*args.nr_gpu), num_labels), args.nr_gpu)
  h_sample = [tf.one_hot(tf.Variable(y_sample[i], trainable=False), num_labels) for i in range(args.nr_gpu)]
  ys = [tf.placeholder(tf.int32, shape=(args.batch_size,)) for i in range(args.nr_gpu)]
  hs = [tf.one_hot(ys[i], num_labels) for i in range(args.nr_gpu)]
  ys_single = [tf.placeholder(tf.int32, shape=(1,))
         for i in range(args.nr_gpu)]
  hs_single = [tf.one_hot(ys_single[i], num_labels) for i in range(args.nr_gpu)]
else:
  h_init = None
  h_sample = [None] * args.nr_gpu
  hs = h_sample
  hs_single = [None] * args.nr_gpu

# create the model
model_opt = { 'nr_resnet': args.nr_resnet,
              'nr_filters': args.nr_filters,
              'nr_logistic_mix': args.nr_logistic_mix,
              'resnet_nonlinearity': args.resnet_nonlinearity,
              'energy_distance': args.energy_distance,
              'var_per_logistic': var_per_logistic }
model = tf.make_template('model', model_spec)

# run once for data dependent initialization of parameters
init_pass = model(x_init, h_init, init=True, dropout_p=args.dropout_p, **model_opt)

# keep track of moving average
all_params = tf.trainable_variables()
# ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
# maintain_averages_op = tf.group(ema.apply(all_params))
# ema_params = [ema.average(p) for p in all_params]

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
    # out = model(xs[i], hs[i], ema=ema, dropout_p=0., **model_opt)
    out = model(xs[i], hs[i], ema=None, dropout_p=0., **model_opt)
    loss_gen_test.append(loss_fun(xs[i], out))

    # sample
    # out = model(xs[i], h_sample[i], ema=ema, dropout_p=0, **model_opt)
    out = model(xs[i], h_sample[i], ema=None, dropout_p=0, **model_opt)
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
  param_updates, rmsprop_updates = nn.rmsprop_updates(
      all_params, grads[0], lr=tf_lr, mom=0.9, dec=0.95, eps=1.0e-4)
  # optimizer = tf.group(*(param_updates+rmsprop_updates), maintain_averages_op)
  optimizer = tf.group(*(param_updates+rmsprop_updates))
  rmsprop_variables = list(set(tf.global_variables()) - set(current_variables))

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

# save
original_variables = tf.global_variables()
saver = tf.train.Saver(original_variables)

##### SECOND PASS TO COMPUTE GRADIENTS FOR EACH INPUT RATHER THAN SUMMED
loss_fun_2 = lambda x, l: nn.discretized_mix_logistic_loss_greyscale(x, l, sum_all=False)
sample_fun_2 = nn.sample_from_discretized_mix_logistic_greyscale

trainable_params = [all_params]
trainable_params[0].sort(key=lambda v: v.name)
rmsprop_variables.sort(key=lambda v: v.name)
all_models = [model]

# get loss gradients over multiple GPUs + sampling
grads_2, loss_gen_2, loss_test, optimizer_2, reset_variables, resetter = [], [], [], [], [], []

for i in range(args.nr_gpu):
  with tf.device('/gpu:%d' % i):

    if i > 0:
      current_trainable_variables = set(tf.trainable_variables())
      all_models.append(tf.make_template('model_{}'.format(i), model_spec))
      init_pass = all_models[i](x_init, h_init, init=True,
              dropout_p=args.dropout_p, **model_opt)
      trainable_params.append(list(set(tf.trainable_variables()) - current_trainable_variables))
      trainable_params[i].sort(key=lambda v: v.name)
      # ema = tf.train.ExponentialMovingAverage(decay=args.polyak_decay)
      # maintain_averages_op = tf.group(ema.apply(trainable_params[i]))

    # Get loss for each image
    out = all_models[i](xs_single[i], hs_single[i], ema=None, dropout_p=args.dropout_p, **model_opt)
    loss_gen_2.append(loss_fun_2(tf.stop_gradient(xs_single[i]), out))

    # Get loss for each image
    out = all_models[i](xs_single[i], hs_single[i], ema=None, dropout_p=0, **model_opt)
    loss_test.append(loss_fun_2(xs_single[i], out))

    # gradients
    grads_2.append(tf.gradients(loss_gen_2[i], trainable_params[i], colocate_gradients_with_ops=True))

    # training op
    param_updates_2, _ = nn.rmsprop_updates(
      trainable_params[i], grads_2[i], init_rmsp=rmsprop_variables,
      lr=tf_lr, mom=0.9, dec=0.95, eps=1.0e-4)
    optimizer_2.append(tf.group(*param_updates_2))

    # create placeholders to reset the weights of the networks
    reset_variables.append([])
    # for p_0, p in zip(trainable_params[0], trainable_params[i]):
    #   v = tf.get_variable(p.name.split(':')[0]+"_reset_"+str(i), initializer=p_0)
    #   reset_variables[i].append(v)
    for p in trainable_params[i]:
      v = tf.get_variable(p.name.split(
          ':')[0]+"_reset_"+str(i), shape=p.shape, initializer=tf.zeros_initializer)
      reset_variables[i].append(v)

    # create ops to reset the weights of the networks
    reset = []
    for v, p in zip(reset_variables[i], trainable_params[i]):
      reset.append(p.assign(v))

    resetter.append(tf.group(*reset))

# init
initializer = tf.variables_initializer(
    list(set(tf.global_variables()) - set(original_variables)),
    name='init'
)

# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False, single=False):
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
    if single:
      feed_dict = {xs_single[i]: x[i] for i in range(args.nr_gpu)}
      if y is not None:
        y = np.split(y, args.nr_gpu)
        feed_dict.update({ys_single[i]: y[i] for i in range(args.nr_gpu)})
    else:
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
  ckpt_file = os.path.join(args.model_dir,'{}_params_{}.cpkt'.format(args.data_set, args.epoch))
  plotting._print('restoring parameters from', ckpt_file)
  saver.restore(sess, ckpt_file)
  plotting._print('initializing parameters')
  sess.run(initializer)
  plotting._print('creating reset operation')
  for i, p in enumerate(trainable_params[0]):
    init_p = p.eval(session=sess)
    print
    print(p.name)
    for r_v in reset_variables:
      print(r_v.name)
      sess.run(r_v[i].assign(init_p))
  sess.run(resetter)
  plotting._print("Run time for preparation = %ds" % (time.time()-begin))
  plotting._print('starting training')
  begin = time.time()

  print(trainable_params[0][0].eval(session=sess))
  print(trainable_params[1][0].eval(session=sess))
  sys.exit()

  # compute likelihood over data
  log_likelihoods = []
  for d in data:
    feed_dict = make_feed_dict(d)
    l = np.array(sess.run(loss_gen_test, feed_dict))
    log_likelihoods.extend(np.reshape(l,(-1)))
  plotting._print("Run time for likelihoods = %ds" % (time.time()-begin))
  begin = time.time()
  log_likelihoods = np.array(log_likelihoods)
  with open(os.path.join(args.model_dir,"log_likelihoods_epoch_{}_action_{}.pkl".format(args.epoch, args.action)), 'wb') as f:
    pickle.dump(log_likelihoods, f)

  # compute pseudo-counts
  if args.compute_pseudo_counts:
    recoding_log_likelihoods, data_points = [], 0
    for d in data_single:
      feed_dict = make_feed_dict(d, single=True)
      feed_dict.update({tf_lr: lr})
      _ = sess.run(optimizer_2, feed_dict)
      l_2 = sess.run(loss_test, feed_dict)
      # Undo update
      sess.run(resetter)
      recoding_log_likelihoods.extend(l_2)
      data_points += args.nr_gpu
      if data_points % 10000 == 0:
        plotting._print("  Run time for %d points = %ds" % (data_points, time.time()-begin))

    plotting._print("Run time for recoding = %ds" % (time.time()-begin))
    recoding_log_likelihoods = np.array(recoding_log_likelihoods)
    with open(os.path.join(args.model_dir, "recoding_epoch_{}_action_{}.pkl".format(args.epoch, args.action)), 'wb') as f:
      pickle.dump(recoding_log_likelihoods, f)
    pseudo_counts, pseudo_counts_approx = [], []
    if args.action is not None:
      log_likelihoods -= np.log(actions_counts[args.action] / num_actions)
      recoding_log_likelihoods -= np.log((actions_counts[args.action] + 1) / (num_actions + 1))
      # true_likelihood = np.exp(0 - log_likelihoods) * actions_counts[args.action] / num_actions
      # true_recoding_likelihood = np.exp(0 - recoding_log_likelihoods) * (actions_counts[args.action] + 1) / (num_actions + 1)

      # pseudo_counts = true_likelihood * (1 - true_recoding_likelihood) / (true_recoding_likelihood - true_likelihood)
      pg = np.max(- recoding_log_likelihoods + log_likelihoods, 0)
      pseudo_counts_approx = 1 / (np.exp(0.1 * pg / np.sqrt(num_actions)) - 1)

      # with open(os.path.join(args.model_dir,"pseudo_counts_{}_action_{}.pkl".format(args.epoch, args.action)), 'wb') as f:
      #   pickle.dump(pseudo_counts, f)
      with open(os.path.join(args.model_dir,"pseudo_counts_approx_{}_action_{}.pkl".format(args.epoch, args.action)), 'wb') as f:
        pickle.dump(pseudo_counts_approx, f)

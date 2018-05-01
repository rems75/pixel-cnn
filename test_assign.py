
import os
import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

with tf.device('/gpu:0'):

  t = tf.Variable(tf.zeros([6, 120]), name='yo')
  t_2 = tf.Variable(tf.zeros([6, 120]), name='yooo')
  sum = t + t_2

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

  print("Initializing")
  sess.run(tf.global_variables_initializer())
  begin = time.time()
  print("Assigning")
  sess.run(t_2.assign(t))
  print("Assign in {} seconds".format(time.time() - begin))
  t_3 = sess.run(sum)
  print(t, t_2, t_3)
  # print(t_3)

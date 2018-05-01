
import os
import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

with tf.device('/gpu:0'):

  t = tf.Variable([6, 120], name='yo')
  t_2 = tf.Variable([6, 120], name='yooo')

with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:

  print("Initializing")
  sess.run(tf.global_variables_initializer())
  begin = time.time()
  print("Assigning")
  sess.run(t_2.assign(t))
  print("Assign in {} seconds".format(time.time() - begin))

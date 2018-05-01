
import os
import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

with tf.device('/gpu:0'):

  t = tf.Variable([6, 120], name='yo')
  t_2 = tf.Variable([6, 120], name='yo_1')

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:

  sess.run(tf.global_variables_initializer())
  begin = time.time()
  sess.run(t_2.assign(t))
  print("Assign in {} seconds".format(time.time() - begin))

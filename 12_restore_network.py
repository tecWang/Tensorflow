import tensorflow as tf
import numpy as np

# restore variables from file
w = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32, name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32, name='biases')

# do not need to init variables

saver = tf.train.Saver() 
with tf.Session() as sess:
    saver.restore(sess, "./network/save_net.ckpt")
    print("weights:", sess.run(w), "\nbiases:", sess.run(b))
    # weights: [[1. 2. 3.]
    #           [3. 4. 5.]]
    # biases: [[1. 2. 3.]]
# CNN -- Convolutional Nerual Network
    # padding -- valid padding && same padding
    # pooling -- max pooling && average pooling

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True)

# accuracay
def compute_accruacy_and_print(v_xs, v_ys):
    global prediction
    y_prediction = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    # argmax : returns the index with the largest value across axes of a tensor. (deprecated arguments)
    correct_prediction = tf.equal(tf.argmax(y_prediction, 1), tf.argmax(v_ys, 1))
    # cast : Casts a tensor to a new type.
    accuracay = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    result = sess.run(accuracay, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

# CNN define Weights ,biases, convolution, pool
def variable_weight(shape):
    # truncate: describes a geometric figure that has the apex or an end removed and replaced with a plane section, often parallel to the base
    # truncated_normal: Outputs random values from a truncated normal distribution.
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def variable_biases(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, Weights):
    # strides rules : 1. [1, x_movement, y_movement, 1]
    #                 2. strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, Weights, strides=[1, 1, 1, 1], padding="SAME")

def max_pool_2_2(x):
    # x_movement = y_movement = 2, so the image can be pressed to smaller image
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding="SAME")

# define placeholder
# Here we assign it a shape of [None, 784],
# where 784 is the dimensionality of a single flattened 28 by 28 pixel MNIST image, 
# and None indicates that the first dimension, corresponding to the batch size, 
# can be of any size
xs = tf.placeholder(tf.float32, [None, 784])/255
ys = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
x_image = tf.reshape(xs, [-1, 28, 28, 1])
# print(x_image.shape) # [n_samples, 28, 28 ,1]


# convolution layer1
#   [5 5 1 32], 1 是输入图片的通道数，
#       1. patch 5*5
#       2. channel 1
#               gray image 1 channel
#               multi-color image: 3 channeles(RGB)
#       3. 32 outputs channels
W_conl1 = variable_weight([5,5,1,32])
b_conv1 = variable_biases([32])
hidden_layer1 = tf.nn.relu(conv2d(x_image, W_conl1) + b_conv1)  
# hidden_layer1 outputs: 28*28*32, because strides = 'SAME'
hidden_pool1 = max_pool_2_2(hidden_layer1)  
# hidden_pool1 outputs: 14*14*32, because step = 2 


# convolution layer2
W_conl2 = variable_weight([5,5,32,64])
b_conv2 = variable_biases([64])
hidden_layer2 = tf.nn.relu(conv2d(hidden_pool1, W_conl2) + b_conv2) 
# hidden_layer1 outputs: 14*14*64
hidden_pool2 = max_pool_2_2(hidden_layer2) 
# hidden_pool1 outputs: 7*7*64


# function layer1
W_func1 = variable_weight([7*7*64,1024])
b_func1 = variable_biases([1024])
hidden_pool2_flat = tf.reshape(hidden_pool2, [-1,7*7*64])
# [n_samples, 7 ,7, 64] --> [n_samples, 7*7*64]
hidden_func1 = tf.nn.relu(tf.matmul(hidden_pool2_flat, W_func1) + b_func1)
hidden_func1_dropout = tf.nn.dropout(hidden_func1, keep_prob)

# function layer2 --> pred iction layer?
W_func2 = variable_weight([1024, 10])
b_func2 = variable_biases([10])
prediction = tf.nn.softmax(tf.matmul(hidden_func1_dropout, W_func2) + b_func2)


# train 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
    if i % 50 == 0:
        print(compute_accruacy_and_print(mnist.test.images[:1000], mnist.test.labels[:1000]))
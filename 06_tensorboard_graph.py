import tensorflow as tf
import numpy as np

# define a function to add a layer into network
def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('layer'):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):    
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
        # biases = tf.Variable(tf.zeros(1, out_size) + 0.1) 
        # TypeError: unsupported operand type(s) for +: 'Tensor' and 'float'
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)

        # activation_function
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')


# add hidden layers
lay1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(lay1, 10, 1, activation_function=None)

# error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys), reduction_indices=[1]))
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# train
init = tf.global_variables_initializer()
sess = tf.Session()



# writer = tf.train.SummaryWriter("./logs/", sess.graph)    
# # SummaryWriter has changed in TensorFlow 1.0. You can use tf.summary.FileWriter
writer = tf.summary.FileWriter("./logs/", sess.graph)



# important step
sess.run(init)

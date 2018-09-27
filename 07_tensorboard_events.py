import tensorflow as tf
import numpy as np

# define a function to add a layer into network
def add_layer(inputs, in_size, n_layer, out_size, activation_function=None):
    with tf.name_scope('layer'):
        layer_name = n_layer
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
            tf.summary.histogram(layer_name + '/weights', Weights)
        with tf.name_scope('biases'):    
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
            tf.summary.histogram(layer_name + '/biases', biases)  
        # biases = tf.Variable(tf.zeros(1, out_size) + 0.1) 
        # TypeError: unsupported operand type(s) for +: 'Tensor' and 'float'
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            tf.summary.histogram(layer_name + '/Wx_plus_b', Wx_plus_b) 
        # activation_function
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
        tf.summary.histogram(layer_name + '/outputs', outputs) 
        return outputs


with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# create data
x_data = np.linspace(-1, 1, 1000)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add hidden layers
lay1 = add_layer(xs, 1, 'layer1',10, activation_function=tf.nn.relu)
prediction = add_layer(lay1, 10, 'layer_pre',  1, activation_function=None)

# error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(prediction - ys), reduction_indices=[1]))
    # use different summary method record result
    tf.summary.scalar('loss', loss)
with tf.name_scope('train_step'):
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

# train
init = tf.global_variables_initializer()
sess = tf.Session()
mergerd = tf.summary.merge_all()
writer = tf.summary.FileWriter("./logs/", sess.graph)

# important step
sess.run(init)

for i in range(10000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        result = sess.run(mergerd, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, i)
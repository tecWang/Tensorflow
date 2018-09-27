# RNN -- Recurrent Nerual Network
    # LSTM RNN -- Long Short-Term Memory RNN
    # when use RNN?
        # when the data have order
    
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# data
mnist = input_data.read_data_sets("MNIST_data",one_hot=True) 

# hyper parameters
learn_rate = 0.001
iteration_times = 100000
batch_size = 128

n_inputs = 28           # 28 pixels in one line
n_steps = 28            # 28 lines
n_hidden_units = 128    # 
n_classes = 10          #  classifier target size(0-9)

# inputs
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])   # 28*28
y = tf.placeholder(tf.float32, [None, n_classes])           # 10

# weights and biases
weights = {
    # (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}
biases = {
    # (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units,])),
    # (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes,]))
    # shape = [10, ] means it is a noraml one dimension array
}


def RNN(X, weights, biases):
    # hidden layer for input to cell 
    #########################################################
    # X --> (128batch, 28steps, 28inputs)
        # why is the shape of x (128, 28, 28)
        # because the actual size of the batch is 128 when running train_step
    X = tf.reshape(X, [-1, n_inputs])                         
    # X  --> (128*28, 28inputs)
    X_in = tf.matmul(X, weights['in']) + biases['in']
    #  X_in --> (128*28, 128hidden)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])  
    #  X_in --> (128*28, 128hidden)


    # cell 
    ##########################################################
    # lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)    # version < 1.0
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units)
    # lstm is divided into two parts (c_state, m_state) <-- tuple
    _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)


    # hidden layer for output as the final result
    ##########################################################
    # method1
    result = tf.matmul(states[1], weights['out']) + biases['out']

    # method2
    # outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    # result = tf.matmul(outputs[-1], weights['out']) + biases['out']
    # state[1] = outputs[-1] 
    # outputs[-1] is the last output of outputs, because RNN is sequential 

    return result

prediction = RNN(x, weights, biases)    # such as 9
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# here, because the optimization method is wrong, is has not been able to improve the training success rate until I change the methods to AdamOptimizer
# train_step = tf.train.AdadeltaOptimizer(learn_rate).minimize(cost)
train_step = tf.train.AdamOptimizer(learn_rate).minimize(cost)


correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
with tf.Session() as sess: 
    sess.run(init)
    step = 0
    while step * batch_size < iteration_times:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])    # 128 * 28 * 28
        sess.run([train_step], feed_dict={
            x: batch_xs,
            y: batch_ys
        })
        if step % 20 == 0:
            # print("current times: ", step*batch_size)
            print("step*batch_size:", step*batch_size, "\taccuracy: ", sess.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys
            }), "\tcost:", sess.run(cost, feed_dict={x: batch_xs, y:batch_ys}))
            # print("prediction:", sess.run(prediction, feed_dict={x: batch_xs}))
        step += 1
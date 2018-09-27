import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# define a function to add a layer into network
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    # biases = tf.Variable(tf.zeros(1, out_size) + 0.1) 
    # TypeError: unsupported operand type(s) for +: 'Tensor' and 'float'
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# create data
x_data = np.linspace(-1, 1, 1000)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# add layers
lay1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
predition = add_layer(lay1, 10, 1, activation_function=None)

# train
loss = tf.reduce_mean(tf.reduce_sum(tf.square(predition - ys), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data, y_data)
plt.ion()
plt.show()

for step in range(10000):
    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    if step % 50 == 0:
        predition_value = sess.run(predition, feed_dict={xs: x_data, ys: y_data})
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        lines = ax.plot(x_data, predition_value, 'r-', lw=5)
        plt.pause(.1)
        print("loss: ", sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
        # print(  "loss: ", sess.run(loss, feed_dict={xs: x_data, ys: y_data}), "\tpredication: ", sess.run(predition, feed_dict={xs: x_data, ys: y_data}))
#  overfitting ?
#   
#  reason of overfitting
#   1. the amount of data is too low
#   2. too much pursuit of error accuracy
# how to solve this?
#   1. increase the amount of data
#   2. regularization
#       y = W * x
#       L1: cost = (Wx - real y)^2 + abs(W)
#       L2: cost = (Wx - real y)^2 + (W)^2
#       L3: cost = (Wx - real y)^2 + ...
#       L4: cost = (Wx - real y)^2 + ...
#   3. dropout solves overfitting

import tensorflow as tf

from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# add_layer
def add_layer(inputs, inputs_size, outputs_size, layer_name="layer", activation_function=None):
    Weights = tf.Variable(tf.random_normal([inputs_size, outputs_size]))
    biases = tf.Variable(tf.zeros([1, outputs_size]) + 0.1)
    Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    # use dropout solve overfitting
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    tf.summary.histogram(layer_name + '/outputs', outputs )
    return outputs

def compute_accruacy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

# define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])    # 64 = 8 * 8
ys = tf.placeholder(tf.float32, [None, 10])

# add layer
lay1 = add_layer(xs, 64, 50, layer_name="lay_1", activation_function=tf.nn.tanh)
prediction = add_layer(lay1, 50, 10, "lay_prediction", activation_function=tf.nn.softmax)

# loss
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    tf.summary.scalar('loss', cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter("logs/train", sess.graph)
test_writer = tf.summary.FileWriter("logs/test", sess.graph)

# train
sess.run(tf.global_variables_initializer())
for i in range(1000):
    # 0.6 means dropout 40%
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob:0.5})
    if i % 50 == 0:
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob:1})
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob:1})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)



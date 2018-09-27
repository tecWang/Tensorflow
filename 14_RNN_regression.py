import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# params
BATCH_START = 0
TIME_STEPS = 20

BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10

LEARN_RATE = 0.006
# BATCH_START_TEST = 0


def get_batch():
    global BATCH_START, TIME_STEPS
    # xs shape (50batch, 20steps)
    # (0, 0+20*50) --> (50, 20)
    xs = np.arange(BATCH_START, BATCH_START+TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS)) / (10*np.pi)
    seq = np.sin(xs)
    res = np.cos(xs)
    BATCH_START += TIME_STEPS
    # plt.plot(xs[0, :], res[0, :], 'r', xs[0, :], seq[0, :], 'b--')
    # plt.show()
    # returned seq, res and xs: shape (batch, step, input)
    print([seq[:, :, np.newaxis], res[:, :, np.newaxis], xs])
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.cell_size = cell_size
        self.batch_size = batch_size
        with tf.name_scope("inputs"):
            # None corresponds to the value of batch_size
            #   in the file "13_RNN.py", batch_size is 128
            #   in the current file, batch_size is BATCH_SIZE = 50
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name="xs")
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name="ys")
        with tf.variable_scope("in_hidden"):
            self.add_input_layer()
        with tf.variable_scope("LSTM_cell"):
            self.add_cell()
        with tf.variable_scope("out_hidden"):
            self.add_output_layer()
        with tf.variable_scope("cost"):
            self.compute_cost()
        with tf.variable_scope("train"):
            self.train_step = tf.train.AdamOptimizer(LEARN_RATE).minimize(self.cost)

    def add_input_layer(self):
        # (batch, n_step, in_size) --> (batch*n_step, input_size)
        layer_in_x = tf.reshape(self.xs, [-1, self.input_size], name="2_2D")
        # (input_size, cell_size)
        w_in = self._weight_variable([self.input_size, self.cell_size])
        # (cell_size, )
        b_in = self._biases_variable([self.cell_size, ])

        with tf.name_scope("wx_plus_b"):
            layer_in_x = tf.matmul(layer_in_x, w_in) + b_in
            # (batch*n_steps, input_size) --> (batch, n_steps, cell_size)
            self.layer_in_x = tf.reshape(layer_in_x, [-1, self.n_steps, self.cell_size], name="2_3D")

    def add_cell(self):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
        with tf.name_scope("initial_state"):
            self.cell_initial_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)
            self.cell_outputs, self.cell_finally_state = tf.nn.dynamic_rnn(lstm_cell, self.layer_in_x, initial_state=self.cell_initial_state)
                
    def add_output_layer(self):

        layer_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name="2_2D")
        # (input_size, cell_size)
        w_out = self._weight_variable([self.cell_size, self.output_size])
        # (cell_size, )
        b_out = self._biases_variable([self.output_size, ])

        with tf.name_scope("prediction"):
            self.prediction = tf.matmul(layer_out_x, w_out) + b_out
        
    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.prediction, [-1], name='reshape_prediction')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.ms_error,
            name='losses'
        )
        with tf.name_scope('average_cost'):
            self.cost = tf.div(
                tf.reduce_sum(losses, name='losses_sum'),
                self.batch_size,
                name='average_cost')
            tf.summary.scalar('cost', self.cost)

    @staticmethod
    def ms_error(labels, logits):
        return tf.square(tf.subtract(labels, logits))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.,)
        return tf.get_variable(shape=shape, initializer=initializer, name=name)

    def _biases_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

if __name__ == "__main__":
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)
    sess = tf.Session()
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter("./RNN_logs", sess.graph)
    # use tensorboard --logdir="xxx" to view in the explore

    sess.run(tf.global_variables_initializer())
    plt.ion()   # make plt can draw many times
    plt.show()

    for i in range(500):
        seq, res, xs = get_batch()
        if i == 0:
            feed_dict = {
                model.xs: seq,
                model.ys: res
            }
        else:
            feed_dict = {
                model.xs: seq,
                model.ys: res,
                model.cell_initial_state: state
            }

        _, cost, state, prediction = sess.run(
            [model.train_step, model.cost, model.cell_initial_state, model.prediction],
            feed_dict=feed_dict
        )

        # plotting
        plt.plot(xs[0, :], res[0].flatten(), 'r', xs[0, :], prediction.flatten()[:TIME_STEPS], 'b--')
        plt.ylim((-1.2, 1.2))
        plt.draw()
        plt.pause(0.3)

        if i % 20 == 0:
            print('cost: ', round(cost, 4))
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)

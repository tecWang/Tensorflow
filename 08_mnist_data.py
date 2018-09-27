import inspect
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# import 0 to 9 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(inspect.getmembers(mnist.train))
# print("train:", mnist.train.images)
# print("train:", mnist.train.labels)
# next_batch return the next `batch_size` examples from this data set.
print("next_batch1:", mnist.train.next_batch(100))
print("next_batch2:", mnist.train.next_batch(100))
print("next_batch3:", mnist.train.next_batch(100))


print("images:", mnist.test.images)
# images: [[0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]
# ...
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]]
print(mnist.test.images.shape) # (10000, 784)

print("labels:", mnist.test.labels)
# labels: [[0. 0. 0. ... 1. 0. 0.]
# [0. 0. 1. ... 0. 0. 0.]
# [0. 1. 0. ... 0. 0. 0.]
# ...
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]
# [0. 0. 0. ... 0. 0. 0.]]
print(mnist.test.labels.shape)  #(10000, 10)


# print("mnist:", inspect.getmembers(mnist))
import tensorflow as tf

matrix1 = tf.constant([[3, 3]])

matrix2 = tf.constant([ [3],
                        [3]])

product = tf.matmul(matrix1, matrix2)

# ## method1
# sess = tf.Session()
# result = sess.run(product)
# print(result)   # [[18]]
# sess.close()

## methods2 (no need to add sess.close())
with tf.Session() as sess:
    result = sess.run(product)
    print(result)

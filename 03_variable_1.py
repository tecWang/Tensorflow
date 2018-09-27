import tensorflow as tf

# def random_normal(shape,mean=0.0,stddev=1.0,dtype=dtypes.float32,seed=None,name=None): 
arr = tf.Variable(tf.random_normal([3,4]))
# def random_crop(value, size, seed=None, name=None)
arr2 = tf.Variable(tf.random_crop())
print("arr: ", arr) 
# arr:  <tf.Variable 'Variable:0' shape=(3, 4) dtype=float32_ref>
print("arr.shape: ", arr.shape)
# arr.shape:  (3, 4)

sess = tf.Session()
# print("arr: ", sess.run(arr))
# Attempting to use uninitialized value Variable
sess.run(tf.global_variables_initializer())
print("arr: ", sess.run(arr))
# arr:  [[ 0.2514747  -1.2619466   0.38439026 -2.0809827 ]
#        [-0.3374227  -1.0765232  -1.4134455   2.179568  ]
#        [-1.010376   -1.1666309  -0.3938666  -0.09558122]]
import tensorflow as tf

sess = tf.Session()

a = tf.constant([10, 20])
va = sess.run(a)
print("va:", va)              
print("va.shape:", va.shape)   
print("va.len:", len(va))       
# va: [10 20]
# va.shape: (2,)
# va.len: 2
print("------------------------------------------------")



b = tf.constant([[10, 20]])
vb = sess.run(b)
print("vb:", vb)           
print("vb.shape:", vb.shape)   
print("vb.len:", len(vb))     
# vb: [[10 20]]
# vb.shape: (1, 2)
# vb.len: 1
print("------------------------------------------------")



c = tf.constant([   [20],
                    [10]])
vc = sess.run(c)
print("vc:", vc)
print("vc.shape:", vc.shape)    # vc.shape: (2, )
# vc:  [[20]
#       [10]]
# vc.shape: (2, 1)
# vc.len: 2
print("------------------------------------------------")


d = tf.reshape(c, [1, 2])
vd = sess.run(d)
print("vd:", vd)
print("vd.shape:", vd.shape)
# vd: [[20 10]]
# vd.shape: (1, 2)
d = tf.reshape(c, [2,])
vd = sess.run(d)
print("vd:", vd)
print("vd.shape:", vd.shape)
# vd: [20 10]
# vd.shape: (2,)
print('------------------------------------------------')



e = tf.random_normal([5,5])
ve = sess.run(e)
print("ve:", ve)
print('------------------------------------------------')


f = tf.constant(1.0, dtype=tf.float32, shape=[10,10])
sum_f = tf.reduce_sum(f)
mean_f = tf.reduce_mean(f)  # need a matrix not a number, it will calc the mean number of a matrix
vf = sess.run([f, sum_f, mean_f])
print("vf:", vf)
reshape_f = tf.reshape(f, [1,100])
reshape_f = sess.run(reshape_f)
print("reshape_f:", reshape_f)
print("reshape_f.shape:", reshape_f.shape)
print("reshape_f.len:", len(reshape_f))
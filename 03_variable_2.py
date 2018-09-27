import tensorflow as tf

state = tf.Variable(0, name='counter')
print(state.name)   # counter:0

one = tf.constant(1)

new_value = tf.add(state, one)  # state will add one

# Update 'ref' by assigning 'value' to it.  state is 'ref'
update = tf.assign(state, new_value)

# initialize_all_variables is deprecated, use global_variables_initializer instead
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))
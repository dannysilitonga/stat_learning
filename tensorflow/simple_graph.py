"""
Building and running computational graph for
f(x,y) = x^2*y + 4*y
"""

import tensorflow as tf

# Define all nodes in the graph 
# placeholder 
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# f = x*x*y + 4*y
# operation nodes
square_node = x*x
mult_node = square_node*y 
quadruple_node = 4*y 
adder_node = mult_node + quadruple_node

# Create a Tensorflow session object 
sess = tf.Session()

print(" \nProgram output: ")
print(sess.run(adder_node, feed_dict={x:3, y:2}))
sess.close()

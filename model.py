import tensorflow as tf 
import numpy as np 
import utils

from conf import args as args

tf.reset_default_graph()

#Input placeholders
x = tf.placeholder(tf.float32, shape=(None, 28,28,1), name='input')
y = tf.placeholder(tf.float32, shape=(None, 10),    name='target')


conv1 = tf.contrib.layers.conv2d(x, num_outputs=256, kernel_size=3, stride=2, padding='SAME', 
																	activation_fn=tf.nn.relu, scope='conv1') #size // 2

conv1_pooling = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name='conv1_pool') #size //4

conv2 = tf.contrib.layers.conv2d(conv1_pooling, num_outputs=512, kernel_size=3, stride=1, padding='SAME', 
																	activation_fn=tf.nn.relu, scope='conv2')


output_shape = (28//4)*(28//4)*512

feature_vector = tf.reshape(conv2, (-1, output_shape), name='feature_vector') # (100, 25088)


W1 = tf.Variable(tf.truncated_normal([output_shape, 512], stddev=0.1), name='W1')
B1 = tf.Variable(tf.constant(1.0, shape=[512]), name='B1')

W2 = tf.Variable(tf.truncated_normal([512, 10], stddev=0.1), name='W2')
B2 = tf.Variable(tf.constant(1.0, shape=[10]), name='B2')

fc1 = tf.add(tf.matmul(feature_vector, W1), B1, name='fc1')
fc1_actv = tf.nn.relu(fc1)

fc2 = tf.add(tf.matmul(fc1_actv, W2), B2, name='fc2')

logits = tf.nn.softmax(fc2, name='output')

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc2), name='loss')
train_step    = tf.train.AdamOptimizer(args.learning_rate).minimize(cross_entropy)
correct_prection = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prection, tf.float32), name='accuracy')




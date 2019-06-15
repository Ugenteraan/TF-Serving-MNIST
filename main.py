import tensorflow as tf 
import numpy as np 
import os

import utils #helper functions
from conf import args as args #import arguments from conf file
import model


images = utils.images #(60000, 28,28)
labels = utils.labels #(60000, 10)

total_image_file = images.shape[0]


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

saver = tf.train.Saver(tf.global_variables())

checkpoint = tf.train.latest_checkpoint(args.save_path)

#For training 
if args.train :

	for epoch_iter in range(args.epoch):

		loss = 0

		for i in range(0, total_image_file, args.batch_size):

			end_batch = i + args.batch_size
			if end_batch >= total_image_file : end_batch = None


			loss += sess.run([model.cross_entropy, model.train_step], feed_dict={model.x:images[i:end_batch], model.y:labels[i:end_batch]})[0]

		print("Epoch :  %d, Loss : %g"%(epoch_iter, loss))


	if not os.path.exists(args.save_path):
		os.mkdir(args.save_path)

	saver.save(sess, args.save_path+'model.ckpt') #save the model checkpoint

#For inference 
else:

	graph = tf.get_default_graph()
	saver = tf.train.import_meta_graph(checkpoint + '.meta')
	saver.restore(sess, checkpoint)

	output = graph.get_tensor_by_name('output:0')

	out = sess.run(output, feed_dict={'input:0':images[0:3]})

	print(np.argmax(out, axis=1))
	print(np.argmax(labels[0:3], axis=1))

sess.close()
import tensorflow as tf 
import os
from conf import args as args



serve_path = './serve/%s/%s'%(args.model_name, args.version)

checkpoint= tf.train.latest_checkpoint(args.save_path)


with tf.Session() as sess:

	saver = tf.train.import_meta_graph(checkpoint + '.meta') #import the meta file

	graph = tf.get_default_graph()

	saver.restore(sess, checkpoint) #restore the variables from the checkpoint file

	inputs = graph.get_tensor_by_name('input:0') #input to the graph
	output = graph.get_tensor_by_name('output:0') #output of the graph

	#build tensor info
	model_input = tf.saved_model.utils.build_tensor_info(inputs)
	model_output = tf.saved_model.utils.build_tensor_info(output)

	#signature definition with input and output. Method name is some constant. Not sure what it does.
	signature_definition = tf.saved_model.signature_def_utils.build_signature_def(
		 													inputs={'inputs': model_input},
		 													outputs={'outputs':model_output},
		 													method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)


	builder = tf.saved_model.builder.SavedModelBuilder(serve_path)

	#add the information on the builder's meta
	builder.add_meta_graph_and_variables(
		sess, [tf.saved_model.tag_constants.SERVING],
		signature_def_map = {
				tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_definition
			}
		)

	builder.save()
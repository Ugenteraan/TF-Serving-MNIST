from conf import args as args #import args from conf file
import h5py
import numpy as np



def one_hot_encoder(labels):
	'''Returns the given MNIST labels from np arrays of integers to np array of one hot labels.

       Parameter
       ---------
       labels : np array of MNIST integer labels
    '''

	total_num_labels = labels.shape[0] #get the total num of labels

	one_hot_label = np.zeros([total_num_labels, 10]) #create a numpy array of zeros. 10 for num of classes

	for i in range(total_num_labels):

		one_hot_label[i][int(labels[i])] = 1.0 #the label value will be marked as 1.0 at that specific index

	return one_hot_label #returns the np one-hot label 




def readImages(file_path):
	'''
	Reads the MNIST file in h5 format. The images are stored in a group called images while labels are stored in a group meta.
	Returns numpy arrays of images and corresponding labels. Labels are integers, NOT one-hot vectors.

	Parameter
	--------
	file_path : path to the h5 file    | string
	'''

	file = h5py.File(file_path, 'r+')

	images = np.array(file["/images"]).astype('uint8')
	labels = np.array(file['/meta']).astype('uint8')

	return (images, labels)


images, labels = readImages(args.path)

labels = one_hot_encoder(labels)
images = np.reshape(images, (-1,28,28,1))




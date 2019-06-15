import argparse 
import os

#default values for the args
defaults = {
	'path'          : os.getcwd()+'/MNIST_dataset.h5',
	'batch_size'    : 100,
	'epoch'         : 10000,
	'learning_rate' : 1e-3,
	'save_path'     : os.getcwd()+'/model/',
	'model_name'    : 'mnistpb',
	'version'       : "1",
}


parser = argparse.ArgumentParser(description='NOTE: main.py runs the inference code by default. Use --train to train the model.')

#add the path args
parser.add_argument('--train',         action='store_true',                           help='Use this to train')
parser.add_argument('--path',          default=defaults['path'],          type=str,   help='Input mnist h5 file path')
parser.add_argument('--batch_size',    default=defaults['batch_size'],    type=int,   help='Input batch size')
parser.add_argument('--epoch',         default=defaults['epoch'],         type=int,   help='Input training epoch')
parser.add_argument('--learning_rate', default=defaults['learning_rate'], type=float, help='Input learning rate')
parser.add_argument('--save_path',     default=defaults['save_path'],     type=str,   help='Input save path')
parser.add_argument('--model_name',    default=defaults['model_name'],    type=str,   help='Input model name')
parser.add_argument('--version',       default=defaults['version'],       type=str,   help='Input model version')


args = parser.parse_args()





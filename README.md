# Tensorflow Serving MNIST

This repository contain codes to train an MNIST model using CNN and build the servable model files. TF serving can be used to serve the servable model files. The instructions are as follows :

1) Run `main.py` with _--train_ argument. Checkpoint files will be saved under a folder named `model_ckpt`.
2) Run `model_builder.py`. The servable model files will be saved under a folder named `serve`.
3) Pull the tensorflow/serving docker image with 
~~~
docker pull tensorflow/serving
~~~
4) Run the docker with 
~~~
docker run -t --rm -p 8500:8500 \
-v "PATH_TO_SERVEFOLDER/mnistpb":/models/mnistpb \
-e MODEL_NAME=mnistpb \
tensorflow/serving &
~~~
5) Run `client.py` file to get inference from the served model.
	
 



**License**
-------
**MIT**


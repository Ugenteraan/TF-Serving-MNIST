import utils
import numpy as np


images = utils.images

images = utils.images
labels = utils.labels


import numpy as np
from predict_client.prod_client import ProdClient

HOST = '0.0.0.0:8500'
MODEL_NAME = 'mnistpb'
MODEL_VERSION = 1

client = ProdClient(HOST, MODEL_NAME, MODEL_VERSION) #connect to the tf-serving running through docker

#parameters to request
req_data = [{'in_tensor_name': 'inputs', 'in_tensor_dtype': 'DT_FLOAT', 'data': images[0:4]}]

prediction = client.predict(req_data, request_timeout=10) #get the output

#axis 1 because of batch request
print(np.argmax(np.asarray(prediction['outputs']), axis=1))
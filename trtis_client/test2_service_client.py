#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 22:52:22 2021

@author: cloudy
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 22:01:00 2021

@author: cloudy
"""


from PIL import Image
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype
import time
import requests

jpg_list = ['input.jpg', 'input1.jpg', 'input2.jpg']

start_time = time.time()

for jpg_name in jpg_list:
    ## 前處理
    img = Image.open(jpg_name).convert('L')
    img = img.resize((28, 28))
    imgArr = np.asarray(img)/255
    imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
    
    ## Client-Server 溝通    
    data = {'modelType': "1", 'arr': imgArr.tolist()}

    response = requests.post("http://localhost:5000/predict", json=data)
    print(response.content)
    
time_period = time.time()-start_time
print(time_period)



#%%
from PIL import Image
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype
import time
import requests

jpg_list = ['input.jpg', 'input1.jpg', 'input2.jpg']

start_time = time.time()

for jpg_name in jpg_list:
    ## 前處理
    img = Image.open(jpg_name).convert('L')
    img = img.resize((28, 28))
    imgArr = np.asarray(img)/255
    imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
    
    ## Client-Server 溝通    
    data = {'modelType': "1", 'arr': imgArr.tolist()}
    response = requests.post("http://localhost:5000/predict", json=data)
    
    print(response.content)
    
    ## Client-Server 溝通    
    data = {'modelType': "0", 'arr': imgArr.tolist()}
    response = requests.post("http://localhost:5000/predict", json=data)
    
    print(response.content)
    
    
time_period = time.time()-start_time
print(time_period)


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

jpg_list = ['input.jpg', 'input1.jpg', 'input2.jpg']

start_time = time.time()

for jpg_name in jpg_list:
    ## 前處理
    img = Image.open(jpg_name).convert('L')
    img = img.resize((28, 28))
    imgArr = np.asarray(img)/255
    imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
    imgArr = imgArr.astype(triton_to_np_dtype('FP32'))
    
    ## Client-Server 溝通
    triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=0)
    inputs = []
    inputs.append(grpcclient.InferInput('flatten_input', imgArr.shape, 'FP32'))
    inputs[0].set_data_from_numpy(imgArr)
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('dense_3',class_count=0))
    responses = []
    responses.append(triton_client.infer('mnistdeep',inputs,
                        request_id=str(1),
                        model_version='1',
                        outputs=outputs))
    ## 後處理
    print (np.argmax(responses[0].as_numpy('dense_3')[0]))
    
    
time_period = time.time()-start_time
print(time_period)




#%%
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

jpg_list = ['input.jpg', 'input1.jpg', 'input2.jpg']

start_time = time.time()

for jpg_name in jpg_list:
    ## 前處理
    img = Image.open(jpg_name).convert('L')
    img = img.resize((28, 28))
    imgArr = np.asarray(img)/255
    imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
    imgArr = imgArr.astype(triton_to_np_dtype('FP32'))
    
    ## Client-Server 溝通
    triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=0)
    inputs = []
    inputs.append(grpcclient.InferInput('flatten_input', imgArr.shape, 'FP32'))
    inputs[0].set_data_from_numpy(imgArr)
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('dense_3',class_count=0))
    responses = []
    responses.append(triton_client.infer('mnistdeep',inputs,
                        request_id=str(1),
                        model_version='1',
                        outputs=outputs))
    ## 後處理
    print (np.argmax(responses[0].as_numpy('dense_3')[0]))
    
    ## Client-Server 溝通
    triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=0)
    inputs = []
    inputs.append(grpcclient.InferInput('flatten_input', imgArr.shape, 'FP32'))
    inputs[0].set_data_from_numpy(imgArr)
    outputs = []
    outputs.append(grpcclient.InferRequestedOutput('dense_1',class_count=0))
    responses = []
    responses.append(triton_client.infer('mnist',inputs,
                        request_id=str(1),
                        model_version='1',
                        outputs=outputs))
    ## 後處理
    print (np.argmax(responses[0].as_numpy('dense_1')[0]))
    
    
time_period = time.time()-start_time
print(time_period)



#%%


from PIL import Image
import numpy as np
import tritonclient.grpc as grpcclient
from tritonclient.utils import triton_to_np_dtype
import time

jpg_list = ['input.jpg', 'input1.jpg', 'input2.jpg']

start_time = time.time()

for _ in range(1000000):
    for jpg_name in jpg_list:
        ## 前處理
        img = Image.open(jpg_name).convert('L')
        img = img.resize((28, 28))
        imgArr = np.asarray(img)/255
        imgArr = np.expand_dims(imgArr[:, :, np.newaxis], 0)
        imgArr = imgArr.astype(triton_to_np_dtype('FP32'))
        
        ## Client-Server 溝通
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=0)
        inputs = []
        inputs.append(grpcclient.InferInput('flatten_input', imgArr.shape, 'FP32'))
        inputs[0].set_data_from_numpy(imgArr)
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput('dense_3',class_count=0))
        responses = []
        responses.append(triton_client.infer('mnistdeep',inputs,
                            request_id=str(1),
                            model_version='1',
                            outputs=outputs))
        ## 後處理
        print (np.argmax(responses[0].as_numpy('dense_3')[0]))
        
        ## Client-Server 溝通
        triton_client = grpcclient.InferenceServerClient(url='localhost:8001', verbose=0)
        inputs = []
        inputs.append(grpcclient.InferInput('flatten_input', imgArr.shape, 'FP32'))
        inputs[0].set_data_from_numpy(imgArr)
        outputs = []
        outputs.append(grpcclient.InferRequestedOutput('dense_1',class_count=0))
        responses = []
        responses.append(triton_client.infer('mnist',inputs,
                            request_id=str(1),
                            model_version='1',
                            outputs=outputs))
        ## 後處理
        print (np.argmax(responses[0].as_numpy('dense_1')[0]))
        
    
time_period = time.time()-start_time
print(time_period)
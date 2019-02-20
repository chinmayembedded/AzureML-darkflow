import json
import numpy as np
import os
import tensorflow as tf
import requests
import base64
import io
import cv2
import base64 
from PIL import Image
import argparse
import ast
import time
import json
parser = argparse.ArgumentParser()
from darkflow.cython_utils.cy_yolo2_findboxes import box_constructor    

# api-endpoint
with open('config.json') as f:
    data = json.load(f)
    URL = data["url"]


def getResults(requestData):

    with open(os.path.join("./built_graph/yolov2.meta"), 'r') as fp:
        meta = json.load(fp)
    imgdata = base64.b64decode(requestData["data"])
    data = Image.open(io.BytesIO(imgdata))

    image_np = cv2.cvtColor(np.array(data), cv2.COLOR_BGR2RGB)

    h, w, c = meta['inp_size']

    imsz = cv2.resize(image_np , (w, h))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    image_np_expanded = np.expand_dims(imsz, 0)
    [h, w] = image_np.shape[:2]

    headers = {'Accept': 'application/octet-stream',
            'content-type': 'application/json'}
    start_time1 = time.time()
    r = requests.post(url = URL , data = json.dumps(requestData), headers = headers)
    elapsed_time1 = time.time() - start_time
    print("Time to get the results : ", elapsed_time1)
    data = r.json()
    data1 = ast.literal_eval(data)

    
    out1 = data1["detection"]
    
    boxes = box_constructor(meta, np.array(out1, dtype=np.float32))
    
    #print(boxes)   
    boxesInfo = list()
    #resultsForJSON = []
    #colors = meta['colors']
    threshold = meta['thresh']

    for b in boxes:
        max_indx = np.argmax(b.probs)
        max_prob = b.probs[max_indx]
        label = meta['labels'][max_indx]
        if max_prob > threshold:
            left  = int ((b.x - b.w/2.) * w)
            right = int ((b.x + b.w/2.) * w)
            top   = int ((b.y - b.h/2.) * h)
            bot   = int ((b.y + b.h/2.) * h)
            if left  < 0    :  left = 0
            if right > w - 1: right = w - 1
            if top   < 0    :   top = 0
            if bot   > h - 1:   bot = h - 1
            mess = '{}'.format(label)
            boxResults = (left, right, top, bot, mess, max_indx, max_prob)
            print(boxResults)

    

if __name__ == "__main__":
    parser.add_argument('-u','--url', type=str, help='Image url', required=True)
    args = parser.parse_args()
    start_time = time.time()
    requestData = {"data": base64.b64encode(requests.get(args.url).content).decode("utf-8")}
    elapsed_time = time.time() - start_time
    print("Time to convert the image in base64 : ", elapsed_time)
    getResults(requestData)
    
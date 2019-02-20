import json
import numpy as np
import os
import tensorflow as tf

import base64
import io
import cv2
import base64 
from PIL import Image

from azureml.core.model import Model
str1 = Model.get_model_path('yolomodel')
str2 = "darkflow/cython_utils/cy_yolo2_findboxes"
str3 = os.path.join(str1, str2)
str3 = str3.replace('/', '.') 

print(os.path.abspath("score.py"))
print(os.path.abspath("cy_yolo2_findboxes.pyx"))
#from cy_yolo2_findboxes import box_constructor

detection_graph = tf.Graph()
sess = None
def init():
    print("Hello")
    global meta, inp, out, sess
    model_root = Model.get_model_path('yolomodel')
    with tf.gfile.FastGFile(os.path.join(model_root, 'yolov2.pb'), "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    
    tf.import_graph_def(
        graph_def,
        name = ""
    )
    with open(os.path.join(model_root, 'yolov2.meta'), 'r') as fp:
        meta = json.load(fp)
    
    inp = tf.get_default_graph().get_tensor_by_name('input:0')
    feed = dict() # other placeholders
    out = tf.get_default_graph().get_tensor_by_name('output:0')
    sess = tf.Session(config = tf.ConfigProto())

def run(image):
    str = json.loads(image)["data"]    
    imgdata = base64.b64decode(str)
    data = Image.open(io.BytesIO(imgdata))
    image_np = cv2.cvtColor(np.array(data), cv2.COLOR_BGR2RGB)
    h, w, c = meta['inp_size']
    print("height and width 1",h,w,c)
    imsz = cv2.resize(image_np , (w, h))
    imsz = imsz / 255.
    imsz = imsz[:,:,::-1]
    image_np_expanded = np.expand_dims(imsz, 0)
    [h, w] = image_np.shape[:2]
    print("height and width 2",h,w)

    print("In mode open 2", sess, out)
    feed_dict = { inp : image_np_expanded}
    
    out1 = sess.run(out, feed_dict)[0]
    print("*&&&&&&&&&&&7",out1)
    result = json.dumps({"detection" : out1.tolist()})
    return result
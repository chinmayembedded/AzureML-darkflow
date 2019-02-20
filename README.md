# Introduction
This project contains the deployment of yolo darkflow module in azure machine learning workspace(preview). `Yolo_azure` contains the code files that are required for the deployment of yolo detection service.
`Yolo_client` contains the client module which requests the web service to get the detection results.


# Execution steps


In `Yolo_azure`, 
<br />
1. <b>00.configuration.ipynb</b>
<br />
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This file contains the steps for creating the environment along with the resource group and workspace creation. 
<br />
2. <b>yolo.ipynb</b>
<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This file contains the steps for deploying the web service.
<br />
3. After deploying the service we get the <b>scoring_uri</b>. Copy this URI.
<br />
4. Download the model specified in the link below and paste it into model folder.
<br />
https://drive.google.com/drive/folders/1i1umciZcld4HVX005eCUDAdOwxpju7K2?usp=sharing
<br />

In `yolo_client`,
<br />
1. <b>config.json</b>
<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Paste the scoring URI that is copied from the previous step.
2. <b>request_file.py</b>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 
<br/>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This file takes the image url as an input agrument. After executing this file it converts the image provided in the base64 payload and opens a window with image and number of detections.

<br />
### Run command
`python request_file.py --url <image-url>`
<br/>
e.g python request_file.py --url https://www.uni-regensburg.de/Fakultaeten/phil_Fak_II/Psychologie/Psy_II/beautycheck/english/durchschnittsgesichter/m(01-32)_gr.jpg

<br/>
Images can be of `.jpg`, `.jpeg`, and `.png` format.



# Tensorflow to dlc conversion of model

snpe-tensorflow-to-dlc --graph yolov2.pb  --input_dim input 1,608,608,3 --out_node output --allow_unconsumed_nodes --verbose

Environment
<br/>
1. Ubuntu 14.04<br/>
2. Python 2.7.x<br/>
3. tensorflow 1.5<br/>
4. snpe 1.18

# YOLOv8-with-RGB-D-and-AFFP
This is an official implementation of a paper titled "Using an improved deep instance segmentation network to extract cow shapes from depth images and point clouds spaces"
All codes will be published here.
![嵌套序列 02_5](https://github.com/dontlearncpp/YOLOv8-with-RGB-D-and-AFFP/assets/103402250/7b23319f-baab-4b61-ad93-54caeccb09f3)

## Train & Eval
For train and val, please refer to the branch "Train&Eval". The train and eval methods of yolov8, please refer to the official.
The main code changes we made are predictor.py, stream_loaders.py, base.py, and so on.
### Specify the depth image path
In YOLOv8-with-RGB-D-and-AFFP/ultralytics/yolo/data/base.py
Lines 122-135
### AFFP
![image](https://github.com/dontlearncpp/YOLOv8-with-RGB-D-and-AFFP/assets/103402250/19cbfbfe-ba9c-4ec2-a4a5-31fb68fa81dd)
In YOLOv8-with-RGB-D-and-AFFP/ultralytics/nn/ssf.py

## Predict





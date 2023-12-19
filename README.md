
## The code for Train & Eval, Predict is in different branches.
# YOLOv8-with-RGB-D-and-AFFP
This is an official implementation of a paper titled "Deep instance segmentation network based cow shape extraction using RGB-D images"
All codes have been published in their branches.
![嵌套序列 02_5](https://github.com/dontlearncpp/YOLOv8-with-RGB-D-and-AFFP/assets/103402250/7b23319f-baab-4b61-ad93-54caeccb09f3)

## Train & Eval
For train and val, please refer to the branch "Train&Eval". The train and eval methods of yolov8, please refer to the official.
The main code changes we made are predictor.py, stream_loaders.py, base.py, and so on.
### Specify the depth image path
In YOLOv8-with-RGB-D-and-AFFP/ultralytics/yolo/data/base.py
Lines 122-135
The depth image path was Specified.
### AFFP
![image](https://github.com/dontlearncpp/YOLOv8-with-RGB-D-and-AFFP/assets/103402250/19cbfbfe-ba9c-4ec2-a4a5-31fb68fa81dd)
In YOLOv8-with-RGB-D-and-AFFP/ultralytics/nn/modules.py

## Predict
The main code changes we made are predictor.py, stream_loaders.py, base.py, /ultralytics/yolo/v8/segment
/predict.py and so on.

Input/output image path was Specified in
/predict.py

## Dep2pointcloud
Using our predict code,  the segmentation image was get as shown, and it was into a point cloud with dep2point.py
![image](https://github.com/dontlearncpp/YOLOv8-with-RGB-D-and-AFFP/assets/103402250/0fb74830-da11-4623-b90b-e2e9660a0ede)

## If this code is helpful to you, please cite our paper. It is helpful to me, thank you
Yang, G., Li, R., Zhang, S., Wen, Y., Xu, X., & Song, H. (2023). Extracting cow point clouds from multi-view RGB images with an improved YOLACT++ instance segmentation. Expert Systems with Applications, 230, 120730.








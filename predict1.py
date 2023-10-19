from ultralytics import YOLO
import  cv2
# Load a model
model = YOLO("/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/ultralytics/yolo/v8/segment/runs/segment/train28/weights/best.pt")
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/test.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/ultralytics/datasets/images/val/100.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img/V104.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img/V159.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-mainrgbd-cowfull-wiseiou-ffm-near-wu-predit-cam/ultralytics/datasets/images/train/402.png'
# source = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img'
# source = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/ultralytics/datasets/images/val'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img1'
# name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img1/V500.png'
# name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img/V159.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-portonet-shuchu/imgtest'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-mainrgbd-cowfull-wiseiou-ffm-near-wu-predit-cam/ultralytics/datasets/images/train'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/mm/img-manfanshe'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/RGBD3.4/img/58299206'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/RGBD3.4/img/58299206'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img/V063.png'
# name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img/V159.png'

result = model.predict(
    source=name,
    show=False,
    save=True)

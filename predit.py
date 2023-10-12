from ultralytics import YOLO
import  cv2
# Load a model
model = YOLO("/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-mainrgbd-cowfull/ultralytics/yolo/v8/segment/runs/segment/train102/weights/best.pt")


name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/ultralytics/datasets/images/val/100.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img/V104.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img/V159.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-mainrgbd-cowfull-wiseiou-ffm-near-wu-predit-cam/ultralytics/datasets/images/train/402.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/test-img/V1623.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img/V159.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/ultralytics/datasets/images/val/100.png'
name = '/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/img/V063.png'
result = model.predict(
    # source=name,
    source=name,
    show=False,
    save=True)

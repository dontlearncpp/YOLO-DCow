from ultralytics import YOLO

model = YOLO("/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-mainrgbd-cowfull/ultralytics/yolo/v8/segment/runs/segment/train102/weights/best.pt")
# model = YOLO("/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-mainrgbd-cowfull-pronet/ultralytics/yolo/v8/segment/runs/segment/train142/weights/best.pt")
#model = YOLO("/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-mainrgbd-cowfull/ultralytics/yolo/v8/segment/runs/segment/train122/weights/best.pt")

metrics = model.val()
metrics = model.val()
print(metrics.seg.all_ap)

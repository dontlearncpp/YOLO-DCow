from ultralytics import YOLO

model = YOLO("/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-mainrgbd-cowfull-wiseiou-ffm-near-wu/ultralytics/yolo/v8/segment/runs/segment/train118/weights/best.pt")

metrics = model.val()
metrics = model.val()
print(metrics.seg.all_ap)

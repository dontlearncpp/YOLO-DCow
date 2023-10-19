from ultralytics import YOLO
import os
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import math

model = YOLO("/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/ultralytics/yolo/v8/segment/runs/segment/train28/weights/best.pt")
depth_path = "/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-mainrgbd-cowfull-wiseiou-ffm-near-wu-predit-cam/ultralytics/datasets/images/traindepth"
img_path = "/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/test-img"
save_path = "/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/ultralytics/datasets/images/result-test"
save_img = "/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/ultralytics/datasets/images/result_image-test"
save_dep = "/media/xingshixu/367a0adf-4bec-4c64-b23d-945aacb28ba5/yangyolo/ultralytics-maincowfull-shuchu/ultralytics/datasets/images/result_dep-test"


imgs =os.listdir(img_path)
for img in imgs:
    result = model.predict(
        name = save_path,
        source=img_path+"/"+img,
        show=False,
        save = True)
    a=1
    cls = result[0] .boxes.cls
    cls = cls.cpu().numpy()
    if (np.where(cls==0) !=0):
        index = np.where(cls == 1)
        index =np.array(index)
        for i in range (index.size):

            ver = result[0].masks.segments[index[0][i]]

            images = cv2.imread(img_path+"/"+img)
            dep_img = cv2.imread(depth_path+"/"+img,-1)

            gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
            mask = np.zeros_like(gray)

            ver[:, 0] = np.floor((ver[:, 0] * 1920))
            ver[:, 1] = np.floor((ver[:, 1] * 1080))
            ver = ver.astype(int)

            cv2.fillPoly(mask,[ver],255)

            mask1 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            dep_img = cv2.cvtColor(dep_img, cv2.COLOR_GRAY2BGR)

            # cv2.imwrite('111.jpg',mask)

            #mask_inv = cv2.bitwise_not(mask)
            masked_img = cv2.bitwise_and(images,mask1)
            # masked_dep = cv2.bitwise_and(dep_img,mask)
            mask2 = np.zeros((1080, 1920, 3))
            mask2[:, :, 0] = mask
            mask2[:, :, 1] = mask
            mask2[:, :, 2] = mask
            mask2.astype(np.bool)
            masked_dep = np.where(mask2, dep_img, np.zeros_like(dep_img))
            # masked_dep = dep_img & (mask2.astype(np.bool))
            masked_img = Image.fromarray(masked_img,'RGB')
            masked_dep = np.asarray(masked_dep,np.uint16)
            cv2.imwrite(save_dep+"/"+img, masked_dep)
            # masked_dep = Image.fromarray(masked_dep,'RGB')

            masked_img.save(save_img+"/"+img)
            # masked_dep.save(save_dep+"/"+img)


            # cv2.imwrite(save_img+"/"+img, masked_img)
            # cv2.imwrite(save_dep+"/"+img, masked_dep)

import numpy as np
import cv2

def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

import os

image1_path = "../model/data/유형별 두피 이미지/Training/탈모/[원천]탈모_2.중등도/"
image2_path = "../model/data/유형별 두피 이미지/Training/탈모/[원천]탈모_3.중증/"
root1, _, files1 = list(os.walk(image1_path))[0]
root2, _, files2 = list(os.walk(image2_path))[0]
for i in range(len(files1)):
    image_1 = cv2.imread(os.path.join(root2, files2[i]))
    image_2 = cv2.imread(os.path.join(root1, files1[i]))
    lam = np.random.beta(1.0, 1.0) 
    bbx1, bby1, bbx2, bby2 = rand_bbox(image_1.shape[:2], lam)
    image_1[bbx1:bbx2, bby1:bby2] = image_2[bbx1:bbx2, bby1:bby2]
    cv2.imwrite(f"{image2_path}cutmix/2{files2[i]}", image_1)

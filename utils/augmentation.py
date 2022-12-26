from PIL import Image
import os

image_path="../model/data/유형별 두피 이미지/Training/탈모/[원천]탈모_3.중증/"
for root, _, files in os.walk(image_path):
    for file in files:
        image = Image.open(os.path.join(root, file))
        flipped_image = image.transpose(Image.FLIP_LEFT_RIGHT)
        image.save(f"{image_path}/1{file}")
        image.close()
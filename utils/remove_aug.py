import os

image_path_1="../model/data/유형별 두피 이미지/Training/탈모/[원천]탈모_2.중등도/"
image_path_2="../model/data/유형별 두피 이미지/Training/탈모/[원천]탈모_1.경증/"
image_path_3="../model/data/유형별 두피 이미지/Training/탈모/[원천]탈모_3.중증/"
image_path_4="../model/data/유형별 두피 이미지/Training/탈모/[원천]탈모_0.양호/"
image_paths = [(image_path_4, "양호  "), (image_path_2,"경증  "), (image_path_1, "중등도"), (image_path_3, "중증  ")]

for _path, sym in image_paths:
    idx = 0
    for root, _, files in os.walk(_path):
        for file in files:
            idx+=1
            # if int(file.split("_")[0]) > 10000:
            #     os.remove(os.path.join(root, file))
    print(f"[탈모] {sym} train datasets: {idx}개")
        
# Telomere

> *두피 질환 진단 및 치료 과정에서 필요한 두피 이미지 데이터 10만건을 딥러닝 모델에 학습시켜   
> 기존 육안으로 의사가 두피를 진단하는 방식이 아닌 두피측정 빅데이터를 이용한 인공지능(AI) 진단 서비스*


## Model & Utils
> 탈모 이미지를 증상의 심화도 별로 학습하여 심화 정도를 4단계로 구분하는 Image Classification 모델   
> Preprocessing, Augmentation, Cut-Mix, Label Smoothing 기법을 적용한 MobileNetV3Large 모델에   
> 탈모 데이터를 학습시켜 가장 정확도가 높은 모델을 저장함.

## Demo
> 최종 모델을 200 epoch 학습시킨 후 이를 aram_model5.pt로 저장하였다. 이후 유저가 두피 사진을 업로드 시   
> 백엔드에서 저장되어 있는 aram_model5.pt 모델을 불러와 업로드한 이미지를 모델에 넣어 결과를 예측하고   
> 이를 웹사이트에 띄우도록 구현하였다.

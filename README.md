<h1> Blurring sensitive Information 🚗🙎</h1>

</br>

<a href = "https://github.com/sungjun4403/ML_BLURinfo/blob/main/BLUR_%EC%B5%9C%EC%A2%85%EC%95%88.pdf"> 🌟 프로젝트 문서 </a>

<a href = "https://github.com/sungjun4403/ML_BLURinfo/blob/main/%EC%82%AC%EC%9A%A9%EC%84%A4%EB%AA%85%EC%84%9C.md"> ⏯️ 시연 영상</a>

<a href = "https://colab.research.google.com/drive/1qm5c9zf-13LGtQDa4vJ8459K_HE60SH3?usp=sharing"> 🖥️ 코드</a> 

</br>

프로젝트 목적 : __번호판, 얼굴 모자이크__ 

과정 : 사람, 자동차 인식 → 얼굴, 번호판 추출 후 모자이크 → 원 이미지 합성

</br>

번호판 인식 : 자동차 인식 결과에서 연속적인 문자 비율 감지

얼굴 인식 : 사람 인식 결과에서 상단 30% 추출 

</br>



<img width="1100" src="https://user-images.githubusercontent.com/96364048/191471682-f6513498-34df-42e5-8823-7b16c031c188.png">

</br>

소스 이미지, 결과 이미지 *

<img style="float:left" width="500" src="https://user-images.githubusercontent.com/96364048/191434531-a74e409a-1324-4546-84c5-7bafb69b66c1.png">'<img style="float:right" width="500" src="https://user-images.githubusercontent.com/96364048/191434545-91223e1a-aa5c-4059-b9a6-5c5af6a4d3f7.png">

</br>

----

<br/>

<h3>참고</h3>

<br/>

⭐번호판 인식
- https://github.com/ChaminLee/2019.10.28-Extracting-Car-Numbers-with-OpenCV/blob/master/extract%20car%20number%20in%20well-done%20image.ipynb

<br/>

사물 인식 튜토리얼
- https://www.tensorflow.org/hub/tutorials/object_detection?hl=ko

<br/>

FasterRCNN + InceptionResNet V2 모델 (tf)
- https://tfhub.dev/tensorflow/faster_rcnn/inception_resnet_v2_1024x1024/1

<br/>

Faster RCNN / SSD / YOLO 비교 
- https://github.com/sejongresearch/FlowerClassification/issues/18

<br/>

InceptionResNet 설명
- https://medium.com/@zahraelhamraoui1997/inceptionresnetv2-simple-introduction-9a2000edcdb6

<br/>

CNN 기본 원리 설명 (합성곱부터)
- http://taewan.kim/post/cnn/#7-%EC%B0%B8%EA%B3%A0%EC%9E%90%EB%A3%8C
- https://halfundecided.medium.com/%EB%94%A5%EB%9F%AC%EB%8B%9D-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-cnn-convolutional-neural-networks-%EC%89%BD%EA%B2%8C-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-836869f88375

<br/>

YOLO
- https://ctkim.tistory.com/91

<br/>

ResNet
- https://wikidocs.net/137252

<br/>


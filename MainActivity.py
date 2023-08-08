import cv2
import numpy as np

# 웹캠 신호 받기
# VideoCapture - opencv 에서 동영상 입력 부분을 관리하는 함수
# cv.VideoCapture(filename) -> <VideoCapture object>
# cv.VideoCapture(device) -> <VideoCapture object>
# parameter에 파일 명을 넣으면 저장된 비디오를 불러오고 0, 1 등을 넣으면 입력 디바이스 순서에 따라 실시간 촬영 frame 을 받아 온다.
videoSignal = cv2.VideoCapture(0)
# YOLO 가중치 파일과 CFG 파일 로드 | 다운로드 링크 : https://pjreddie.com/darknet/yolo/
YOLO_net = cv2.dnn.readNet("yolov2-tiny.weights", "yolov2-tiny.cfg")

# YOLO NETWORK 재구성
classes = []
with open("yolo.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = YOLO_net.getLayerNames()
output_layers = [layer_names[i - 1] for i in YOLO_net.getUnconnectedOutLayers()]

while True:
    # 웹캠 프레임
    ret, frame = videoSignal.read()
    h, w, c = frame.shape
    size = videoSignal.get(cv2.CAP_PROP_FRAME_WIDTH)
    # YOLO 입력
    # cv2.dnn.blobFromImage(image, scalefactor=None, size=None, mean=None, swapRB=None, crop=None, ddepth=None) -> retval
    # scalefactor   : 입력 영상 픽셀 값에 곱할 값 | 0 ~ 255 픽셀값을 이용했는지, 0 ~ 1로 정규화해서 이용했는지에 맞게 지정
    #                   0 ~ 1로 정규화하여 학습을 진행했으면 1/255를 입력(0.00392...)
    # size          : 학습 영상의 크기, size로 resize를 해주어 출력함. yolov2-tiny.cfg(학습 모델) 의 size 가 416, 416
    # mean          : 입력 영상 각 채널에서 뺄 평균 값, 학습할 때 mean 값을 빼서 계산한 경우 그와 동일한 mean 값을 지정합니다.
    # swapRB        : RGB에서 R 과 B 채널을 서로 바꿀 것인지를 결정하는 플래그. default = False
    # crop          : 학습할 때 영상을 잘라서 학습하였으면 그와 동이하게 입력
    # ddepth        : 출력 블롭의 깊이. CV_32F 또는 CV_8U. default = CV_32F
    # retval        : 영상으로부터 구한 블롭 객체. numpy.ndarray.shpe=(N, C, H, W). N : 개수, C : 채널 개수, HW : 영상 크기
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # setInput      : 입력 blob을 이용하여 네트워크 실행
    # forward       : 네트워크 입력을 설정한 후 네트워크를 순방향으로 실행하여 결과를 예측하는 함수
    YOLO_net.setInput(blob)
    outs = YOLO_net.forward(output_layers)

    # 이 뒤는 네트워크를 실행한 결과에 따라서 결과(사각테두리)를 출력하는 부분
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 검출 신뢰도
            if confidence > 0.5:
                # Object detected
                # 검출기의 경계상자 좌표는 0 ~ 1로 정규화 되어있으므로 다시 전처리
                cx = int(detection[0] * w)
                cy = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)

                # Rectangle coordinate
                x = int(cx - dw / 2)
                y = int(cy - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]

            # 경계상자와 클래스 정보 투영
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
            cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

    cv2.imshow("YOLOv2", frame)

    # 100ms 마다 영상 갱신
    if cv2.waitKey(100) > 0:
        break
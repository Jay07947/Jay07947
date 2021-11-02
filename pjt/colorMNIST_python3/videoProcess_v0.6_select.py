import cv2 as cv
import numpy as np
from tkinter.filedialog import *
from sklearn.svm import SVC
from sklearn import metrics
import pandas as pd

## 함수선언
## 영상에 그림그리기
def draw_ball_location(img_color1, locations):
    global img_color, movieName, list_ball_location, history_ball_locations, isDraw, img_draw
    global hsv, lower_color, upper_color

    for i in range(len(locations) - 1):
        if locations[0] is None or locations[1] is None:
            continue
        cv.line(img_color1, tuple(locations[i]), tuple(locations[i + 1]), (255, 255, 255), 30)
    return img_color1

def writeInVideo(a) :
    global img_color , movieName,  list_ball_location , history_ball_locations, isDraw, img_draw
    global hsv, lower_color, upper_color, cap, answer
    #################### 딥러닝 내장함수 #######################
    def figureString():
        global img_color, movieName, list_ball_location, history_ball_locations, isDraw, img_draw
        global hsv, lower_color, upper_color, cap, answer
        import joblib

        print('문자 인식 실행중... 잠시만 기다리세요...!!')
        # key = cv.waitKey(-1)
        img1 = cv.resize(img_draw, dsize=(28, 28))
        ret1, img1 = cv.threshold(img1, 10, 255, cv.THRESH_BINARY) # 라인값 강화 전처리...
        # cv.imshow('capture', img1) # 데이터셋이 될 바이너리 이미지

        # 이미지를 데이터셋으로 만들기
        # Array 펴주기  > (28*28) = 1 * 784
        c_train = img1.reshape(1,img1.shape[0] * img1.shape[1])# 2차원 픽셀 데이터를 2차원데이터(1개, 784데이터셋)로 재배열

        print(c_train)
        #c_train = c_train.astype('float32')  # 0~255를 0~1로 스케일링2


        # 미리 훈련된 딥러닝 모델 불러와서 돌리기
        clf = joblib.load('mnistMLP.dmp')
        print('모델 로딩 성공')

        answer = clf.predict(c_train)
        print('읽어온 문자 : ', answer)


        #key = cv.waitKey(-1) # 분석보다 waitkey가 먼저 들어가면 뒤에가 안돌아감.. ㅠ

        list_ball_location.clear()
        history_ball_locations.clear()
   #############################################################
    if a == '1' :
        # 1) 저장된 파일 불러오기
        movieName = askopenfilename(parent=cv.namedWindow('Result'),filetypes=(('동영상 파일', '*.mp4;*.avi'), ('All File', '*.*')))
        cap = cv.VideoCapture(movieName)
    else :
        # 2) 실시간 카메라 (default camera) 사용
        # 카메라 윈도우 사이즈 조정
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, 512)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, 512)

    while True:
        cv.setMouseCallback('Result', mouse_callback)
        ret, img_color = cap.read()

        #img_color = cv.flip(img_color, 1)  # 동영상 좌우반전
        img_hsv = cv.cvtColor(img_color, cv.COLOR_BGR2HSV) # HSV 영상으로 변환


        # cv.inRange() : 소스인 hsv의 모든 값을 lower_color, upper_color로 지정한 범위에 있는지 체크 후
        # 범위에 해당하는 부분은 값 그대로, 나머지는 0으로 채워서 결과값 반환 (이진화개념)
        img_mask = cv.inRange(img_hsv, lower_color, upper_color)
        # 선택 물체 색공간 팽창을 통해 빈공간 메꿔주기 (모폴로지 3번반복)
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
        img_mask = cv.morphologyEx(img_mask, cv.MORPH_DILATE, kernel, iterations=3)
        img_mask = cv.resize(img_mask, dsize=(512, 512))
        # 선택 영역 네모칸 쳐주기(레이블링)
        nlabels, labels, stats, centroids = cv.connectedComponentsWithStats(img_mask)
        max = -1
        max_index = -1
        # 마스크로 영역표시 한 바이너리 이미지를 다시 컬러로 변환, 나중에 컬러 라인과 점 표시하려고!
        #img_draw = cv.cvtColor(img_mask, cv.COLOR_GRAY2BGR)
        img_draw = img_mask.copy()
        ## 전체 검출된 영역중에
        for i in range(nlabels):
            if i < 1:
                continue
            area = stats[i, cv.CC_STAT_AREA]
        # 최대영역 (해당물체영역)을 찾아서
            if area > max:
                max = area
                max_index = i
        # 최대영역의 좌표 및 넓이 스펙 계산
        if max_index != -1:
            center_x = int(centroids[max_index, 0])
            center_y = int(centroids[max_index, 1])
            left = stats[max_index, cv.CC_STAT_LEFT]
            top = stats[max_index, cv.CC_STAT_TOP]
            width = stats[max_index, cv.CC_STAT_WIDTH]
            height = stats[max_index, cv.CC_STAT_HEIGHT]
            # 해당 물체 외곽에 사각형, 중심에 원 그리기
            #cv.rectangle(img_draw, (left, top), (left + width, top + height), (0, 0, 255), 1)
            cv.circle(img_draw, (center_x, center_y), 10, (0, 0, 255), -1)


            if isDraw:
                # isDraw == True 일때 현재 ball location을 list_ball에 저장
                list_ball_location.append((center_x, center_y))

            else:
                # isDraw == False 일때 현재까지 그렸던 ball locaion 위치들을 복사하여
                # history_ball_location에 삽입
                history_ball_locations.append(list_ball_location.copy())
                # 현재까지 그렸던 그림 삭제
                list_ball_location.clear()

        text ='The answer is ' + str(answer)

        # 화면에  ball location 그려주기
        cv.rectangle(img_draw, (64, 64), (448, 448), (255), 1)
        img_draw = draw_ball_location(img_draw, list_ball_location)
        for ball_locations in history_ball_locations:
            img_draw = draw_ball_location(img_draw, ball_locations)

        # 화면에 읽어온 문자 출력
        cv.putText(img_color, text , (100, 100), cv.FONT_HERSHEY_SIMPLEX, 1, (42, 42, 165), 3, cv.LINE_AA)
        # olive(128, 128, 0), violet(221, 160, 221), brown(42, 42, 165)

        cv.imshow('Draw', img_draw)
        cv.imshow('Result', img_color)

        # space bar 32

        key = cv.waitKey(1)  # key > 0 읻 때 재생 , 아닐때 일시정지

        if key == 27:  # 동영상 종료 esc 27
            break
        elif key == ord('c'):  # 화면 clear
            list_ball_location.clear()
            history_ball_locations.clear()
        elif key == ord('d'):  # 그리기 Stop & Run
            isDraw = not isDraw
        elif key == ord('s') :  # 동영상 정지
            key = cv.waitKey(-1)
            cv.setMouseCallback('Result', mouse_callback)
        elif key == ord('a'):  # 해당 그림 딥러닝으로 분석하여 문자 변환 실행 # 동영상 정지
            figureString()
            cv.setMouseCallback('Result', mouse_callback)


def nothing(x) :
    pass

def mouse_callback(event, x, y, flags, param):
    global img_color, movieName, list_ball_location, history_ball_locations, isDraw
    global hsv, lower_color, upper_color
    # 마우스 왼쪽 버튼 누를시 위치에 있는 픽셀값을 읽어와서 HSV로 변환합니다.
    if event == cv.EVENT_LBUTTONDOWN:
        print('### selected color ###')
        print(' BGR :',img_color[y, x])
        color = img_color[y, x]

        one_pixel = np.uint8([[color]])
        hsv = cv.cvtColor(one_pixel, cv.COLOR_BGR2HSV)
        hsv = hsv[0][0]
        print(' hsv : ', hsv[0], ', ', hsv[1], ', ', hsv[2])

        threshold = cv.getTrackbarPos('threshold', 'img_result')
        # HSV 색공간에서 마우스 클릭으로 얻은 픽셀값과 유사한 픽셀값의 범위를 정합니다.
        # h = 0 red / ~30 orange / ~60 yellow / ~ 120 green ~ / ~180 cyan / ~ 240 blue / 270 ~ violet / 300 magenta
        # cvtColor 함수를 사용하여 변화하면 0(=180) < Hue < 179,  0 < Saturation < 255, 0 < Value < 255 의 범위를 갖습니다.
        # 위 그림의 값에 0.5를 곱하면 원하는 색의 hue 값입니다.
        # hue값 over/underflow처리
        if hsv[0] < 1:
            lower_color = np.array([hsv[0]-1+180, 130, 80])
            upper_color = np.array([hsv[0]+1, 255, 255])
            print("#####################")
        elif hsv[0] > 178:
            lower_color = np.array([hsv[0]-1, 130, 80])
            upper_color = np.array([hsv[0]+1-179, 255, 255])
            print("#####################")
        else:
            lower_color = np.array([hsv[0], 130, 80])
            upper_color = np.array([hsv[0]+3, 255, 255])
            print("#####################")





## 전역변수
img_color , movieName, img_draw, cap , answer= None, None, None, None, ' '
list_ball_location = []
history_ball_locations = []
isDraw = True
hsv, lower_color, upper_color = 0,0,0

## 메인코드
mode = input("실행 모드를 선택하세요 (1: 파일 불러오기, 2: 실시간 웹캠 실행)")
writeInVideo(mode) #메인 동영상 메소드
cap.lelease() #카메라 닫기
cv.destroyAllWindows() # 윈도우 닫기
from tkinter import *
from tkinter.filedialog import *
from tkinter.simpledialog import *
import math
import cv2
import numpy

## 함수 선언부
# 공통함수
def malloc(h, w, value=0) :

    retMemory = [ [ value for _ in range(w)]  for _ in range(h) ]
    return retMemory

def bufferFile() :
    global window, canvas, paper, inImage, outImage,orgImage, inH, inW, outH, outW, filename, orgH, orgW
    inH = outH;    inW = outW
    ## 버퍼 메모리 할당
    inImage = []
    for _ in range(RGB):
        inImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                inImage[rgb][i][k] = outImage[rgb][i][k]

def undoImage() : # 임시메모리를 이용해 버퍼이미지(입력이미지)와 출력이미지를 바꿔줌
    global window, canvas, paper, inImage, outImage, orgImage, inH, inW, outH, outW, filename, orgH, orgW
    # 임시 메모리에 현재 출력이미지 저장
    tmpH = outH ; tmpW = outH
    tmpImage = []
    for _ in range(RGB) :
        tmpImage.append(malloc(outH, outW))
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpImage[rgb][i][k] = outImage[rgb][i][k]
    #입력이미지를 출력으로 (이전이미지를 현재 출력으로)저장
    outH = inH;    outW = inW
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(inH, inW))
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                outImage[rgb][i][k] = inImage[rgb][i][k]
    #임시메모리에 있던 출력이미지를 입력으로 저장
    inH = tmpH;    inW = tmpW
    inImage = []
    for _ in range(RGB):
        inImage.append(malloc(tmpH, tmpW))
    for rgb in range(RGB):
        for i in range(tmpH):
            for k in range(tmpW):
                inImage[rgb][i][k] = tmpImage[rgb][i][k]
    print('undoImage success')

def openFile() :
    global window, canvas, paper, inImage, outImage, orgImage, inH, inW, outH, outW, filename,  orgH, orgW
    global cvInImage, cvOutImage
    ## 파일 선택하기
    filename = askopenfilename(parent=window,
           filetypes=(('Color 파일', '*.jpg;*.png;*.bmp;*.tif'), ('All File', '*.*')))
    ## (중요!) 입력이미지의 높이와 폭 알아내기
    cvInImage = cv2.imread(filename)
    orgH = cvInImage.shape[0]
    orgW = cvInImage.shape[1]
    ## 입력이미지용 메모리 할당
    orgImage=[]
    for _ in range(RGB):
        orgImage.append(malloc(orgH, orgW))
    ## 파일 --> 메모리 로딩
    for i in range(orgH):
        for k in range(orgW):
            orgImage[R][i][k] = cvInImage.item(i, k ,B)
            orgImage[G][i][k] = cvInImage.item(i, k, G)
            orgImage[B][i][k] = cvInImage.item(i, k, R)
    outH = orgH;    outW = orgW
    ## 출력할 이미지 저장
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    for rgb in range(RGB):
        for i in range(orgH):
            for k in range(orgW):
                outImage[rgb][i][k] = orgImage[rgb][i][k]
    ########################
    displayImageColor()

import numpy as np
def saveImage() :
    global window, canvas, paper, inImage, outImage, orgImage, inH, inW, outH, outW, filename, orgH, orgW
    global cvInImage, cvOutImage
    if filename == None or filename == '' :
        return
    saveCvPhoto = np.zeros((outH, outW, 3), np.uint8)
    for i in range(outH) :
        for k in range(outW) :
            tup = tuple(([outImage[B][i][k],outImage[G][i][k],outImage[R][i][k]]))
            saveCvPhoto[i,k] = tup

    saveFp = asksaveasfile(parent=window, mode='wb',defaultextension='.', filetypes=(("그림 파일", "*.png;*.jpg;*.bmp;*.tif"), ("모든 파일", "*.*")))
    if saveFp == '' or saveFp == None:
        return
    cv2.imwrite(saveFp.name, saveCvPhoto)

def displayImageColor() :
    global window, canvas, paper, inImage, outImage, orgImage, inH, inW, outH, outW, filename, orgH, orgW
    global cvInImage, cvOutImage
    window.geometry(str(outW)+'x'+str(outH))
    if canvas != None :
        canvas.destroy()
    canvas = Canvas(window, height=outH, width=outW)
    paper = PhotoImage(height=outH, width=outW)
    canvas.create_image((outW / 2, outH / 2), image=paper, state='normal')
    # 메모리에서 처리한 후, 한방에 화면에 보이기 --> 완전 빠름
    rgbString =""
    for i in range(outH) :
        tmpString = "" # 각 줄
        for k in range(outW) :
            r = outImage[R][i][k]
            g = outImage[G][i][k]
            b = outImage[B][i][k]
            tmpString += "#%02x%02x%02x " % (r, g, b)
        rgbString += '{' + tmpString + '} '
    paper.put(rgbString)
    canvas.pack()
    status.configure(text='이미지정보:' + str(outH) + 'x' + str(outW)+'      '+filename)



######### 영상 처리 함수 ##########
# 색상 조절 메뉴
def rgbChangeColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(inH, inW))
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                outImage[rgb][i][k] = orgImage[rgb][i][k]
    ### 진짜 영상처리 알고리즘 ###
    selectC = askinteger('어떤 색상을 변경하시겠습니까?','1.Red 2. Green 3. Blue', minvalue=1,maxvalue=3)
    value = askinteger('변경할 값', '변경하실 값을 입력하세요(-255 ~ 255) :', minvalue=-255,maxvalue=255)
    for i in range(inH):
        for k in range(inW):
            if inImage[selectC-1][i][k]+value >255 :
                outImage[selectC-1][i][k] = 255
            elif inImage[selectC-1][i][k]+value < 0 :
                outImage[selectC-1][i][k] = 0
            else :
                outImage[selectC-1][i][k] = inImage[selectC-1][i][k]+value
    ########################
    displayImageColor()

# 화소점처리
def equalColor() :
    global window, canvas, paper, inImage, outImage, orgImage,inH, inW, outH, outW, filename, orgH, orgW
    global cvInImage, cvOutImage
    global  orgImage,bufferImage, orgH, orgW, bfH, bfW
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = orgH;    outW = orgW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(orgH, orgW))
    ### 진짜 영상처리 알고리즘 ###
    for rgb in range(RGB):
        for i in range(orgH):
            for k in range(orgW):
                outImage[rgb][i][k] = orgImage[rgb][i][k]

    ########################
    print('equalColor success')
    displayImageColor()

def addColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    value = askinteger("변경할 밝기 값", "값을 입력하세요(-255~255) :")
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                out = inImage[rgb][i][k] + value
                if out > 255 :
                    outImage[rgb][i][k] = 255
                elif out < 0 :
                    outImage[rgb][i][k] = 255
                else :
                    outImage[rgb][i][k] = out
    ########################
    displayImageColor()

def grayColor() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    for i in range(inH):
        for k in range(inW):
            c = inImage[R][i][k] + inImage[G][i][k] + inImage[B][i][k]
            c = int(c/3)
            outImage[R][i][k] = outImage[G][i][k] = outImage[B][i][k] = c
    ########################
    displayImageColor()

def paraCupColor():  # 파라볼라 컵 알고리즘 실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    outH = inH;
    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
     outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    ## Out = 255.0 *((In / 128.0 - 1.0)**2)
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                v = 255.0 * ((inImage[rgb][i][k] / 128.0 - 1.0) ** 2)
                if v > 255:
                    outImage[rgb][i][k] = 255
                elif v < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(v)
    # 출력이미지 현재 상태에 저장
    displayImageColor()

def paraCapColor():  # 파라볼라 캡 알고리즘 실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;
    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    ## Out = 255.0 *((In / 128.0 - 1.0)**2)
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                v = 255 - 255.0 * ((inImage[rgb][i][k] / 128.0 - 1.0) ** 2)
                if v > 255:
                    outImage[rgb][i][k] = 255
                elif v < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(v)
    ########################
    displayImageColor()

def gammaColor():  # 감마 알고리즘 / 실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;
    outW = inW
    ## 출력이미지 메모리 할당
    for _ in range(RGB):
         outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    # Out = I **(1/r)
    r = askfloat("감마연산", "값을 입력하세요 : ")
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                v = inImage[rgb][i][k] ** (1 / r)
                if v > 255:
                    outImage[rgb][i][k] = 255
                elif v < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(v)
    #
    displayImageColor()

def bwColor(): #이진화 (기본) / 실행완료
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;
    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
       outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    hap1, hap2, hap3 = 0,0,0
    for i in range(inH):
        for k in range(inW):
            hap1 += inImage[0][i][k]
            hap2 += inImage[1][i][k]
            hap3 += inImage[2][i][k]
    avg1 = hap1 / (inH * inW)
    avg2 = hap2 / (inH * inW)
    avg3 = hap3 / (inH * inW)
    print(str(avg1))
    for i in range(inH):
        for k in range(inW):
            if inImage[0][i][k] > avg1:
                outImage[0][i][k] = 255
            else:
                outImage[0][i][k] = 0
            if inImage[1][i][k] > avg2:
                outImage[1][i][k] = 255
            else:
                outImage[1][i][k] = 0
            if inImage[1][i][k] > avg3:
                outImage[1][i][k] = 255
            else:
                outImage[1][i][k] = 0
    ########################
    displayImageColor()

def bw2Color(): # 이진화 평균값 / 실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;
    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
         outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    hap1, hap2, hap3 = 0,0,0
    for i in range(inH):
        for k in range(inW):
            hap1 += inImage[0][i][k]
            hap2 += inImage[1][i][k]
            hap3 += inImage[2][i][k]
    avg1 = hap1 / (inH * inW)
    avg2 = hap2 / (inH * inW)
    avg3 = hap3 / (inH * inW)

    for i in range(inH):
        for k in range(inW):
            if inImage[0][i][k] > avg1:
                outImage[0][i][k] = 255
            else:
                outImage[0][i][k] = 0

            if inImage[1][i][k] > avg2:
                outImage[1][i][k] = 255
            else:
                outImage[1][i][k] = 0

            if inImage[2][i][k] > avg3:
                outImage[2][i][k] = 255
            else:
                outImage[2][i][k] = 0
    ########################
    displayImageColor()

def bw3Color(): # 이진화 중위수 /실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;
    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
         outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    mid = 0
    tmpAry = []
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpAry.append(inImage[rgb][i][k])
        tmpAry.sort()
        mid = tmpAry[int((inH * inW) / 2)]
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                if inImage[rgb][i][k] > mid:
                    outImage[rgb][i][k] = 255
                else:
                    outImage[rgb][i][k] = 0
    ########################
    displayImageColor()

def point2Color(): #실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
            outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    p1 , p2 = 0,0
    p1 = askinteger("강조하고 싶은 범위", "값 :")
    p2 = askinteger("강조하고 싶은 범위", "값 :")
    if p1 > p2:
        p1, p2 = p2, p1
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                if p1 < int(inImage[rgb][i][k]) < p2:
                    outImage[rgb][i][k] = 255
                else:
                    outImage[rgb][i][k] = inImage[rgb][i][k]
    ########################
    displayImageColor()

#기하학 처리
def moveColor():  # 영상이동 실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
         outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    dx = askinteger("", "x변위 :")
    dy = askinteger("", "y변위 :")
    for i in range(inH):
        for k in range(inW):
            if 0 <= i + dx < outH and 0 <= k + dy < outW:
                outImage[0][i + dx][k + dy] = inImage[0][i][k]
                outImage[1][i + dx][k + dy] = inImage[1][i][k]
                outImage[2][i + dx][k + dy] = inImage[2][i][k]
    ########################
    displayImageColor()

def zoomOutColor(): #실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    scale = askinteger("축소", "배율을 입력하세요 :")  # 짝수로 함.. 홀수는 처리할게 많음 먼저 입력받아야 메모리 결정
    outH = int(inH / scale)
    outW = int(inW / scale)
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
         outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    for i in range(outH):
        for k in range(outW):
            outImage[0][i][k] = inImage[0][i * scale][k * scale]
            outImage[1][i][k] = inImage[1][i * scale][k * scale]
            outImage[2][i][k] = inImage[2][i * scale][k * scale]
    # 출력이미지 현재 상태에 저장
    ########################
    displayImageColor()

def zoomInColor():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    scale = askinteger("확대", "배율을 입력하세요 :")
    outH = inH*scale;
    outW = inW*scale
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    for rgb in range(RGB):
        for i in range(outH):  # inH /inW 만큼 돌림
            for k in range(outW):
                #  outImage[i*scale][k*scale] = inImage[i][k]   ## forwording 기법 hole문제 발생
                outImage[rgb][i][k] = inImage[rgb][int(i / scale)][int(k / scale)]  ## 픽셀이 너무 커짐.. 깨짐 보간법 필요
    ########################
    displayImageColor()

def zoomInColor2(): # 실행취소 적용
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    scale = askinteger("확대", "배율")
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH*scale;
    outW = inW*scale
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    # 보간법 이미지 처리시 index가 +1 만큼 모자라서 만드는 임시 이미지
    tmpInImage = []
    for _ in range(RGB):
        tmpInImage.append(malloc(outH+1, outW+1))
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i][k] = float(inImage[rgb][i][k])
    ### 진짜 영상처리 알고리즘 ###
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if i % scale == 0 and k % scale == 0:
                    outImage[rgb][i][k] = int(tmpInImage[rgb][int(i / scale)][int(k / scale)])
                else:
                    a = float((scale - (k % scale)) / scale)
                    b = float(((k % scale)) / scale)
                    num1 = a* tmpInImage[rgb][int(i / scale)][int(k / scale)] + b * tmpInImage[rgb][int(i / scale) + 1][int(k / scale)]
                    ## 앞에 곱해준거는 보간법 거리비 가중치 ## 뒤에는 본인 행의 가장 가까운 원이미지의 좌우 포인트 값 (사각형으로 치면 윗 꼭지점 두개)
                    num2 = a * tmpInImage[rgb][int(i / scale)][int(k / scale) + 1] + b * tmpInImage[rgb][int(i / scale) + 1][int(k / scale) + 1]
                    ## 앞에 곱해준거는 보간법 거리비 가중치 ## 뒤에는 본인 다음 행의 가장 가까운 원이미지의 좌우 포인트 값 (사각형으로 치면 아래 꼭지점)
                    num3 = a * num1 + b * num2
                    ## num1과 num2를 다시 거리비 가중치로 해당 포인트 값 구해줌
                    outImage[rgb][i][k] = int(num3)
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if outImage[rgb][i][k]  > 255:
                    outImage[rgb][i][k] = 255
                elif outImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
    ########################
    displayImageColor()

def mirrorLRColor(): #실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;
    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = inImage[rgb][i][outW - k - 1]
    ########################
    displayImageColor()

def mirrorUDColor(): #실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;
    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = inImage[rgb][outH - i - 1][k]
    ########################
    displayImageColor()

def rotationColor():  # 밝게하기 알고리즘 /실행취소 적용완료 / 아직 다 못적음
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    cx = inH // 2
    cy = inW // 2
    #xd = cos*xs-sin*ys
    #xy = sin*xs+cos*ys
    angle = askinteger('회전','각도',minvalue=0,maxvlaue=360)
    r = angle * math.pi/180
    for rgb in range(RGB):
        for i in range(outH) :
            for k in range(outW) :
                xs = i ; ys = k
                xd = int(math.cos(r)*xs - math.sin(r)*ys - math.sin(r)*(ys-cy)+cx)
                yd = int(math.sin(r)*xs + math.cos(r)*ys + math.con(r)*(ys-cy)+cy)
                if 0<= xd < outH and 0 <= yd < outW :
                    outImage[rgb][xd][yd] = inImage[rgb][xs][ys]
    ########################
    displayImageColor()

def rotation90LColor(): #실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inW;    outW = inH
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = inImage[rgb][k][outH - i - 1]
    # 출력이미지 현재 상태에 저장
    ########################
    displayImageColor()

def rotation90RColor(): #실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inW;    outW = inH
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = inImage[rgb][outW - k - 1][i]
    ########################
    displayImageColor()

#특수효과 메뉴
def embossSelect():
    winMessage = Tk()
    winMessage.title('엠보싱 유형 선택')
    rb1 = Button(winMessage, text='첫번째', command=embossColor1)
    rb2 = Button(winMessage, text='두번째', command=embossColor3)
    btn = Button(winMessage, text="닫기", command=winMessage.destroy)
    #rb3 = Button(winMessage, text='세번째', command=embossColor2)
    rb1.pack()
    rb2.pack()
    btn.pack()
    #rb3.pack()


    winMessage.mainloop()

def embossColor1(): #실행취소
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    ## (중요!) 마스크
    mask = [[-1, 0, 0], [0, 0, 0], [0, 0, 1]]
    mSize = 3
    for rgb in range(RGB):
        tmpInImage = malloc(outH + 2, outW + 2, 127)
        tmpOutImage = malloc(outH, outW)
    # inImage - > tmpInImage
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])
    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = 0.0
                for m in range(mSize):
                    for n in range(mSize):
                        S += mask[rgb][m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                tmpOutImage[rgb][i - 1][k - 1] = S
    # 마무리 마스크에 따라서 127 더할지 결정
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpOutImage[rgb][i][k] += 127.0
    ## 임시 Outtmp에서 outImage로 넣으면서 overflow 체크하기
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    ########################
    displayImageColor()


def embossColor3():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, RGB
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;
    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    ## (중요!) 마스크
    mask = [[1, 1, 1], [1, -8, 1], [1, 1, 1]]
    mSize = 3
    for rgb in range(RGB):
        tmpInImage = malloc(outH + 2, outW + 2, 127)
        tmpOutImage = malloc(outH, outW)
    # inImage - > tmpInImage
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])
    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = 0.0
                for m in range(mSize):
                    for n in range(mSize):
                        S += mask[rgb][m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                tmpOutImage[rgb][i - 1][k - 1] = S
    # 마무리 마스크에 따라서 127 더할지 결정
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpOutImage[rgb][i][k] += 127.0
    ## 임시 Outtmp에서 outImage로 넣으면서 overflow 체크하기
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    ########################
    displayImageColor()

def blurrColor():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, RGB
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;
    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    # 보간법 이미지 처리시 index가 +1 만큼 모자라서 만드는 임시 이미지
    tmpInImage = []
    for _ in range(RGB):
        tmpInImage.append(malloc(outH+1, outW+1))
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i][k] = float(inImage[rgb][i][k])
    ## (중요!) 마스크
    # 임시공간에 채워줄 중위값 구하기

    # 마스크 사이즈 및 마스크 값 결정
    value = askfloat("블러링 효과","블러의 강도를 입력해주세요(2~30) : ",minvalue=2, maxvalue = 30)
    mSize = int(value)
    mask = []
    a = ( 1/(mSize*mSize) )
    mask = malloc(mSize, mSize, a )
    tmpInImage , tmpOutImage = [], []
    for rgb in range(RGB):
        tmpInImage.append(malloc(outH + mSize -1, outW + mSize -1, 127))
        tmpOutImage.append(malloc(outH, outW))
    # inImage - > tmpInImage
    j = int(mSize/2)
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i][k] = float(inImage[rgb][i][k])
    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(j, inH + j):
            for k in range(j, inW + j):
                # 각 점을 처리
                S = 0.0
                for m in range(mSize):
                    for n in range(mSize):
                        S += mask[m][n] * tmpInImage[rgb][m + i - j][n + k - j]
                tmpOutImage[rgb][i - j][k - j] = S
    ## 임시 Outtmp에서 outImage로 넣으면서 overflow 체크하기
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    ########################
    displayImageColor()

def sharpSelect():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, selectV
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return

    def sharpColor():
        global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
        global cvInImage, cvOutImage
        if filename == '' or filename == None:
            return
        bufferFile()
        ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
        outH = inH;
        outW = inW
        ## 출력이미지 메모리 할당
        outImage = []
        for _ in range(RGB):
            outImage.append(malloc(outH, outW))
        ## (중요!) 마스크  #### 엠보싱에서 마스크만 바꾸면 된다!!!!!
        mask = []
        if selectV.get() == 1:
            mask = [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]

        elif selectV.get() == 2:
            mask = [[0.0, -1.0, 0.0], [-1.0, 5.0, -1.0], [0.0, -1.0, 0.0]]

        elif selectV.get() == 3:
            mask = [[-1.0, 0.0, -1.0], [0.0, 5.0, 0.0], [-1.0, 0.0, -1.0]]

        elif selectV.get() == 4:
            mask = [[-1 / 9.0, -1 / 9.0, -1 / 9.0], [-1 / 9.0, -8 / 9.0, -1 / 9.0], [-1 / 9.0, -1 / 9.0, -1 / 9.0]]

        mSize = 3
        # 임시 이미지에 입력이미지 복사
        tmpInImage , tmpOutImage = [], []
        for _ in range(RGB):
            tmpInImage.append(malloc(outH + 2, outW + 2,127))
            tmpOutImage.append(malloc(outH, outW))
        for rgb in range(RGB):
            for i in range(inH):
                for k in range(inW):
                    tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])
        # 회선 연산 : 마스크로 긁어가면서 처리하기
        for rgb in range(RGB):
            for i in range(1, inH + 1):
                for k in range(1, inW + 1):
                    # 각 점을 처리
                    S = 0.0
                    for m in range(mSize):
                        for n in range(mSize):
                            S += mask[rgb][m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                    tmpOutImage[rgb][i - 1][k - 1] = S
        ### 마스크 연산시 손실값 보상
        for rgb in range(RGB):
            for i in range(outH):
                for k in range(outW):
                    tmpOutImage[rgb][i][k] += 127.0
        ## 임시 Outtmp에서 outImage로 넣으면서 overflow 체크하기
        for rgb in range(RGB):
            for i in range(outH):
                for k in range(outW):
                    if tmpOutImage[rgb][i][k] > 255:
                        outImage[rgb][i][k] = 255
                    elif tmpOutImage[rgb][i][k] < 0:
                        outImage[rgb][i][k] = 0
                    else:
                        outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
        ########################
        displayImageColor()


    ## (중요!) 마스크  #### 엠보싱에서 마스크만 바꾸면 된다!!!!!
    winMessage1 = tkinter.Tk()
    winMessage1.title('샤프닝 유형 선택')
    selectV = tkinter.IntVar()
    rb1 = tkinter.Radiobutton(winMessage1, text='1번 마스크', value=1, variable=selectV, command=sharpColor)
    rb2 = tkinter.Radiobutton(winMessage1, text='2번 마스크', value=2, variable=selectV, command=sharpColor)
    rb3 = tkinter.Radiobutton(winMessage1, text='3번 마스크', value=3, variable=selectV, command=sharpColor)
    rb4 = tkinter.Radiobutton(winMessage1, text='4번 마스크', value=4, variable=selectV, command=sharpColor)
    btn = Button(winMessage1, text="닫기", command=winMessage1.destroy)
    rb1.pack()
    rb2.pack()
    rb3.pack()
    rb4.pack()
    btn.pack()

    winMessage1.mainloop()

def smoothColor():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    ## (중요!) 마스크  ####
    mask = [[1 / 16.0, 1 / 8.0, 1 / 16.0], [1 / 8.0, 1 / 4.0, 1 / 8.0], [1 / 16.0, 1 / 8.0, 1 / 16.0]]
    # 가우시안 필터 마스크.. 경계선이 약화되는 효과
    mSize = 3
    # 임시 이미지
    tmpInImage, tmpOutImage = [] , []
    for _ in range(RGB):
        tmpInImage.append(malloc(outH+2, outW+2,127))
        tmpOutImage.append(malloc(outH, outW))
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i][k] = float(inImage[rgb][i][k])
    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = 0.0
                for m in range(mSize):
                    for n in range(mSize):
                        S += mask[rgb][m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                tmpOutImage[rgb][i - 1][k - 1] = S

    ## 임시 Outtmp에서 outImage로 넣으면서 overflow 체크하기
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    ########################
    displayImageColor()

def edgeColor1(): #실행취소
    global window, canvas, paper, inImage, outImage, currentImage, undoImage, inH, inW, outH, outW, crrH, crrW, undoH, undoW, filename
    if filename == '' or filename == None:
        return
    outH = crrH
    outW = crrW
    outImage = malloc(outH, outW)
    # 현재상태 undo 공간에 저장
    undoW = crrW
    undoH = crrH
    undoImage = malloc(crrH, crrW)
    for i in range(crrH):
        for k in range(crrW):
            undoImage[i][k] = currentImage[i][k]
    ## (중요!) 마스크  ####
    mask = [[0, 0, 0], [-1, 1, 0], [0, 0, 0]]
    # 수직에지검출 마스크
    mSize = 3
    tmpInImage = malloc(outH + 2, outW + 2, 127)
    tmpOutImage = malloc(outH, outW)
    # inImage - > tmpInImage
    for i in range(inH):
        for k in range(inW):
            tmpInImage[i + 1][k + 1] = float(inImage[i][k])

    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = 0.0
                for m in range(mSize):
                    for n in range(mSize):
                        S += mask[rgb][m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                tmpOutImage[rgb][i - 1][k - 1] = S
    ### 마스크 연산시 손실값 보상
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpOutImage[rgb][i][k] += 127.0
    ## 임시 Outtmp에서 outImage로 넣으면서 overflow 체크하기
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    ########################
    displayImageColor()

def edgeColor2():
    global window, canvas, paper, inImage, outImage, currentImage, undoImage, inH, inW, outH, outW, crrH, crrW, undoH, undoW, filename
    if filename == '' or filename == None:
        return
    outH = crrH
    outW = crrW
    outImage = malloc(outH, outW)
    # 현재상태 undo 공간에 저장
    undoW = crrW
    undoH = crrH
    undoImage = malloc(crrH, crrW)
    for i in range(crrH):
        for k in range(crrW):
            undoImage[i][k] = currentImage[i][k]
    ## (중요!) 마스크  ####
    mask = [[0, -1, 0], [0, 1, 0], [0, 0, 0]]
    # 수직에지검출 마스크
    mSize = 3
    tmpInImage = malloc(outH + 2, outW + 2, 127)
    tmpOutImage = malloc(outH, outW)
    # inImage - > tmpInImage
    for i in range(inH):
        for k in range(inW):
            tmpInImage[i + 1][k + 1] = float(inImage[i][k])

    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(1, inH + 1):
            for k in range(1, inW + 1):
                # 각 점을 처리
                S = 0.0
                for m in range(mSize):
                    for n in range(mSize):
                        S += mask[rgb][m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
                tmpOutImage[rgb][i - 1][k - 1] = S
    ### 마스크 연산시 손실값 보상
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpOutImage[rgb][i][k] += 127.0
    ## 임시 Outtmp에서 outImage로 넣으면서 overflow 체크하기
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    ########################
    displayImageColor()

def doGColor():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB):
        outImage.append(malloc(outH, outW))
    ## (중요!) 마스크  #### 엠보싱에서 마스크만 바꾸면 된다!!!!!
    mask = [[0, 0, -1, -1, -1, 0, 0], [0, -2, -3, -3, -3, -2, 0], [-1, -3, 5, 5, 5, -3, -1], [-1, -3, 5, 16, 5, -3, -1],
            [-1, -3, 5, 5, 5, -3, -1], [0, -2, -3, -3, -3, -2, 0], [0, 0, -1, -1, -1, 0, 0]]
    # DoG마스크 7*7
    mSize = 7
    # 임시 이미지
    tmpInImage, tmpOutImage = [] , []
    for _ in range(RGB):
        tmpInImage.append(malloc(outH+6, outW+6,127))
        tmpOutImage.append(malloc(outH, outW))
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i][k] = float(inImage[rgb][i][k])
    # 회선 연산 : 마스크로 긁어가면서 처리하기
    for rgb in range(RGB):
        for i in range(3, inH + 3):  # 0,0이 마스크에 들어가려면
            for k in range(3, inW + 3):
                # 각 점을 처리
                S = 0.0
                for m in range(mSize):
                    for n in range(mSize):
                        S += mask[rgb][m][n] * tmpInImage[rgb][m + i - 3][n + k - 3]
                tmpOutImage[rgb][i - 3][k - 3] = int(S)
    ### 마스크 연산시 손실값 보상
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                tmpOutImage[rgb][i][k] += 127.0
    ## 임시 Outtmp에서 outImage로 넣으면서 overflow 체크하기
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if tmpOutImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif tmpOutImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(tmpOutImage[rgb][i][k])
    ########################
    displayImageColor()

#히스토그램 메뉴
def stretchColor1():  # 히스토그램 스트레치  / 실행취소 적용
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH
    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    # out = (in - low) / (high - low) *255
    low = [0,0,0]
    high = [0,0,0]
    low[0] = high[0] = inImage[0][0][0]
    low[1] = high[1] = inImage[1][0][0]
    low[2] = high[2] = inImage[2][0][0]
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                if low[rgb] > inImage[rgb][i][k]:
                    low[rgb] = inImage[rgb][i][k]
                elif high[rgb] < inImage[rgb][i][k]:
                    high[rgb] = inImage[rgb][i][k]
    print('최소, 최대값 가져오기 성공')
    for rgb in range(RGB):
        out = 0
        for i in range(inH):
            for k in range(inW):
                out = (inImage[rgb][i][k] - low[rgb]) / (high[rgb] - low[rgb]) * 255.0
                if out > 255:
                    outImage[rgb][i][k] = 255
                elif out < 0:
                    outImage[rgb][i][k] = 0
                else:
                    outImage[rgb][i][k] = int(out)
    print('stretchColor1 success')
    ########################
    displayImageColor()

def stretchColor2():  # 엔드 인 탐색/ 실행취소 적용
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ## (중요!) 출력이미지의 높이, 폭을 결정 ---> 알고리즘에 의존
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))

    ### 진짜 영상처리 알고리즘 ###
    #out = (in - low) / (high - low) *255
    low = [0,0,0]
    high = [0,0,0]
    low[0] = high[0] = inImage[0][0][0]
    low[1] = high[1] = inImage[1][0][0]
    low[2] = high[2] = inImage[2][0][0]
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                if low[rgb] > inImage[rgb][i][k]:
                    low[rgb] = inImage[rgb][i][k]
                elif high[rgb] < inImage[rgb][i][k]:
                    high[rgb] = inImage[rgb][i][k]
    for rgb in range(RGB):
        if 0 < low[rgb] < 50 :
            low[rgb] = 50
        if 205 < high[rgb] < 255 :
            high[rgb] = 205
    for rgb in range(RGB):
        for i in range(outH) :
            for k in range(outW) :
                out = (inImage[rgb][i][k]-low[rgb])/(high[rgb]-low[rgb])*255.0
                if out > 255 :
                    outImage[rgb][i][k] = 255
                elif out < 0 :
                    outImage[rgb][i][k] = 0
                else :
                    outImage[rgb][i][k] = int(out)
    print('stretchColor2 success')
    ########################
    displayImageColor()

def eqealizedColor():  # 히스토그램 평활화 / 실행취소 적용
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    outH = inH;    outW = inW
    ## 출력이미지 메모리 할당
    outImage = []
    for _ in range(RGB) :
        outImage.append(malloc(outH, outW))
    ### 진짜 영상처리 알고리즘 ###
    # 1단계 : 히스토그램 만들기
    histo = [[0 for _ in range(256)] for _ in range(RGB)]
    for i in range(inH):
        for k in range(inW):
            histo[0][inImage[0][i][k]] += 1
            histo[1][inImage[1][i][k]] += 1
            histo[2][inImage[2][i][k]] += 1
    # 2단계 : 누적 히스토그램 만들기
    sumHisto = [[0 for _ in range(256)] for _ in range(RGB)]
    for rgb in range(RGB):
      sumHisto[rgb][0]=histo[rgb][0]
    for rgb in range(RGB):
        for i in range(1,256):
            sumHisto[rgb][i] = histo[rgb][i] +  sumHisto[rgb][i-1]
    # 3단계 : 정규화 히스토그램
    # n = 누적합*(1/(inH*inW))*255
    normalHisto = [[0 for _ in range(256)] for _ in range(RGB)]
    for rgb in range(RGB):
        for i in range(256):
            normalHisto[rgb][i] = sumHisto[rgb][i]*(1/(inH*inW))*255.0
    ### 진짜 영상처리 알고리즘  ###
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                outImage[rgb][i][k] = int(normalHisto[rgb][inImage[rgb][i][k]])
    ########################
    print('eqealizedColor success')
    displayImageColor()


#스티커 꾸미기

def click():
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, xPoint, yPoint
    global cvInImage, cvOutImage
    messagebox.showinfo('위치선택','원하는 위치를 클릭하세요')
    canvas.bind("<Button>",sticker)
    canvas.mainloop()
    ########################

def sticker(event) :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, xPoint, yPoint
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    messagebox.showinfo('스티커 선택','스티커 이미지를 선택하세요')
        ## 파일 선택하기
    filename = askopenfilename(parent=window,
           filetypes=(('Color 파일', '*.jpg;*.png;*.bmp;*.tif'), ('All File', '*.*')))
    ## (중요!) 입력이미지의 높이와 폭 알아내기
    cvInImage = cv2.imread(filename)
    skH = cvInImage.shape[0]
    skW = cvInImage.shape[1]
    ## 입력이미지용 메모리 할당
    skImage=[]
    for _ in range(RGB):
        skImage.append(malloc(skH, skW))
    ## 파일 --> 메모리 로딩
    for i in range(skH):
        for k in range(skW):
            skImage[R][i][k] = cvInImage.item(i, k ,B)
            skImage[G][i][k] = cvInImage.item(i, k, G)
            skImage[B][i][k] = cvInImage.item(i, k, R)
    for m in range(skH):
        for n in range(skW):
            if skImage[0][m][n]==255  and skImage[1][m][n] ==255 and skImage[2][m][n] ==255 :
                    skImage[0][m][n] = 300
                    skImage[1][m][n] = 300
                    skImage[2][m][n] = 300
    xPoint, yPoint = int(event.y), int(event.x)
    print(str(xPoint),str(yPoint))
    for rgb in range(RGB):
        for m in range(skH):
          for n in range(skW):
                 if (n+yPoint)<= outW and m+xPoint <= outH and  skImage[rgb][m][n] <=255 :
                    outImage[rgb][m+xPoint][n+yPoint] = skImage[rgb][m][n]

    print('stikerColor success')
    displayImageColor()

def stikerFrameColor3() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
        ## 파일 선택하기
    filename = askopenfilename(parent=window,
           filetypes=(('Color 파일', '*.jpg;*.png;*.bmp;*.tif'), ('All File', '*.*')))
    ## (중요!) 입력이미지의 높이와 폭 알아내기
    cvInImage = cv2.imread(filename)
    skH = cvInImage.shape[0]
    skW = cvInImage.shape[1]
    ## 입력이미지용 메모리 할당
    skImage=[]
    for _ in range(RGB):
        skImage.append(malloc(skH, skW))
    ## 파일 --> 메모리 로딩
    for i in range(skH):
        for k in range(skW):
            skImage[R][i][k] = cvInImage.item(i, k ,B)
            skImage[G][i][k] = cvInImage.item(i, k, G)
            skImage[B][i][k] = cvInImage.item(i, k, R)
    ### 진짜 영상처리 알고리즘 ### 
    for rgb in range(RGB):
        for k in range(0,outW,skW):
            for m in range(skH):
                for n in range(skW):
                    if skImage[rgb][m][n]<255 and (k+n)<outW :
                        outImage[rgb][20+m][k+n] = skImage[rgb][m][n]
                        outImage[rgb][outH-skH-35+m][k+n] = skImage[rgb][m][n]
    ########################
    print('stikerFrameColor success')
    displayImageColor()

def stikerFrameColor1() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    if filename == '' or filename == None:
        return
    bufferFile()
    ### 진짜 영상처리 알고리즘 ###
    for rgb in range(RGB):
        for i in range(17):
            for k in range(0,outW):
                    outImage[rgb][i][k] = 255               
        for i in range(outH-65,outH):
            for k in range(0,outW):
                    outImage[rgb][i][k] = 255               
        for i in range(outH):
            for k in range(0,17):
                outImage[rgb][i][k] = 255
        for i in range(outH):
            for k in range(outW-17,outW):
                outImage[rgb][i][k] = 255
    ########################
    print('stikerFrameColor1 success')
    displayImageColor()


## 전역 변수부
window, canvas, paper = None, None, None
inImage, outImage, orgImage= [], [] , [];
inH, inW, outH, outW, orgH, orgW= 0,0,0,0,0,0
cvInImage, cvOutImage = None, None
filename = ''
RGB,R, G, B= 3, 0, 1, 2
## 메인 코드부

if __name__ == '__main__' :
    window = Tk()
    window.title('칼라 영상처리 Ver 0.1')
    window.geometry('512x512')
    window.resizable(height=False, width=False)
    status = Label(window, text='이미지정보:', bd=1, relief=SUNKEN, anchor=W)
    status.pack(side=BOTTOM, fill=X)
    ### 메뉴 만들기 ###
    mainMenu = Menu(window)
    window.configure(menu=mainMenu)
    fileMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="파일", menu=fileMenu)
    fileMenu.add_command(label="열기(Open)", command=openFile)
    fileMenu.add_command(label="저장(Save)", command=saveImage)
    fileMenu.add_separator()
    fileMenu.add_command(label="닫기(Close)")
    runMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="실행", menu=runMenu)
    runMenu.add_command(label="원본이미지", command=equalColor)
    runMenu.add_command(label="실행취소", command=undoImage)
    runMenu.add_command(label="재실행", command=undoImage)
    valueMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="밝기조절", menu=valueMenu)
    valueMenu.add_command(label="밝기조절", command=addColor)
    valueMenu.add_command(label="감마연산", command=gammaColor)
    valueMenu.add_command(label="파라볼라(Cup)", command=paraCupColor)
    valueMenu.add_command(label="파라볼라(Cap)", command=paraCapColor)
    saturationMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="색상조절", menu=saturationMenu)
    saturationMenu.add_command(label="그레이스케일", command=grayColor)
    saturationMenu.add_command(label="색상 조절", command=rgbChangeColor)
    saturationMenu.add_command(label="이진화(기본)", command=bwColor)
    saturationMenu.add_command(label="이진화(평균값)", command=bw2Color)
    saturationMenu.add_command(label="이진화(중위수)", command=bw3Color)
    saturationMenu.add_command(label="범위강조 변환", command=point2Color)
    geometryMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="기하학 처리", menu=geometryMenu)
    geometryMenu.add_command(label="이동", command=moveColor)
    geometryMenu.add_command(label="축소", command=zoomOutColor)
    geometryMenu.add_command(label="확대", command=zoomInColor)
    geometryMenu.add_command(label="확대(보간법 적용)", command=zoomInColor2)
    geometryMenu.add_command(label="미러링(좌우)", command=mirrorLRColor)
    geometryMenu.add_command(label="미러링(상하)", command=mirrorUDColor)
    #geometryMenu.add_command(label="회전", command=rotationColor)
    geometryMenu.add_command(label="좌로 90도회전", command=rotation90LColor)
    geometryMenu.add_command(label="우로 90도회전", command=rotation90RColor)
    AreaMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="특수효과처리", menu=AreaMenu)
    AreaMenu.add_command(label="엠보싱", command=embossSelect)
    AreaMenu.add_command(label="블러링", command=blurrColor)
    AreaMenu.add_command(label="부드럽게", command=smoothColor)
    AreaMenu.add_command(label="샤프닝", command=sharpSelect)
    AreaMenu.add_command(label="수직에지검출", command=edgeColor1)
    AreaMenu.add_command(label="수평에지검출", command=edgeColor2)
    AreaMenu.add_command(label="DoG", command=doGColor)
    histogramMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="히스토그램", menu=histogramMenu)
    histogramMenu.add_command(label="스트레칭", command=stretchColor1)
    histogramMenu.add_command(label="스트레칭 앤드-인", command=stretchColor2)
    histogramMenu.add_command(label="평활화", command=eqealizedColor)
    stikerMenu = Menu(mainMenu)
    mainMenu.add_cascade(label="꾸미기", menu=stikerMenu)
    stikerMenu.add_command(label="스티커 붙이기", command=click)
    stikerMenu.add_command(label="프레임 꾸미기1", command=stikerFrameColor1)
    stikerMenu.add_command(label="프레임 꾸미기(사용자정의)", command=stikerFrameColor3)

    ######################

    window.mainloop()
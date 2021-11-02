from tkinter import *
from tkinter.filedialog import *
from tkinter.simpledialog import *
import math
import cv2
import numpy
from datetime import datetime
import userManager

## 함수 선언부
# 로그인 함수


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
    logSave('undoImage')
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
    upMySQL()
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
    logSave('saveImage')

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

def logSave(log) :
    global u_id, p_id, logMsg
    logDateTime = datetime.now()
    logMsg = log
    ###################
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()
    sql = "INSERT INTO userlog_table VALUES(null,'"+str(u_id)+"','"+str(p_id)+"','"+str(logDateTime)+"','"+str(logMsg)+"');"
    print(sql)
    cur.execute(sql)
    conn.commit()
    cur.close()
    conn.close()

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
    logSave('rgbChangeColor')

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
    logSave('equalColor')
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
    logSave('addColor')

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
    logSave('grayColor')

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
    logSave('moveColor')

def zoomOutColor():
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
    logSave('zoomOutColor')

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
    logSave('zoomOutColor2')

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
    logSave('mirrorLRColor')

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
    logSave('mirrorUDColor')

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
    logSave('rotationColor')

#특수효과 메뉴
def pixelation () :
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
    # 마스크 사이즈 및 마스크 값 결정
    value = askinteger("모자이크 효과","모자이크 크기를 입력해주세요 : ", minvalue=2, maxvalue = outH/2)
    for rgb in range(RGB):
        for i in range(0, inH, value):
            for k in range(0, inW, value):
                # 각 점을 처리
                S = 0.0
                for m in range(value):
                    for n in range(value):
                        if i+m < inH and k+n < inW :
                           S += inImage[rgb][i+m][k+n]
                for m in range(value):
                    for n in range(value):
                        if i+m < inH and k+n < inW :
                           outImage[rgb][i+m][k+n] = int(S/(value*value))
    for rgb in range(RGB):
        for i in range(outH):
            for k in range(outW):
                if outImage[rgb][i][k] > 255:
                    outImage[rgb][i][k] = 255
                elif outImage[rgb][i][k] < 0:
                    outImage[rgb][i][k] = 0               
    ########################
    displayImageColor()
    logSave('pixelation')

def embossColor1():

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
    mask = [[-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    mSize = 3
    for rgb in range(RGB):
        tmpInImage = malloc(outH + 2, outW + 2, 127.0)
        tmpOutImage = malloc(outH, outW, 0.0)
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
    logSave('embossing')

def blurrColor():
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
    # 임시공간에 채워줄 중위값 구하기
    mid = 0
    # 마스크 사이즈 및 마스크 값 결정
    value = askfloat("블러링 효과","블러의 강도를 입력해주세요(2~30) : ",minvalue=2, maxvalue = 30)
    mSize = int(value)
    mask = []
    a = ( 1/(mSize*mSize) )
    mask = malloc(mSize, mSize, a )
    tmpOutImage = []
    a = mid()
    for rgb in range(RGB):
        tmpInImage = malloc(outH + 2, outW + 2, 127)
        tmpOutImage = malloc(outH, outW)
    # inImage - > tmpInImage
    for rgb in range(RGB):
        for i in range(inH):
            for k in range(inW):
                tmpInImage[rgb][i + 1][k + 1] = float(inImage[rgb][i][k])                      
    for rgb in range(RGB):
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
    logSave('blurr Size : '+mSize)

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
            tmpInImage.append(malloc(outH + 2, outW + 2, 127))
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
                            S += mask[m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
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
        logSave('sharp mask: ' + selectV)


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
                        S += mask[m][n] * tmpInImage[rgb][m + i - 1][n + k - 1]
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
    logSave('smooth')


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
    ########################
    displayImageColor()
    logSave('stretchColor1')

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
    ########################
    displayImageColor()
    logSave('stretchColor2')

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
    displayImageColor()
    logSave('eqealizedColor')
    
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

    logSave('stikerColor :'+filename)
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
    logSave('stikerFrameColor : '+filename)
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
    logSave('stikerFrameColor1')
    displayImageColor()

###### DB 연동 함수 (MySQL)######

import tempfile
import os
import pymysql
import random
def upMySQL() :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename, p_id
    global cvInImage, cvOutImage, conn, cur, fileList
    if filename == None or filename == '':
        return
    saveCvPhoto = np.zeros((outH, outW, 3), np.uint8)
    for i in range(outH):
        for k in range(outW):
            tup = tuple(([outImage[B][i][k], outImage[G][i][k], outImage[R][i][k]]))
            saveCvPhoto[i, k] = tup
    saveFname = tempfile.gettempdir() + '/' + os.path.basename(filename)
    cv2.imwrite(saveFname, saveCvPhoto)
    ##############
    '''
    DROP DATABASE IF exists photo_db;
    CREATE DATABASE photo_db;
    USE photo_db;
    CREATE TABLE photo_table (
      p_id INT PRIMARY KEY,
      p_fname VARCHAR(255),
      p_ext   CHAR(5),
      p_size  BIGINT,
      p_height INT,
      p_width  INT,
      p_upDate datetime,
      p_upUser CHAR(10),
      p_photo LONGBLOB
    );
    '''
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()  # 빈 트럭 준비
    p_id = random.randint(0, 2100000000)
    tmpName = os.path.basename(os.path.basename(saveFname))
    p_fname, p_ext = tmpName.split('.')
    p_size = os.path.getsize(saveFname)
    tmpImage = cv2.imread(saveFname)
    p_height = tmpImage.shape[0]
    p_width = tmpImage.shape[1]
    p_upDate = str(datetime.now())
    p_upUser = 'root' # 로그인한 사용자
    # 파일을 읽기
    fp = open(saveFname, 'rb')
    blobData = fp.read()
    fp.close()
    # 파일 정보 입력
    sql = "INSERT INTO photo_table(p_id, p_fname, p_ext, p_size, p_height, p_width, "
    sql += "p_upDate, p_UpUser, p_photo) VALUES (" + str(p_id) + ", '" + p_fname + "', '" + p_ext
    sql += "', " + str(p_size) + "," + str(p_height) + "," + str(p_width) + ", '" + p_upDate
    sql += "', '" + p_upUser +  "', %s )"
    tupleData = (blobData,)
    cur.execute(sql,tupleData)
    conn.commit()
    cur.close()
    conn.close()
    messagebox.showinfo("MySQL에 저장",filename + '을 성공적으로 SQL에 저장하였습니다.')
    logSave('upMySQL : '+filename)
    #############

def downMySQL() : # 파일 열기 개념....
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage, conn, cur, fileList, p_id
    ###################
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()  # 빈 트럭 준비
    sql = "SELECT p_id, p_fname, p_ext, p_size FROM photo_table"
    cur.execute(sql)
    fileList = cur.fetchall()
    p_id=fileList[0]
    cur.close()
    conn.close()
    ##################
     # 서브 윈도창 나오기.
    def downLoad() :
        global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
        global cvInImage, cvOutImage, conn, cur, fileList
        selectIndex = listData.curselection()[0]
        conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
        cur = conn.cursor()  # 빈 트럭 준비
        sql = "SELECT  p_fname, p_ext, p_photo FROM photo_table WHERE p_id= "
        sql += str(fileList[selectIndex][0])
        cur.execute(sql)
        p_fname, p_ext, p_photo = cur.fetchone()
        fullPath = tempfile.gettempdir() + '/' + p_fname + '.' + p_ext
        fp = open(fullPath, 'wb')
        fp.write(p_photo)
        print(fullPath)
        fp.close()
        cur.close()
        conn.close()
        filename = fullPath
        subWindow.destroy()
        ####
        cvInImage = cv2.imread(filename)
        orgH = cvInImage.shape[0]
        orgW = cvInImage.shape[1]
        ## 입력이미지용 메모리 할당
        orgImage = []
        for _ in range(RGB):
            orgImage.append(malloc(orgH, orgW))
        ## 파일 --> 메모리 로딩
        for i in range(orgH):
            for k in range(orgW):
                orgImage[R][i][k] = cvInImage.item(i, k, B)
                orgImage[G][i][k] = cvInImage.item(i, k, G)
                orgImage[B][i][k] = cvInImage.item(i, k, R)
        outH = orgH; outW = orgW
        outImage = []
        for _ in range(RGB):
            outImage.append(malloc(orgH, orgW))
        for rgb in range(RGB):
            for i in range(orgH):
                for k in range(orgW):
                    outImage[rgb][i][k] = orgImage[rgb][i][k]
        displayImageColor()
        messagebox.showinfo("SQL에서 불러오기",filename + '을 성공적으로 SQL에서 불러왔습니다.')
        logSave('downMySQL : ' + filename)
        ####
    subWindow = Toplevel(window)
    subWindow.geometry('300x400')
    ## 스크롤바 나타내기
    frame = Frame(subWindow)
    scrollbar = Scrollbar(frame)
    scrollbar.pack(side='right', fill = 'y')
    listData = Listbox(frame, yscrollcommand=scrollbar.set); listData.pack()
    scrollbar['command']=listData.yview
    frame.pack()
    for fileTup in fileList:
        listData.insert(END, fileTup[1:])
    btnDownLoad = Button(subWindow, text='다운로드', command=downLoad)
    btnDownLoad.pack(padx=10, pady=10)

####### 관리자 메뉴 #######
def adminMenu1():
    global window, canvas, paper, listFrame, listData, conn, cur
    def selectUser() :
        global conn, cur
        def userSel():
            global canvas
            u_id, u_name, u_phone, u_email = txt1.get(), txt3.get(), txt4.get(), txt5.get()
            ###################
            if u_id!=None and u_id!='':
                canvas.destroy()
                canvas = Canvas(window, width=800)
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()  # 빈 트럭 준비
                sql = "SELECT * FROM user_table WHERE u_id='"+u_id+"';"
                cur.execute(sql)
                fileList = cur.fetchall()
                listFrame = Frame(canvas)
                scrollbar = Scrollbar(listFrame)
                listData = Listbox(listFrame, width=600, yscrollcommand=scrollbar.set)
                for fileTup in fileList:
                    listData.insert(END, fileTup[0:])
                print(listData)
                messagebox.showinfo("회원조회성공", u_id + "회원의 정보를 조회하였습니다")
            elif u_name!=None and u_name!='':
                canvas.destroy()
                canvas = Canvas(window, width=800)
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()  # 빈 트럭 준비
                sql = "SELECT * FROM user_table WHERE u_name='"+u_name+"';"
                cur.execute(sql)
                fileList = cur.fetchall()
                listFrame = Frame(canvas)
                scrollbar = Scrollbar(listFrame)
                listData = Listbox(listFrame, width=600, yscrollcommand=scrollbar.set)
                for fileTup in fileList:
                    listData.insert(END, fileTup[0:])
                print(listData)
                messagebox.showinfo("회원조회성공", u_name + "회원의 정보를 조회하였습니다")
            elif u_phone!=None and u_phone!='':
                canvas.destroy()
                canvas = Canvas(window, width=800)
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()  # 빈 트럭 준비
                sql = "SELECT * FROM user_table WHERE u_phone='"+u_phone+"';"
                cur.execute(sql)
                fileList = cur.fetchall()
                listFrame = Frame(canvas)
                scrollbar = Scrollbar(listFrame)
                listData = Listbox(listFrame, width=600, yscrollcommand=scrollbar.set)
                for fileTup in fileList:
                    listData.insert(END, fileTup[0:])
                print(listData)
                messagebox.showinfo("회원조회성공", u_phone + "회원의 정보를 조회하였습니다")
            elif u_email!=None and u_email!='':
                canvas.destroy()
                canvas = Canvas(window, width=800)
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()  # 빈 트럭 준비
                sql = "SELECT * FROM user_table WHERE u_email='"+u_email+"';"
                cur.execute(sql)
                fileList = cur.fetchall()
                listFrame = Frame(canvas)
                scrollbar = Scrollbar(listFrame)
                listData = Listbox(listFrame, width=600, yscrollcommand=scrollbar.set)
                for fileTup in fileList:
                    listData.insert(END, fileTup[0:])
                print(listData)
                messagebox.showinfo("회원조회성공", u_email + "회원의 정보를 조회하였습니다")
            else:
                messagebox.showinfo("회원정보 입력오류", "입력이 잘못되었거나 존재하지 않는 회원입니다")

        def close() :
            subwindow.destroy()

        subwindow = Tk()
        subwindow.geometry('300x200')
        subwindow.title("회원조회")
        lbl1 = Label(subwindow, text="ID");
        lbl1.grid(row=1, column=1)
        txt1 = Entry(subwindow);
        txt1.grid(row=1, column=2)
        lbl3 = Label(subwindow, text="NAME");
        lbl3.grid(row=3, column=1)
        txt3 = Entry(subwindow);
        txt3.grid(row=3, column=2)
        lbl4 = Label(subwindow, text="PHONE");
        lbl4.grid(row=4, column=1)
        txt4 = Entry(subwindow);
        txt4.grid(row=4, column=2)
        lbl5 = Label(subwindow, text="E-MAIL");
        lbl5.grid(row=5, column=1)
        txt5 = Entry(subwindow);
        txt5.grid(row=5, column=2)

        btn1 = Button(subwindow, text="회원정보조회", width=15, command=userSel)
        btn1.grid(row=6, column=1)
        btn2 = Button(subwindow, text="닫기", width=15, command=close)
        btn2.grid(row=6, column=2)

        subwindow.mainloop()

    def addUser() :
        global conn, cur

        def userInsert():
            u_id, u_pw, u_name, u_phone, u_email = txt1.get(), txt2.get(), txt3.get(), txt4.get(), txt5.get()
            ###################
            a: bool = (
                        u_id != None and u_pw != None and u_name != None and u_phone != None and u_email and u_id != '' and u_pw != '' and u_name != '' and u_phone != '' and u_email != '')
            if a:
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()
                sql = "INSERT INTO user_table VALUES('" + u_id + "','" + u_pw + "','" + u_name + "','" + u_phone
                sql += "','" + u_email + "',null,null,null,null,null,null);"
                print(sql)
                cur.execute(sql)
                conn.commit()
                cur.close()
                conn.close()
                messagebox.showinfo("회원입력 성공", u_id + " 유저를 추가하였습니다.")
                adminMenu1()
            else:
                messagebox.showinfo("회원정보 입력오류", "정보를 다시 입력해주세요")

        def close() :
            subwindow.destroy()

        subwindow = Tk()
        subwindow.geometry('300x200')
        subwindow.title("회원가입 정보입력")
        lbl1 = Label(subwindow, text="ID");
        lbl1.grid(row=1, column=1)
        txt1 = Entry(subwindow);
        txt1.grid(row=1, column=2)
        lbl2 = Label(subwindow, text="PASSWORD");
        lbl2.grid(row=2, column=1)
        txt2 = Entry(subwindow);
        txt2.grid(row=2, column=2)
        lbl3 = Label(subwindow, text="NAME");
        lbl3.grid(row=3, column=1)
        txt3 = Entry(subwindow);
        txt3.grid(row=3, column=2)
        lbl4 = Label(subwindow, text="PHONE");
        lbl4.grid(row=4, column=1)
        txt4 = Entry(subwindow);
        txt4.grid(row=4, column=2)
        lbl5 = Label(subwindow, text="E-MAIL");
        lbl5.grid(row=5, column=1)
        txt5 = Entry(subwindow);
        txt5.grid(row=5, column=2)

        btn1 = Button(subwindow, text="회원가입", width=15, command=userInsert)
        btn1.grid(row=6, column=1)
        btn2 = Button(subwindow, text="닫기", width=15, command=close)
        btn2.grid(row=6, column=2)

        subwindow.mainloop()

    def editUser() :
        global conn, cur

        def userUpdate():
            u_id, u_pw, u_name, u_phone, u_email = txt1.get(), txt2.get(), txt3.get(), txt4.get(), txt5.get()
            ###################
            a: bool = (
                    u_id != None and u_pw != None and u_name != None and u_phone != None and u_email and u_id != '' and u_pw != '' and u_name != '' and u_phone != '' and u_email != '')
            if a:
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()
                sql = "UPDATE user_table SET u_pw='" + u_pw + "', u_phone='" + u_phone + "', u_phone='"+u_phone+"' WHERE u_id='"+u_id+"';"
                print(sql)
                cur.execute(sql)
                conn.commit()
                cur.close()
                conn.close()
                messagebox.showinfo("회원정보수정완료", u_id + " 의 정보를 수정하였습니다.")
                adminMenu1()
            elif (u_id != None and u_name != None and u_phone != None and u_email and u_id != '' and u_name != '' and u_phone != '' and u_email != ''):
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()
                sql = "UPDATE user_table SET u_phone='" + u_phone + "', u_email='"+u_email+"' WHERE u_id='"+u_id+"';"
                print(sql)
                cur.execute(sql)
                conn.commit()
                cur.close()
                conn.close()
                messagebox.showinfo("회원정보수정완료", u_id + " 의 정보를 수정하였습니다.")
                adminMenu1()
            elif  (u_id != None and u_name != None and  u_email and u_id != '' and u_name != '' and u_email != ''):
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()
                sql = "UPDATE user_table SET u_email='"+u_email+"' WHERE u_id='"+u_id+"';"
                print(sql)
                cur.execute(sql)
                conn.commit()
                cur.close()
                conn.close()
                messagebox.showinfo("회원정보수정완료", u_id + " 의 정보를 수정하였습니다.")
                adminMenu1()
            elif  (u_id != None and u_name != None and  u_phone and u_id != '' and u_name != '' and u_phone != ''):
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()
                sql = "UPDATE user_table SET u_phone='"+u_phone+"' WHERE u_id='"+u_id+"';"
                print(sql)
                cur.execute(sql)
                conn.commit()
                cur.close()
                conn.close()
                messagebox.showinfo("회원정보수정완료", u_id + " 의 정보를 수정하였습니다.")
                adminMenu1()
            else:
                messagebox.showinfo("회원정보 입력오류", "정보를 다시 입력해주세요")
        def close() :
            subwindow.destroy()

        subwindow = Tk()
        subwindow.geometry('300x200')
        subwindow.title("회원정보수정")
        lbl1 = Label(subwindow, text="ID");
        lbl1.grid(row=1, column=1)
        txt1 = Entry(subwindow);
        txt1.grid(row=1, column=2)
        lbl2 = Label(subwindow, text="PASSWORD");
        lbl2.grid(row=2, column=1)
        txt2 = Entry(subwindow);
        txt2.grid(row=2, column=2)
        lbl3 = Label(subwindow, text="NAME");
        lbl3.grid(row=3, column=1)
        txt3 = Entry(subwindow);
        txt3.grid(row=3, column=2)
        lbl4 = Label(subwindow, text="PHONE");
        lbl4.grid(row=4, column=1)
        txt4 = Entry(subwindow);
        txt4.grid(row=4, column=2)
        lbl5 = Label(subwindow, text="E-MAIL");
        lbl5.grid(row=5, column=1)
        txt5 = Entry(subwindow);
        txt5.grid(row=5, column=2)
        btn1 = Button(subwindow, text="회원정보수정", width=15, command=userUpdate)
        btn1.grid(row=6, column=1)
        btn2 = Button(subwindow, text="닫기", width=15, command=close)
        btn2.grid(row=6, column=2)

        subwindow.mainloop()

    def deleteUser() :
        global conn, cur

        def userDel():
            u_id = txt1.get()
            ###################
            if u_id != None and u_id != '':
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()
                sql = "DELETE FROM user_table WHERE u_id='" + u_id + "';"
                print(sql)
                cur.execute(sql)
                conn.commit()
                cur.close()
                conn.close()
                messagebox.showinfo("회원입력 성공", u_id + " 유저를 삭제하였습니다.")
                adminMenu1()
            else:
                messagebox.showinfo("회원정보 입력오류", "삭제할 회원의 ID를 다시 입력해주세요")
        def close() :
            subwindow.destroy()
        subwindow = Tk()
        subwindow.geometry('300x200')
        subwindow.title("회원삭제")
        lbl1 = Label(subwindow, text="ID");
        lbl1.grid(row=1, column=1)
        txt1 = Entry(subwindow);
        txt1.grid(row=1, column=2)

        btn1 = Button(subwindow, text="회원삭제", width=15, command=userDel)
        btn1.grid(row=6, column=1)
        btn2 = Button(subwindow, text="닫기", width=15, command=close)
        btn2.grid(row=6, column=2)

    canvas.destroy()
    window.geometry('800x280')
    canvas = Canvas(window, width=800)
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()  # 빈 트럭 준비
    sql = "SELECT * FROM user_table"
    cur.execute(sql)
    fileList = cur.fetchall()
    listFrame = Frame(canvas)
    scrollbar = Scrollbar(listFrame)
    listData = Listbox(listFrame, width=600, yscrollcommand=scrollbar.set)
    for fileTup in fileList:
        listData.insert(END, fileTup[0:])
    print(listData)

    scrollbar.pack(side="right", fill="y")
    listData.pack(side="left", fill=BOTH,expand=1)
    scrollbar["command"] = listData.yview
    userlabel = Label(canvas,text="ID   PW   NAME   PHONE   EMAIL   ADDRESS   GRADE   CARD_No   KAKAO   NAVER")
    userlabel.pack(side=TOP)
    listFrame.pack()
    edtFrame = Frame(canvas)
    edtFrame.pack(side=BOTTOM)
    btn0 = Button(edtFrame, text="회원조회", command=selectUser)
    btn0.pack(side=RIGHT, padx=10, pady=10)
    btn1 = Button(edtFrame, text="회원추가", command=addUser)
    btn1.pack(side=RIGHT, padx=10, pady=10)
    btn2 = Button(edtFrame, text="회원수정", command=editUser)
    btn2.pack(side=RIGHT, padx=10, pady=10)
    btn3 = Button(edtFrame, text="회원삭제", command=deleteUser)
    btn3.pack(side=RIGHT, padx=10, pady=10)
    canvas.pack()


def adminMenu2() :
    global window, canvas, paper, listFrame, listData, conn, cur
    canvas.destroy()
    window.geometry('800x300')
    canvas = Canvas(window, width=800)
    conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
    cur = conn.cursor()  # 빈 트럭 준비
    sql = "SELECT * FROM userlog_table"
    cur.execute(sql)
    fileList = cur.fetchall()
    listFrame = Frame(canvas,width=600)
    scrollbar = Scrollbar(listFrame)
    listData = Listbox(listFrame,width=550, yscrollcommand=scrollbar.set)
    for fileTup in fileList:
        listData.insert(END, fileTup[0:])

    print(listData)
    scrollbar["command"] = listData.yview
    userlabel = Label(canvas,anchor='n',text="로그넘버   유저아이디   사진아이디   사진이름   사용날짜   사용기록                                 ")
    userlabel.pack(side=TOP)
    scrollbar.pack(side="right")
    listData.pack(side="left")
    listFrame.pack()
    btn1 = Button(canvas, text="회원목록보기", command=adminMenu1)
    btn1.pack(side=BOTTOM, padx=10, pady=10)
    canvas.pack()

##### 로그인 메뉴 #####
def userLogin() :
    global u_id
    id_input, pw_input = edt1.get(), edt2.get()
    u_id=id_input
    a = userManager.userCheck(id_input, pw_input)
    if a >0 :
        imageProcess(a)

def guestLogin() :
    global u_id
    id_input, pw_input = "guest","guest"
    u_id=id_input
    print("guestlogin")
    imageProcess(3)

##### 영상처리 메뉴 #######
def imageProcess(inputGrade) :
    global window, canvas, paper, inImage, outImage, inH, inW, outH, outW, filename
    global cvInImage, cvOutImage
    userGrade = inputGrade
    window.geometry('512x512')
    canvas.destroy()
    status.pack(side=BOTTOM, fill=X)
    if userGrade == 1 :
        # 0: 실행안함 1: 관리자 2: 일반사용자 3: 게스트
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
        mainMenu.add_cascade(label="밝기/색상조절", menu=valueMenu)
        valueMenu.add_command(label="밝기조절", command=addColor)
        valueMenu.add_command(label="그레이스케일", command=grayColor)
        valueMenu.add_command(label="색상 조절", command=rgbChangeColor)
        geometryMenu = Menu(mainMenu)
        mainMenu.add_cascade(label="기하학 처리", menu=geometryMenu)
        geometryMenu.add_command(label="이동", command=moveColor)
        geometryMenu.add_command(label="축소", command=zoomOutColor)
        geometryMenu.add_command(label="확대(보간법 적용)", command=zoomInColor2)
        geometryMenu.add_command(label="미러링(좌우)", command=mirrorLRColor)
        geometryMenu.add_command(label="미러링(상하)", command=mirrorUDColor)
        #geometryMenu.add_command(label="회전", command=rotationColor)
        AreaMenu = Menu(mainMenu)
        mainMenu.add_cascade(label="특수효과처리", menu=AreaMenu)
        AreaMenu.add_command(label="엠보싱", command=embossColor1)
        AreaMenu.add_command(label="블러링", command=blurrColor)
        AreaMenu.add_command(label="모자이크", command=pixelation)
        AreaMenu.add_command(label="부드럽게", command=smoothColor)
        AreaMenu.add_command(label="샤프닝", command=sharpSelect)
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
        mysqlMenu = Menu(mainMenu)
        mainMenu.add_cascade(label="MySQL", menu=mysqlMenu)
        mysqlMenu.add_command(label="MySQL에 저장", command=upMySQL)
        mysqlMenu.add_command(label="MySQL에서 열기", command=downMySQL)
        adminOnlyMenu = Menu(mainMenu)
        mainMenu.add_cascade(label="관리자메뉴", menu=adminOnlyMenu)
        adminOnlyMenu.add_command(label="회원정보관리", command=adminMenu1)
        adminOnlyMenu.add_command(label="로그보기", command=adminMenu2)
    elif userGrade == 2 :
        # 0: 실행안함 1: 관리자 2: 일반사용자 3: 게스트
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
        mainMenu.add_cascade(label="밝기/색상조절", menu=valueMenu)
        valueMenu.add_command(label="밝기조절", command=addColor)
        valueMenu.add_command(label="그레이스케일", command=grayColor)
        valueMenu.add_command(label="색상 조절", command=rgbChangeColor)
        geometryMenu = Menu(mainMenu)
        mainMenu.add_cascade(label="기하학 처리", menu=geometryMenu)
        geometryMenu.add_command(label="이동", command=moveColor)
        geometryMenu.add_command(label="축소", command=zoomOutColor)
        geometryMenu.add_command(label="확대(보간법 적용)", command=zoomInColor2)
        geometryMenu.add_command(label="미러링(좌우)", command=mirrorLRColor)
        geometryMenu.add_command(label="미러링(상하)", command=mirrorUDColor)
        # geometryMenu.add_command(label="회전", command=rotationColor)
        AreaMenu = Menu(mainMenu)
        mainMenu.add_cascade(label="특수효과처리", menu=AreaMenu)
        AreaMenu.add_command(label="엠보싱", command=embossColor1)
        AreaMenu.add_command(label="블러링", command=blurrColor)
        AreaMenu.add_command(label="모자이크", command=pixelation)
        AreaMenu.add_command(label="부드럽게", command=smoothColor)
        AreaMenu.add_command(label="샤프닝", command=sharpSelect)
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
        mysqlMenu = Menu(mainMenu)
        mainMenu.add_cascade(label="MySQL", menu=mysqlMenu)
        mysqlMenu.add_command(label="MySQL에 저장", command=upMySQL)
        mysqlMenu.add_command(label="MySQL에서 열기", command=downMySQL)
    else :
        # 0: 실행안함 1: 관리자 2: 일반사용자 3: 게스트
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
        mainMenu.add_cascade(label="밝기/색상조절", menu=valueMenu)
        valueMenu.add_command(label="밝기조절", command=addColor)
        valueMenu.add_command(label="그레이스케일", command=grayColor)
        valueMenu.add_command(label="색상 조절", command=rgbChangeColor)
        geometryMenu = Menu(mainMenu)
        mainMenu.add_cascade(label="기하학 처리", menu=geometryMenu)
        geometryMenu.add_command(label="이동", command=moveColor)
        geometryMenu.add_command(label="축소", command=zoomOutColor)
        geometryMenu.add_command(label="확대(보간법 적용)", command=zoomInColor2)
        geometryMenu.add_command(label="미러링(좌우)", command=mirrorLRColor)
        geometryMenu.add_command(label="미러링(상하)", command=mirrorUDColor)
        # geometryMenu.add_command(label="회전", command=rotationColor)
    ######################

    window.mainloop()


## 전역 변수부
#영상처리 관련
window, canvas, paper = None, None, None
inImage, outImage, orgImage= [], [] , [];
inH, inW, outH, outW, orgH, orgW= 0,0,0,0,0,0
xPoint, yPoint = 0,0
cvInImage, cvOutImage = None, None
filename = ''
RGB,R, G, B= 3, 0, 1, 2
listData = []
# DB관련
conn, cur = None, None
IP = '127.0.0.1'
USER = 'root'
PASSWORD = '1234'
DB = 'photo_db'
fileList = None
u_id, p_id = None,None

if __name__ == '__main__' :
    window = Tk()
    window.title('Project 1 . Photo Edit Program Ver 0.05')
    window.geometry('200x200')
    canvas = Canvas(window)
    status = Label(window, text='이미지정보:', bd=1, relief=SUNKEN, anchor=W)

    edt1 = Entry(canvas, width = 10)
    edt2 = Entry(canvas, width = 10)
    btn1 = Button(canvas, text='로그인', command=userLogin)
    btn2 = Button(canvas, text='게스트로 로그인', command=guestLogin)
    btn3 = Button(canvas, text='회원가입', command=userManager.userJoin)
    btn4 = Button(canvas, text='ID찾기', command=userManager.findID)
    btn5 = Button(canvas, text='비밀번호찾기', command=userManager.findPW)

    edt1.pack(); edt2.pack()
    btn1.pack(); btn2.pack()
    btn3.pack(); btn4.pack()
    btn5.pack();
    canvas.pack(side=TOP)
    window.mainloop()



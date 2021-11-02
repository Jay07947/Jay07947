import pygame
import random
from time import sleep
from pygame import font

WHITE =(255, 255, 255)
pad_width=750
pad_height=370
background_width = 750
aircraft_width=40
aircraft_height=30

bat_width = 40
bat_height = 40
fire_width = 40
fire_height = 30
heart_w = 40
heart_h = 40
font = None
enermy = []

# 공통 함수
# 적을 맞춘 개수 계산
def writeScore(count) :
    global gamepad, font
    text = font.render('Score :'+str(count),True,(255,255,255))
    gamepad.blit(text,(10,0))

# 적이 화면 아래로 통과한 개수
def writePassed(count):
    global gamepad, font
    text = font.render('Life:'+str(5-count),True,(255,0,255))
    gamepad.blit(text, (330, 0))

# 게임 시간 표시
def writeTime(count):
    global gamepad, font
    text = font.render('Time:'+str(count),True,(255,255,0))
    gamepad.blit(text, (630, 0))

# 만든 사람 이름 표시
def writename():
    global gamepad, font
    text = font.render('MADE BY JAY07947',True,(255,255,255))
    gamepad.blit(text, (470, 320))

# 화면에 글씨 보이게 하기
def dispMessage(text):
    global gamepad
    textfont = pygame.font.Font('freesansbold.ttf', 80)
    text = textfont.render(text, True, (255,0,0))
    textpos = text.get_rect()
    textpos.center = (pad_width/2, pad_height/2)
    gamepad.blit(text,textpos)
    pygame.display.update()
    sleep(4)

# 게임오버 메시지 보이기
def gameover():
    global gamepad, record
    dispMessage('Game Over')
    initGame()

# 게임에 등장하는 객체를 드로잉
def drawObject(obj, x, y):
    global gamepad
    gamepad.blit(obj, (x, y))

# 초기화 함수
def initGame():
    global gamepad, aircraft, clock, background1, background2, font, heart, fire2, enermy
    global bat, fire, bullet, boom, shot_sound, explosion_sound, start_time, elapsed_time, fire_sound

    start_time = 0
    elapsed_time = 0
    pygame.init()
    enermy = []
    font = pygame.font.SysFont("arial", 30, True, False)
    gamepad = pygame.display.set_mode((pad_width, pad_height))
    pygame.display.set_caption('PyShooting ')
    aircraft = pygame.image.load('images/shooter1.png')
    background1 = pygame.image.load('images/background3.png')
    background2 = background1.copy()
    bat = pygame.image.load('images/bat_1.png')
    fire = pygame.image.load('images/fire1.png')
    fire2 = pygame.image.load('images/fire2.png')
    boom = pygame.image.load('images/crash1.png')
    bullet = pygame.image.load('images/bullet.png')
    heart = pygame.image.load('images/heart.png')
    enermy.append(pygame.image.load('images/bat_1.png'))
    enermy.append(pygame.image.load('images/bat_2.png'))
    enermy.append(pygame.image.load('images/bat_3.png'))
    shot_sound = pygame.mixer.Sound('shot.wav')
    fire_sound = pygame.mixer.Sound('big.wav')
    explosion_sound = pygame.mixer.Sound('explosion.wav')
    pygame.mixer.music.load('background.wav')
    pygame.mixer.music.play(-1)

    clock = pygame.time.Clock()
    runGame()

######################################

# 게임 실행 메인 함수
def runGame():
    global gamepad, aircraft , clock, background1, background2, enermy
    global bat, fires, bullet, boom, enemypassed, heart, isheart, fire2, fire_sound

    enemypassed = 0     # 적이 지나간 개수
    shot_count = 0      # 적을 맞춘 개수
    start_time = pygame.time.get_ticks() # 게임 시작 시각 지정

    # 무기 좌표를 위한 리스트 자료
    bullet_xy=[]

    # 플레이어 초기 위치 (x, y) 설정
    x=pad_width*0.05
    y=pad_height*0.8
    y_change=0

    # 배경 위치 지정
    background1_x=0
    background2_x = background_width
    gamepad.fill(WHITE)

    # 적(박쥐) 초기위치
    bat_x = pad_width
    bat_y=random.randrange(0,pad_height-bat_height/2)
    bat_speed=3

    # 불 초기 위치
    fire_x=pad_width
    fire2_x = pad_width+100
    fire_y=random.randrange(0,pad_height-fire_height/2)
    fire2_y=random.randrange(30,pad_height-fire_height/2)
    fire_speed=random.randrange(3,15)
    fire2_speed = random.randrange(10, 20)

    # 하트 초기 위치
    heart_x=pad_width
    heart_y=random.randrange(0,pad_height-heart_h/2)
    heart_speed=2
    isheart = False

    crashed = False
    while not crashed:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    y_change=-5

                elif event.key == pygame.K_DOWN:
                    y_change=5

                elif event.key == pygame.K_LCTRL:
                    bullet_x=x+aircraft_width
                    bullet_y=y+aircraft_height/2
                    bullet_xy.append([bullet_x,bullet_y])
                    pygame.mixer.Sound.play(shot_sound)

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP or event.key==pygame.K_DOWN:
                    y_change=0

        # 시간 조건 설정
        elapsed_time = (pygame.time.get_ticks() - start_time) / 1000
        # 스테이지 2 변수 초기화
        if int(elapsed_time)==10 :
            gamepad.fill((255,255,255))
            dispMessage('stage 2')
            background1 = pygame.image.load('images/background2.png')
            background2 = background1.copy()
            enermy = []
            enermy.append(pygame.image.load('images/seagull1.png'))
            enermy.append(pygame.image.load('images/seagull2.png'))
            enermy.append(pygame.image.load('images/seagull3.png'))
            # 적(박쥐) 초기위치
            bat_x =  pad_width
            bat_y = random.randrange(0, pad_height - bat_height / 2)
            bat_speed = 3
            # 불 초기 위치
            fire_x = pad_width
            fire2_x = pad_width + 100
            fire_y = random.randrange(0, pad_height - fire_height / 2)
            fire2_y = random.randrange(30, pad_height - fire_height / 2)
            fire_speed = random.randrange(3, 15)
            fire2_speed = random.randrange(10, 20)
            # 하트 초기 위치
            heart_x = pad_width
            heart_y = random.randrange(0, pad_height - heart_h / 2)
            heart_speed = 2
            isheart = False
        # 배경 돌아가게 하는 부분
        background1_x -= 2
        background2_x -= 2
        if background1_x == -background_width:
            background1_x = background_width
        if background2_x == -background_width:
            background2_x = background_width
        drawObject(background1, background1_x, 0)
        drawObject(background2, background2_x, 0)


        # 화면에 띄우기
        drawObject(aircraft,x,y)
        drawObject(fire, fire_x, fire_y)
        drawObject(bat, bat_x, bat_y)
        drawObject(heart, heart_x, heart_y)

        if int(elapsed_time)>10 :
            drawObject(fire2, fire2_x, fire2_y)

        if int(elapsed_time)>0 and int(elapsed_time)%11==0 :
            isheart = True

        if isheart :
            heart_x -= heart_speed

        if len(bullet_xy) != 0:
            for bx, by in bullet_xy:
                drawObject(bullet, bx, by)

        writename()
        writePassed(enemypassed)
        writeTime(str(int(elapsed_time)))
        writeScore(shot_count)


        # 플레이어/적의 위치를 재조정
        y+=y_change
        if y<0:
            y=0
        elif y>pad_height-aircraft_height:
            y=pad_height-aircraft_height
        fire_x -= fire_speed
        fire2_x -= fire2_speed
        fire2_y -= 1
        bat_x -= bat_speed


        # 5번 적과 부딫히거나 박쥐가 지나가면 게임오버
        if bat_x < 0 :
            bat_x = pad_width
            bat_y = random.randrange(0, pad_height - bat_height)
            enemypassed += 1
        if fire_x < 0 :
            fire_x = pad_width
            fire_y = random.randrange(0, pad_height - fire_height / 2)
        if fire2_x < 0 :
            fire2_x = pad_width+80
            fire2_y = random.randrange(30, pad_height - fire_height / 2 )
        if heart_x < 0 :
            heart_x = pad_width
            heart_y = random.randrange(0, pad_height - heart_h / 2)


        if x + aircraft_width > bat_x: #적이 플레이어의 x 위치에 접근 했을 때
            if (y < bat_y and bat_y < y+aircraft_height) or \
               (y < bat_y + bat_height and bat_y + bat_height < y + aircraft_height ):
                    pygame.mixer.Sound.play(explosion_sound)
                    drawObject(boom, bat_x, bat_y)
                    bat_x = pad_width
                    bat_y = random.randrange(0, pad_height - bat_height / 2)
                    enemypassed += 1

        if x + aircraft_width > fire_x:
            if (y < fire_y and fire_y < y+aircraft_height) or \
               (y < fire_y + fire_height and fire_y + fire_height < y + aircraft_height ):
                    pygame.mixer.Sound.play(fire_sound)
                    fire_x = pad_width
                    fire_y = random.randrange(0, pad_height - fire_height / 2)
                    enemypassed += 1

        if x + aircraft_width > fire2_x:
            if (y < fire2_y and fire2_y < y+aircraft_height) or \
               (y < fire2_y + fire_height and fire2_y + fire_height < y + aircraft_height ):
                    pygame.mixer.Sound.play(fire_sound)
                    fire2_x = pad_width+130
                    fire2_y = random.randrange(30, pad_height - fire_height / 2)
                    enemypassed += 1

                # 하트를 먹으면 Life +1
        if heart_x < 0 :
            heart_x = pad_width
            heart_y = random.randrange(0, pad_height - heart_h)

        if x + aircraft_width >= heart_x:
            if (y> heart_y and y<heart_y+heart_h) or \
               (y+aircraft_height>=heart_y and y<heart_y+heart_h):
                    isheart = False
                    heart_x = pad_width
                    heart_y = random.randrange(0, pad_height - heart_h)
                    enemypassed -= 1

        if enemypassed == 5:
            gameover()
        # 총알 발사시 조건
        if len(bullet_xy) != 0:
            for i, bxy in enumerate(bullet_xy):
                bxy[0] +=15
                bullet_xy[i][0]=bxy[0]
                # 적이 총에 맞았을 때
                if bxy[0]>bat_x:
                    if bxy[1]>bat_y and bxy[1]<bat_y+bat_height:
                        bullet_xy.remove(bxy)
                        shot_count += 1
                        drawObject(boom, bat_x, bat_y)
                        bat_x = pad_width
                        bat_y = random.randrange(0, pad_height - bat_height)
                        if shot_count%3 ==1:
                            bat = enermy[0]
                        elif shot_count%3 ==2:
                            bat = enermy[1]
                            if bat_speed < 10 :
                                bat_speed += 1
                            else :
                                bat_speed = 10
                        else :
                            bat = enermy[2]
                if bxy[0]>=pad_width:
                    try:
                        bullet_xy.remove(bxy)
                    except:
                        pass

        pygame.display.update()
        clock.tick(60)

    pygame.quit()
    quit()


if __name__=='__main__':
    initGame()
import pymysql
import random
import string
from tkinter import *
from tkinter import messagebox

def userCheck(id,pw) :
    id_input = id
    pw_input = pw
    print(id_input, pw_input)
    try:
        if id_input =='admin' :
            conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
            cur = conn.cursor()
            cur.execute("SELECT u_pw FROM user_table WHERE u_id ='" + id_input + "'")
            u_pw = cur.fetchone()
            if u_pw[0] == pw_input:  # 튜플로 가져오기 때문에 인덱스로 값 가져오기
                messagebox.showinfo(title="관리자 로그인", message="관리자 모드로 실행합니다.")
                return 1
            else:
                messagebox.showinfo(title="로그인 오류", message="비밀번호를 잘못 입력하셨습니다.")
                return -1
        else :
            conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
            cur = conn.cursor()
            cur.execute("SELECT u_pw FROM user_table WHERE u_id ='"+id_input+"'")
            u_pw = cur.fetchone()
            if u_pw[0] == pw_input : #튜플로 가져오기 때문에 인덱스로 값 가져오기
                return 2
            else:
                messagebox.showinfo(title="로그인 오류", message="비밀번호를 잘못 입력하셨습니다.")
    except:
        messagebox.showinfo(title="로그인 오류", message="ID를 잘못 입력하셨습니다.")
        return -1

def findID() :
    global conn, cur
    def showID():
        try :
            u_name, u_phone = txt1.get(), txt2.get()
            ###################
            a: bool = (
                        u_name != None and u_phone != None and u_name != '' and u_phone != '')
            if a:
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()
                sql = "SELECT u_id FROM user_table WHERE u_name='" + u_name + "' and u_phone='" + u_phone+"';"
                cur.execute(sql)
                u_id = cur.fetchone()
                print('id조회성공')
                messagebox.showinfo("ID찾기 성공", str(u_name) + "님의 ID는  " +str(u_id[0]) +"  입니다. 다시 로그인 해주시길 바랍니다.")
                cur.close()
                conn.close()
            else:
                messagebox.showinfo("정보 입력오류", "정보를 모두 입력해주세요")
        except :
            messagebox.showinfo("가입정보없음", "가입되지 않은 회원입니다. 회원가입해주세요")

    subwindow = Tk()
    subwindow.geometry('300x200')
    subwindow.title("ID 찾기 정보입력")
    lbl1 = Label(subwindow, text="NAME"); lbl1.grid(row=2, column=1)
    txt1 = Entry(subwindow); txt1.grid(row=2, column=2)
    lbl2 = Label(subwindow, text="PHONE"); lbl2.grid(row=3, column=1)
    txt2 = Entry(subwindow); txt2.grid(row=3, column=2)

    btn = Button(subwindow, text="ID찾기", width=15, command=showID); btn.grid(row=5, column=2)

    subwindow.mainloop()

def findPW():
    global conn, cur
    def sendPW():
        try :
            u_id, u_phone = txt1.get(), txt2.get()
            ###################
            a: bool = (
                        u_id != None and u_phone != None and u_id != '' and u_phone != '')
            if a:
                conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
                cur = conn.cursor()
                sql = "SELECT u_id FROM user_table WHERE u_id='" + u_id + "' and u_phone='" + u_phone+"';"
                cur.execute(sql)
                u_id = cur.fetchone()
                print('정보 일치')
                tempPW = "".join([random.choice(string.ascii_letters) for _ in range(8)])
                print(tempPW)
                print("UPDATE user_table SET u_pw='"+tempPW+"' WHERE u_id='"+u_id[0]+"';")
                sql1 = "UPDATE user_table SET u_pw='"+tempPW+"' WHERE u_id='"+u_id[0]+"';"

                cur.execute(sql1)
                conn.commit()
                messagebox.showinfo("임시비밀번호 발송","임시비밀번호를 발급하였습니다. 다시 로그인 바랍니다. \n 임시비밀번호 : "+tempPW)
                cur.close()
                conn.close()
            else:
                messagebox.showinfo("정보 입력오류", "정보를 모두 입력해주세요")
        except :
            messagebox.showinfo("가입정보없음", "가입되지 않은 회원입니다. 회원가입해주세요")

    subwindow = Tk()
    subwindow.geometry('300x200')
    subwindow.title("ID 찾기 정보입력")
    lbl1 = Label(subwindow, text="ID"); lbl1.grid(row=2, column=1)
    txt1 = Entry(subwindow); txt1.grid(row=2, column=2)
    lbl2 = Label(subwindow, text="PHONE"); lbl2.grid(row=3, column=1)
    txt2 = Entry(subwindow); txt2.grid(row=3, column=2)

    btn = Button(subwindow, text="PASSWORD 찾기", width=15, command=sendPW); btn.grid(row=5, column=2)

    subwindow.mainloop()

def userJoin() :
    global conn, cur

    def userInsert() :
        u_id, u_pw, u_name, u_phone, u_email = txt1.get(), txt2.get(), txt3.get(), txt4.get(), txt5.get()
        ###################
        a : bool = (u_id != None and u_pw!= None and u_name!= None and u_phone!= None and u_email and u_id != '' and u_pw!= '' and u_name!= '' and u_phone!= '' and u_email!= '')
        if a :
            conn = pymysql.connect(host=IP, user=USER, password=PASSWORD, db=DB, charset='utf8')
            cur = conn.cursor()
            sql = "INSERT INTO user_table VALUES('"+u_id+"','"+ u_pw+"','"+u_name+"','"+u_phone
            sql += "','"+u_email+"',null,null,null,null,null,null);"
            print(sql)
            cur.execute(sql)
            conn.commit()
            cur.close()
            conn.close()
            messagebox.showinfo("회원가입 성공",u_id+"님 가입을 축하드립니다. 다시 로그인 해주시길 바랍니다.")
        else:
            messagebox.showinfo("회원정보 입력오류","정보를 다시 입력해주세요")

    subwindow = Tk()
    subwindow.geometry('300x200')
    subwindow.title("회원가입 정보입력")
    lbl1 = Label(subwindow, text="ID"); lbl1.grid(row=1, column=1)
    txt1 = Entry(subwindow); txt1.grid(row=1, column=2)
    lbl2 = Label(subwindow, text="PASSWORD"); lbl2.grid(row=2, column=1)
    txt2 = Entry(subwindow); txt2.grid(row=2, column=2)
    lbl3 = Label(subwindow, text="NAME"); lbl3.grid(row=3, column=1)
    txt3 = Entry(subwindow); txt3.grid(row=3, column=2)
    lbl4 = Label(subwindow, text="PHONE"); lbl4.grid(row=4, column=1)
    txt4 = Entry(subwindow); txt4.grid(row=4, column=2)
    lbl5 = Label(subwindow, text="E-MAIL"); lbl5.grid(row=5, column=1)
    txt5 = Entry(subwindow); txt5.grid(row=5, column=2)

    btn = Button(subwindow, text="회원가입", width=15 ,command=userInsert)
    btn.grid(row=6, column=2)

    subwindow.mainloop()


## 전역 변수부
# DB관련
conn, cur = None, None
IP = '127.0.0.1'
USER = 'root'
PASSWORD = '1234'
DB = 'photo_db'
listFrame = None
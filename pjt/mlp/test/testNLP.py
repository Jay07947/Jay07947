import json
import pandas as pd
import numpy as np
# from konlpy.tag import Komoran
# from konlpy.tag import Twitter
from time import time
import pickle
import os
# 네이버 영화리뷰 json 데이터 불러오기
# {'date': '15.08.26',
#  'movie_id': 92575,
#  'rating': 4,
#  'review': '종합 평점은 4점 드립니다.'}
with open('C:\Users\Sunny\Desktop\nsmc-master\synopses.json') as data_file:
    data = json.load(data_file)
#id	document	label
with open("C:\Users\Sunny\Desktop\nsmc-master\ratings_test.txt", 'r') as f_x:
    x = f_x.read().splitlines()
with open("C:\Users\Sunny\Desktop\nsmc-master\ratings_train.txt", 'r') as f_y:
    y = f_y.read().splitlines()

# ‘리뷰’, ‘평점’에 대한 데이터만을 가져와 데이터프레임 형식으로 저장
df_movie = pd.DataFrame({
    "rating": [data[i]['rating'] for i in range(len(data))],
    "review": [data[i]['review'] for i in range(len(data))],
})
#‘평점’은 범수 범위에 따라서 ‘NEG’, ‘NEU’, ‘POS’ 세 가지로 분류되므로 해당 평점에 대해서 분류된 새로운 컬럼을 생성
emotion_class = [
    "POS" if df.iloc[i]['rating'] >= 8
    else
    "NEU" if df.iloc[i]['rating'] >= 4
    else
    "NEG"
    for i in range(df.shape[0])
]
df_movie["class"] = emotion_class


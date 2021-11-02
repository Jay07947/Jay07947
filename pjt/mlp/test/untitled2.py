import json
import os
from pprint import pprint
from konlpy.tag import Okt
import nltk

okt = Okt()


def read_data(filename):
    with open(filename, 'r',encoding='UTF-8') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
        # txt 파일의 헤더(id document label)는 제외하기
        data = data[1:]
    return data

def tokenize(doc):
    # norm은 정규화, stem은 근어로 표시하기를 나타냄
    return ['/'.join(t) for t in okt.pos(doc, norm=True, stem=True)]

########################################################################

train_data = read_data('C:/jaesunLee/nsmc-master (2)/nsmc-master/ratings_train.txt')
test_data = read_data('C:/jaesunLee/nsmc-master (2)/nsmc-master/ratings_test.txt')
print(len(train_data))
print(len(test_data))


#이미 태깅이 완료된 train_docs.json 파일이 존재하면 반복하지 않도록
if os.path.isfile('train_docs.json'):
    with open('train_docs.json') as f:
        train_docs = json.load(f)
    with open('test_docs.json') as f:
        test_docs = json.load(f)
        
# 파일 없으면 생성
else:
    print('else')
    train_docs = [(tokenize(row[1]), row[2]) for row in train_data]
    print('tok')
    test_docs = [(tokenize(row[1]), row[2]) for row in test_data]
    print(train_docs[0])
    print(test_docs[0])
    # JSON 파일로 저장
    with open('train_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(train_docs, make_file, ensure_ascii=False, indent="\t")
        print('3')

    with open('test_docs.json', 'w', encoding="utf-8") as make_file:
        json.dump(test_docs, make_file, ensure_ascii=False, indent="\t")
        print('4')


# 예쁘게(?) 출력하기 위해서 pprint 라이브러리 사용
pprint(train_docs[0])

#분석한 데이터의 토큰(문자열을 분석을 위한 작은 단위)의 갯수를 확인
tokens = [t for d in train_docs for t in d[0]]
print(len(tokens))

import nltk
text = nltk.Text(tokens, name='NMSC')

# 전체 토큰의 개수
print(len(text.tokens))

# 중복을 제외한 토큰의 개수
print(len(set(text.tokens)))            

# 출현 빈도가 높은 상위 토큰 10개
pprint(text.vocab().most_common(10))

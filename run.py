import numpy as np
import cv2
import utils

FILE_NAME = "trained.npz"

# 각 글자의 (1 x 400) 데이터와 정답(0 ~ 9, +, -, *)
with np.load(FILE_NAME) as data:
    train = data['train']
    train_labels = data['train_labels']

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

def check(test, train, train_labels):
    # 가장 가까운 k개의 글자를 찾아, 어떤 숫자에 해당하는지 찾음(테스트 데이터 개수에 따라서 조절)
    # 이 예제에서는 같은 숫자의 경우 이미지가 모두 같기 때문에 가장 가까운 이미지를 하나만 찾아도 분류가 가능
    ret, result, neighbours, dist = knn.findNearst(test, k=1)
    return result

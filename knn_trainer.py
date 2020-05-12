import os
import cv2
import numpy as np

file_names = list(range(0, 13))
train = []
train_labels = []

# 0 ~ 12까지의 폴더
for file_name in file_names:
    path = './training_data/' + str(file_name) + '/'
    file_count = len(next(os.walk(path))[2])
    # 각 폴더에 있는 모든 이미지 파일에 접근
    for i in range(1, file_count + 1):
        img = cv2.imread(path + str(i) + '.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        train.append(gray)
        train_labels.append(file_name)

x = np.array(train)
# 학습을 시키기 위해 1차원 배열로 변경(20 x 20이기 때문에 400)
train = x[:, :].reshape(-1, 400).astype(np.float32)
train_labels = np.array(train_labels)[:, np.newaxis]

# print(train.shape)
# print(train_labels.shape)
# print(train_labels)

np.savez("trained.npz", train=train, train_labels=train_labels)
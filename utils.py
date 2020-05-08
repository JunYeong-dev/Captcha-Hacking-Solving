import cv2
import numpy as np

# 이미지 내의 문자 형태가 거의 동일하므로, 데이터를 조금만 수집해도 됨
# 가능한 12개의 문자 : 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, +, -, x
# 색상 추출기를 활용하여, 각 문자의 색상이 어떻게 구성되어 있는지 확인
# 파란색 : RGB에서 B값이 항상 FF
# 초록색 : RGB에서 G값이 항상 FF
# 빨간색 : RGB에서 R값이 항상 FF
# 파란색 & 초록색 : RGB에서 R값이 항상 AA 이하
# 파란색 & 빨간색 : RGB에서 G값이 항상 AA 이하
# 초록색 & 빨간색 : RGB에서 B값이 항상 AA 이하

BLUE = 0
GREEN = 1
RED = 2

# 특정한 색상의 모든 단어가 포함된 이미지를 추출
# color이 blue일 경우
def get_chars(image, color):
    # other_1 : green, other_2 : red
    other_1 = (color + 1) % 3
    other_2 = (color + 2) % 3

    # 255는 FF, 즉 색이 섞이지 않은 것
    # green의 숫자와 기호의 RGB를 0, 0, 0으로 변경
    c = image[:, :, other_1] == 255
    image[c] = [0, 0, 0]
    # red의 숫자와 기호의 RGB를 0, 0, 0으로 변경
    c = image[:, :, other_2] == 255
    image[c] = [0, 0, 0]
    # 170은 AA, 즉 색이 섞여 있는 것
    # green과 red가 섞여있는 숫자의 RGB를 0, 0, 0으로 변경
    c = image[:, :, color] < 170
    image[c] = [0, 0, 0]
    # blue의 숫자와 기호의 RGB를 255, 255, 255로 변경
    c = image[:, :, color] != 0
    image[c] = [255, 255, 255]

    return image
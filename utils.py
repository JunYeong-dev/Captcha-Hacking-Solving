import cv2
import numpy as np
import re

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

# 전체 이미지에서 왼쪽부터 단어별로 이미지를 추출
def extract_chars(image):
    chars = []
    colors = [BLUE, GREEN, RED]
    for color in colors:
        image_from_one_color = get_chars(image.copy(), color)
        image_gray = cv2.cvtColor(image_from_one_color, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(image_gray, 127, 255, 0)
        # RETR_EXTERNAL 옵션으로 숫자의 외각을 기준으로 분리
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]

        for contour in contours:
            # 추출된 이미지 크기가 50 이상인 경우만 실제 문자 데이터인 것으로 파악
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, width, height = cv2.boundingRect(contour)
                roi = image_gray[y:y + height, x:x + width]
                chars.append((x, roi))

    # x축 기준으로 정렬
    chars = sorted(chars, key=lambda char: char[0])
    return chars

# 이미지를 (20 x 20) 크기로 통일
def resize20(image):
    resized = cv2.resize(image, (20, 20))
    return resized.reshape(-1, 400).astype(np.float32)

# 수식 정제
# eval() 함수를 이용하여 계산을 하는데(eval(7 + 5) => 12) 001 + 7 등과 같이 앞에
# 0이 있으면 에러가 발생할 수 있기 때문에 수식을 정제해 줘야 함
def remove_first_0(string):
    temp = []
    for i in string:
        if i == '+' or i == '-' or i == '*':
            temp.append(i)
    split = re.split('\*|\+|\-', string)
    i = 0
    temp_count = 0
    result = ""
    for a in split:
        # lstrip() : 왼쪽부터 해당하는 문자열을 지워줌
        a = a.lstrip('0')
        # 만약 000 과 같은 경우는 전부 지워져서 빈 문자열이 되기 때문에 0을 따로 넣어줌
        if a == '':
            a = '0'
        result += a
        if i < len(split) - 1:
            result += temp[temp_count]
            temp_count = temp_count + 1
        i = i + 1
    return result
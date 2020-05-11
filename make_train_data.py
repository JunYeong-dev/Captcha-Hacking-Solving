import os
import cv2
import utils

# 0 ~ 9 : 숫자, 10 : +, 11 : -, 12 : *
# training_data 폴더 생성 및 그 내부에 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 폴더 생성
image = cv2.imread("3.png")
# 색상별로 숫자, 기호 이미지를 추출해 chars에 넣음
chars = utils.extract_chars(image)

for char in chars:
    # 먼저 각각의 이미지를 보여줌
    cv2.imshow('Image', char[1])
    # 사용자로 출력된 이미지와 같은 값을 입력 받음
    input = cv2.waitKey(0)
    resized = cv2.resize(char[1], (20, 20))

    # 48 : 0의 아스키 코드, 57 : 9의 아스키 코드
    if input >= 48 and input <= 57:
        name = str(input - 48)
        # os.walk('경로').next()[0] : 디렉토리 경로
        # os.walk('경로').next()[1] : 디렉토리 내의 디렉토리 개수
        # os.walk('경로').next()[2] : 디렉토리 내의 파일 개수
        file_count = len(next(os.walk('./training_data/' + name + '/'))[2])
        # 해당 폴더에 저장
        cv2.imwrite('./training_data/' + str(input - 48) + '/' + str(file_count + 1) + '.png', resized)
    # a : +, b : -, c : * 로 입력; ord() : 문자의 아스키 코드 값을 돌려주는 함수
    elif input == ord('a') or input == ord('b') or input == ord('c'):
        name = str(input - ord('a') + 10)
        file_count = len(next(os.walk('./training_data/' + name + '/'))[2])
        cv2.imwrite('./training_data/' + name + '/' + str(file_count + 1) + '.png', resized)

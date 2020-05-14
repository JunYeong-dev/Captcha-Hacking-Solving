import numpy as np
import cv2
import utils
import requests
import shutil
import time

FILE_NAME = "trained.npz"

# 각 글자의 (1 x 400) 데이터와 정답(0 ~ 9, +, -, *)
with np.load(FILE_NAME) as data:
    train = data['train']
    train_labels = data['train_labels']

knn = cv2.ml.KNearest_create()
knn.train(train, cv2.ml.ROW_SAMPLE, train_labels)

def check(test):
    # 가장 가까운 k개의 글자를 찾아, 어떤 숫자에 해당하는지 찾음(테스트 데이터 개수에 따라서 조절)
    # 이 예제에서는 같은 숫자의 경우 이미지가 모두 같기 때문에 가장 가까운 이미지를 하나만 찾아도 분류가 가능
    ret, result, neighbours, dist = knn.findNearest(test, k=1)
    return result

def get_result(file_name):
    image = cv2.imread(file_name)
    # 각각의 이미지를 chars변수에 담음
    chars = utils.extract_chars(image)
    result_string = ""

    for char in chars:
        # 하나하나의 숫자 혹은 기호
        matched = check(utils.resize20(char[1]))

        # 숫자
        if matched < 10:
            result_string += str(int(matched))
            continue
        # 기호
        if matched == 10:
            matched = '+'
        elif matched == 11:
            matched = '-'
        elif matched == 12:
            matched = '*'

        result_string += matched

    return result_string

host = "http://localhost:10000"
# start를 눌렀을때 접속하게 되는 url
url = '/start'

# target_images 라는 폴더 생성
with requests.Session() as s:
    # 첫 실행시은 단순 접속이기 때문에 답을 빈 문자열로 전송
    answer = ''
    # 총 20문제를 풀어야 하기 때문에
    for i in range(0, 20):
        # 한문제를 푸는데 걸리는 시간을 알아보기 위해 설정
        start_time = time.time()
        params = {'ans': answer}

        # 정답을 파라미터에 달아서 전송하여, 이미지 경로를 받아옴
        response = s.post(host + url, params)
        print('Server Return:' + response.text)
        # 가장 처음은 start버튼을 눌렀을 때기 때문에 따로 설정을 해줘야함
        if i == 0:
            returned = response.text
            image_url = host + returned
            url = '/check'
        else:
            returned = response.json()
            image_url = host + returned['url']
        print('Problem ' + str(i) + ': ' + image_url)

        # 특정한 폴더에 문제 이미지 파일을 다운로드
        response = s.get(image_url, stream=True)
        target_image = './target_images/' + str(i) + '.png'
        with open(target_image, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        del response

        # 다운로드 받은 이미지 파일을 분석하여 답을 도출
        # 이미지에서 문자열을 추출
        answer_string = get_result(target_image)
        print('String: ' + answer_string)
        # 추출한 문자열을 정제
        answer_string = utils.remove_first_0(answer_string)
        answer = str(eval(answer_string))
        print('Answer: ' + answer)
        print("--- %s seconds ---" % (time.time() - start_time))
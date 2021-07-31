# 원본: https://github.com/Kazuhito00/Tokyo2020-Pictogram-using-MediaPipe
# 수정: dplus1016@gmail.com (울산경의고 정보교사 안득하)
# 참조(필수): https://google.github.io/mediapipe/solutions/pose.html
# 종료 key: q

import copy
import math
import cv2 as cv
import numpy as np
import mediapipe as mp
from collections import deque

class CvFpsCalc(object):
    def __init__(self, buffer_len=1):
        self._start_tick = cv.getTickCount()
        self._freq = 1000.0 / cv.getTickFrequency()
        self._difftimes = deque(maxlen=buffer_len)

    def get(self):
        current_tick = cv.getTickCount()
        different_time = (current_tick - self._start_tick) * self._freq
        self._start_tick = current_tick

        self._difftimes.append(different_time)

        fps = 1000.0 / (sum(self._difftimes) / len(self._difftimes))
        fps_rounded = round(fps, 2)

        return fps_rounded

# 픽토그램 그리기 함수
def draw_stick_figure(
        image,
        landmarks,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
        visibility_th=0.4,
):
    image_width, image_height = image.shape[1], image.shape[0]

    # 랜드마크 위치 저장
    landmark_point = []
    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append(
            [index, landmark.visibility, (landmark_x, landmark_y), landmark_z])   

    '''
    • 반드시 참고!! https://google.github.io/mediapipe/solutions/pose.html
    • landmark_z: 거리값(카메라에 가까울수록 값이 작음)
    • visibility: [0.0, 1.0]의 값은 이미지에서 랜드마크가 표시될 가능성(가려져 있지 않고 존재함)을 나타냅니다.
    '''
    
    # 다리의 시작점을 허리의 중심으로 설정
    right_leg = landmark_point[23]  # 23: left_hip
    left_leg = landmark_point[24]   # 24: right_hip
    leg_x = int((right_leg[2][0] + left_leg[2][0]) / 2)  # [2][0]: landmark_x 
    leg_y = int((right_leg[2][1] + left_leg[2][1]) / 2)  # [2][1]: landmark_y

    landmark_point[23][2] = (leg_x, leg_y)
    landmark_point[24][2] = (leg_x, leg_y)

    # 거리값 기준 정렬
    sorted_landmark_point = sorted(landmark_point,
                                   reverse=True,
                                   key=lambda x: x[3])

    # 크기 계산
    (face_x, face_y), face_radius = min_enclosing_face_circle(landmark_point)

    face_x = int(face_x)
    face_y = int(face_y)
    face_radius = int(face_radius * 1.5)

    stick_radius01 = int(face_radius * (4 / 5))
    stick_radius02 = int(stick_radius01 * (3 / 4))
    stick_radius03 = int(stick_radius02 * (3 / 4))

    # 그리기 리스트
    draw_list = [
        11,  # 우측 어깨
        12,  # 좌측 어깨
        23,  # 우측 힙
        24,  # 죄측 힙
    ]

    # 배경색상
    cv.rectangle(image, (0, 0), (image_width, image_height),
                 bg_color,
                 thickness=-1)

    # 얼굴 그리기
    cv.circle(image, (face_x, face_y), face_radius, color, -1)

    # 팔, 다리 그리기
    for landmark_info in sorted_landmark_point:
        index = landmark_info[0]

        if index in draw_list:
            point01 = [p for p in landmark_point if p[0] == index][0]
            point02 = [p for p in landmark_point if p[0] == (index + 2)][0]
            point03 = [p for p in landmark_point if p[0] == (index + 4)][0]

            if point01[1] > visibility_th and point02[1] > visibility_th:
                image = draw_stick(
                    image,
                    point01[2],
                    stick_radius01,
                    point02[2],
                    stick_radius02,
                    color=color,
                    bg_color=bg_color,
                )
            if point02[1] > visibility_th and point03[1] > visibility_th:
                image = draw_stick(
                    image,
                    point02[2],
                    stick_radius02,
                    point03[2],
                    stick_radius03,
                    color=color,
                    bg_color=bg_color,
                )

    return image


def min_enclosing_face_circle(landmark_point):
    landmark_array = np.empty((0, 2), int)

    index_list = [1, 4, 7, 8, 9, 10]
    for index in index_list:
        np_landmark_point = [
            np.array(
                (landmark_point[index][2][0], landmark_point[index][2][1]))
        ]
        landmark_array = np.append(landmark_array, np_landmark_point, axis=0)

    center, radius = cv.minEnclosingCircle(points=landmark_array)

    return center, radius


def draw_stick(
        image,
        point01,
        point01_radius,
        point02,
        point02_radius,
        color=(100, 33, 3),
        bg_color=(255, 255, 255),
):
    cv.circle(image, point01, point01_radius, color, -1)
    cv.circle(image, point02, point02_radius, color, -1)

    draw_list = []
    for index in range(2):
        rad = math.atan2(point02[1] - point01[1], point02[0] - point01[0])

        rad = rad + (math.pi / 2) + (math.pi * index)
        point_x = int(point01_radius * math.cos(rad)) + point01[0]
        point_y = int(point01_radius * math.sin(rad)) + point01[1]

        draw_list.append([point_x, point_y])

        point_x = int(point02_radius * math.cos(rad)) + point02[0]
        point_y = int(point02_radius * math.sin(rad)) + point02[1]

        draw_list.append([point_x, point_y])

    points = np.array((draw_list[0], draw_list[1], draw_list[3], draw_list[2]))
    cv.fillConvexPoly(image, points=points, color=color)

    return image


def draw_landmarks(
    image,
    landmarks,
    visibility_th=0.4,
):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    for index, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_z = landmark.z
        landmark_point.append([landmark.visibility, (landmark_x, landmark_y)])

        if landmark.visibility < visibility_th:
            continue
        
        '''
        cv2.circle(img, center, radian, color, thickness)
        Parameters:	
            img – 그림을 그릴 이미지
            center – 원의 중심 좌표(x, y)
            radian – 반지름
            color – BGR형태의 Color
            thickness – 선의 두께, -1 이면 원 안쪽을 채움
        '''

        if index == 0:  # 코
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 1:  # 우측 눈
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 2:  # 우측 눈동자
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 3:  # 우측 눈꼬리
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 4:  # 좌측 눈
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 5:  # 좌측 눈동자
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 6:  # 좌측 눈꼬리
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 7:  # 우측귀
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 8:  # 좌측귀
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 9:  # 우측 입
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 10:  # 좌측 입
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 11:  # 우측 어깨
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 12:  # 좌측 어깨
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 13:  # 우측 팔꿈치
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 14:  # 좌측 팔꿈치
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 15:  # 우측 손목
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 16:  # 좌측 손목
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 17:  # 우측 손1(손에서 우)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 18:  # 좌측 손1(손에서 좌)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 19:  # 우측 손2(손에서 상)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 20:  # 좌축 손2(손에서 상)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 21:  # 우측 손3(손에서 안쪽)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 22:  # 좌측 손3(손에서 안쪽)
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 23:  # 우측 허리
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 24:  # 좌측 허리
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 25:  # 우측 무릎
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 26:  # 좌측 무릎
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 27:  # 우측 발목
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 28:  # 좌측 발목
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 29:  # 우측 발뒤꿈치
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 30:  # 좌측 발뒤꿈치
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 31:  # 우측 발앞꿈치
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)
        if index == 32:  # 좌측 발앞꿈치
            cv.circle(image, (landmark_x, landmark_y), 5, (0, 255, 0), 2)

        if True:
            cv.putText(image, "z:" + str(round(landmark_z, 3)),              # 이미지 / 문자열
                       (landmark_x - 10, landmark_y - 10),                   # 좌표
                       cv.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (0, 255, 0), 1,   # 폰트 타입, 크기, 컬러, 두께
                       cv.LINE_AA)                                           # 선의 끝 유형
    
    # 랜드마크 간의 선 긋기
    # 우측 눈
    if landmark_point[1][0] > visibility_th and landmark_point[2][0] > visibility_th:
        cv.line(image, landmark_point[1][1], landmark_point[2][1], (0, 255, 0), 2)
    if landmark_point[2][0] > visibility_th and landmark_point[3][0] > visibility_th:
        cv.line(image, landmark_point[2][1], landmark_point[3][1], (0, 255, 0), 2)

    # 좌측 눈
    if landmark_point[4][0] > visibility_th and landmark_point[5][0] > visibility_th:
        cv.line(image, landmark_point[4][1], landmark_point[5][1], (0, 255, 0), 2)
    if landmark_point[5][0] > visibility_th and landmark_point[6][0] > visibility_th:
        cv.line(image, landmark_point[5][1], landmark_point[6][1], (0, 255, 0), 2)

    # 입
    if landmark_point[9][0] > visibility_th and landmark_point[10][0] > visibility_th:
        cv.line(image, landmark_point[9][1], landmark_point[10][1], (0, 255, 0), 2)

    # 어깨
    if landmark_point[11][0] > visibility_th and landmark_point[12][0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[12][1], (0, 255, 0), 2)

    # 우측팔
    if landmark_point[11][0] > visibility_th and landmark_point[13][0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[13][1], (0, 255, 0), 2)
    if landmark_point[13][0] > visibility_th and landmark_point[15][0] > visibility_th:
        cv.line(image, landmark_point[13][1], landmark_point[15][1], (0, 255, 0), 2)

    # 좌측팔
    if landmark_point[12][0] > visibility_th and landmark_point[14][0] > visibility_th:
        cv.line(image, landmark_point[12][1], landmark_point[14][1], (0, 255, 0), 2)
    if landmark_point[14][0] > visibility_th and landmark_point[16][0] > visibility_th:
        cv.line(image, landmark_point[14][1], landmark_point[16][1], (0, 255, 0), 2)

    # 우측손
    if landmark_point[15][0] > visibility_th and landmark_point[17][0] > visibility_th:
        cv.line(image, landmark_point[15][1], landmark_point[17][1], (0, 255, 0), 2)
    if landmark_point[17][0] > visibility_th and landmark_point[19][0] > visibility_th:
        cv.line(image, landmark_point[17][1], landmark_point[19][1], (0, 255, 0), 2)
    if landmark_point[19][0] > visibility_th and landmark_point[21][0] > visibility_th:
        cv.line(image, landmark_point[19][1], landmark_point[21][1], (0, 255, 0), 2)
    if landmark_point[21][0] > visibility_th and landmark_point[15][0] > visibility_th:
        cv.line(image, landmark_point[21][1], landmark_point[15][1], (0, 255, 0), 2)

    # 좌측손
    if landmark_point[16][0] > visibility_th and landmark_point[18][0] > visibility_th:
        cv.line(image, landmark_point[16][1], landmark_point[18][1], (0, 255, 0), 2)
    if landmark_point[18][0] > visibility_th and landmark_point[20][0] > visibility_th:
        cv.line(image, landmark_point[18][1], landmark_point[20][1], (0, 255, 0), 2)
    if landmark_point[20][0] > visibility_th and landmark_point[22][0] > visibility_th:
        cv.line(image, landmark_point[20][1], landmark_point[22][1], (0, 255, 0), 2)
    if landmark_point[22][0] > visibility_th and landmark_point[16][0] > visibility_th:
        cv.line(image, landmark_point[22][1], landmark_point[16][1], (0, 255, 0), 2)

    # 몸체
    if landmark_point[11][0] > visibility_th and landmark_point[23][0] > visibility_th:
        cv.line(image, landmark_point[11][1], landmark_point[23][1], (0, 255, 0), 2)
    if landmark_point[12][0] > visibility_th and landmark_point[24][0] > visibility_th:
        cv.line(image, landmark_point[12][1], landmark_point[24][1], (0, 255, 0), 2)
    if landmark_point[23][0] > visibility_th and landmark_point[24][0] > visibility_th:
        cv.line(image, landmark_point[23][1], landmark_point[24][1], (0, 255, 0), 2)

    if len(landmark_point) > 25:
        # 우측발
        if landmark_point[23][0] > visibility_th and landmark_point[25][0] > visibility_th:
            cv.line(image, landmark_point[23][1], landmark_point[25][1], (0, 255, 0), 2)
        if landmark_point[25][0] > visibility_th and landmark_point[27][0] > visibility_th:
            cv.line(image, landmark_point[25][1], landmark_point[27][1], (0, 255, 0), 2)
        if landmark_point[27][0] > visibility_th and landmark_point[29][0] > visibility_th:
            cv.line(image, landmark_point[27][1], landmark_point[29][1], (0, 255, 0), 2)
        if landmark_point[29][0] > visibility_th and landmark_point[31][0] > visibility_th:
            cv.line(image, landmark_point[29][1], landmark_point[31][1], (0, 255, 0), 2)

        # 좌측발
        if landmark_point[24][0] > visibility_th and landmark_point[26][0] > visibility_th:
            cv.line(image, landmark_point[24][1], landmark_point[26][1], (0, 255, 0), 2)
        if landmark_point[26][0] > visibility_th and landmark_point[28][0] > visibility_th:
            cv.line(image, landmark_point[26][1], landmark_point[28][1], (0, 255, 0), 2)
        if landmark_point[28][0] > visibility_th and landmark_point[30][0] > visibility_th:
            cv.line(image, landmark_point[28][1], landmark_point[30][1], (0, 255, 0), 2)
        if landmark_point[30][0] > visibility_th and landmark_point[32][0] > visibility_th:
            cv.line(image, landmark_point[30][1], landmark_point[32][1], (0, 255, 0), 2)
    return image

def main():
    # 기본 설정
    cap_device = 0
    cap_width = 640
    cap_height = 480

    static_image_mode = 0
    model_complexity = 1  #help='model_complexity(0(가벼움),1(보통),2(무거움))'
    min_detection_confidence = 0.5
    min_tracking_confidence = 0.5
    rev_color = 0  #색상 반전효과(0: 배경-흰색, 픽토-파랑 / 1: 배경-파랑, 픽토-흰색)

    # 카메라 설정 
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # mediapipe 모델 
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode,
        model_complexity,
        min_detection_confidence,
        min_tracking_confidence,
    )

    # FPS 계산 
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 색상 설정
    if rev_color:
        color = (255, 255, 255)
        bg_color = (100, 33, 3)
    else:
        color = (100, 33, 3)
        bg_color = (255, 255, 255)

    while True:
        display_fps = cvFpsCalc.get()

        # 카메라 동작 
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # 좌우 반전(미러링)
        debug_image01 = copy.deepcopy(image)  
        debug_image02 = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        cv.rectangle(debug_image02, (0, 0), (image.shape[1], image.shape[0]),
                    bg_color,
                    thickness=-1)


        # 이미지 추출 및 (포즈)분석 
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        results = pose.process(image)

        # 포즈 그리기 
        if results.pose_landmarks is not None:
            debug_image01 = draw_landmarks(
                debug_image01,
                results.pose_landmarks,
            )
            debug_image02 = draw_stick_figure(
                debug_image02,
                results.pose_landmarks,
                color=color,
                bg_color=bg_color,
            )

        cv.putText(debug_image01, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        cv.putText(debug_image02, "FPS:" + str(display_fps), (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv.LINE_AA)

        # 종료 key: q 
        key = cv.waitKey(1)
        if key == ord('q'):  # ESC: 27
            break

        # 화면 생성 
        cv.imshow('Tokyo2020 Original', debug_image01)
        cv.imshow('Tokyo2020 Pictogram', debug_image02)

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()

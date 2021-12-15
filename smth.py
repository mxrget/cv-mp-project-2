from cv2 import cv2
import mediapipe as mp
import numpy as np


def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)

def palm_size(landmark, shape):
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) **.5

def palm_size_2(landmark, shape):
    x1, y1 = landmark[12].x * shape[1], landmark[12].y * shape[0]
    x2, y2 = landmark[16].x * shape[1], landmark[16].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) **.5

def palm_size_3(landmark, shape):
    x1, y1 = landmark[4].x * shape[1], landmark[4].y * shape[0]
    x2, y2 = landmark[20].x * shape[1], landmark[20].y * shape[0]
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) **.5

handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
        ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        ws_2 = palm_size_2(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        ws_3 = palm_size_3(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        print('1: ' + str(2 * r / ws))
        print('2: ' + str(2 * r / ws_2))
        print('3: ' + str(2 * r / ws_3))
        print('-------------------------')
        if 2 * r / ws < 1.65 and 2 * r / ws_2 > 4 and 2 * r / ws_3 > 1.35:
            cv2.putText(flippedRGB, 'rock', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), thickness = 3)
        elif 2 * r / ws < 2.5 and 2 * r / ws_2 < 3 and 2 * r / ws_3 > 2.9:
            cv2.putText(flippedRGB, 'scissors', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), thickness = 3)
        elif 2 * r / ws > 1.8 and 2 * r / ws_2 > 3 and 2 * r / ws_3 < 1.8:
            cv2.putText(flippedRGB, 'paper', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), thickness = 3)
        else:
            cv2.putText(flippedRGB, 'unrecognizable', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), thickness = 3)
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow('play', res_image)

handsDetector.close()
from cv2 import cv2
import mediapipe as mp
import numpy as np


def get_points(landmark, shape):
    points = []
    for mark in landmark:
        points.append([mark.x * shape[1], mark.y * shape[0]])
    return np.array(points, dtype=np.int32)

#def palm_size(landmark, shape): - кулак распознавать
    x1, y1 = landmark[0].x * shape[1], landmark[0].y * shape[0]
    x2, y2 = landmark[5].x * shape[1], landmark[5].y * shape[0]
    return ((x1 - x2)**2 + (y1 - y2) **2) **.5

def palm_size(landmark, shape): # распознавание ножниц
    x1, y1 = landmark[12].x * shape[1], landmark[12].y * shape[0]
    x2, y2 = landmark[16].x * shape[1], landmark[16].y * shape[0]
    return ((x1 - x2)**2 + (y1 - y2) **2) **.5

#этот параметр для ножниц всегда меньше двух

#создаем детектор
handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
count = 0
prev_fist = False
while cap.isOpened():
    ret, frame = cap.read()
    if cv2.waitKey(1) & 0xFF == ord('q') or not ret:
        break
    flipped = np.fliplr(frame)
    # переводим его в формат RGB для распознавания
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    # Распознаем
    results = handsDetector.process(flippedRGB)
    # Рисуем распознанное, если распозналось
    if results.multi_hand_landmarks is not None:
        (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
        ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        print(2 * r / ws)
        if 2 * r / ws > 1.35:
            cv2.circle(flippedRGB,(int(x), int(y)), int(r), (0, 0, 255), 2)
            # кулак разжат
            prev_fist = False
        elif 2 * r / ws < 1.8:
            cv2.circle(flippedRGB,(int(x), int(y)), int(r), (0, 255, 0), 2)
            if not prev_fist:
                # произошло сжимание
                count += 1
                # Сейчас кулак зажат
                prev_fist = True
    cv2.putText(flippedRGB, str(count), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), thickness=2)
    res_image = cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR)
    cv2.imshow("Hands", res_image)

handsDetector.close()
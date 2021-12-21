from cv2 import cv2
import mediapipe as mp
import numpy as np
import time
import random


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

def moment_detecting():
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        (x, y), r = cv2.minEnclosingCircle(get_points(results.multi_hand_landmarks[0].landmark, flippedRGB.shape))
        ws = palm_size(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        ws_2 = palm_size_2(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        ws_3 = palm_size_3(results.multi_hand_landmarks[0].landmark, flippedRGB.shape)
        if 2 * r / ws < 1.8 and 2 * r / ws_2 > 4 and 2 * r / ws_3 > 1.2:
            return 'rock'
        elif 2 * r / ws < 2.6 and 2 * r / ws_2 < 3 and 2 * r / ws_3 > 2.3:
            return 'scissors'
        elif 2 * r / ws > 1.8 and 2 * r / ws_2 > 3 and 2 * r / ws_3 < 2.3:
            return 'paper'
        else:
            return 'unrecognized'

handsDetector = mp.solutions.hands.Hands()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    result = moment_detecting()
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
        computer_choice = random.choice(['rock', 'paper', 'scissors'])
        time.sleep(1)
        print('Нашёл руку в кадре! Показывай жест через 3....')
        time.sleep(1)
        print('2....')
        time.sleep(1)
        print('1....')
        time.sleep(1)
        result = moment_detecting()
        if result == 'rock':
            print('Кажется, ты выбрал камень...')
            if computer_choice == 'rock':
                print('Ничья! Компьютер тоже выбрал камень')
                break
            elif computer_choice == 'paper':
                print('Ты проиграл :( компьютер выбрал бумагу!')
                break
            elif computer_choice == 'scissors':
                print('Победа! Компьютер выбрал ножницы!')
                break
        elif result == 'paper':
            print('Кажется, ты выбрал бумагу...')
            if computer_choice == 'rock':
                print('Победа! Компьютер выбрал камень')
                break
            elif computer_choice == 'paper':
                print('Ничья! Выбор компьютера тоже пал на бумагу')
                break
            elif computer_choice == 'scissors':
                print('Поражение - компьютер разрезал бумагу своими ножницами')
                break
        elif result == 'scissors':
            print('Кажется, ты выбрал ножницы...')
            if computer_choice == 'rock':
                print('Компьютер победил - он выбрал камень')
                break
            elif computer_choice == 'paper':
                print('Победа! Компьютер выбрал бумагу')
                break
            elif computer_choice == 'scissors':
                print('Ничья! Ваши мнения с компьютером совпали - оба выбрали ножницы')
                break
        else:
            print('Я не смог распознать, что именно ты хотел показать :/')
            break
    else:
        print('не вижу руку.....')


handsDetector.close()
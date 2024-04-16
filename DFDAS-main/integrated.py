from scipy.spatial import distance as dist
from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import pygame
import face_recognition
import os
from datetime import datetime

# Function to calculate eye aspect ratio
def calculate_eye_aspect_ratio(eye):
    vertical_dist_A = dist.euclidean(eye[1], eye[5])
    vertical_dist_B = dist.euclidean(eye[2], eye[4])
    horizontal_dist_C = dist.euclidean(eye[0], eye[3])
    ear = (vertical_dist_A + vertical_dist_B) / (2.0 * horizontal_dist_C)
    return ear

# Function to calculate mouth aspect ratio
def calculate_mouth_aspect_ratio(mouth):
    horizontal_dist_X = dist.euclidean(mouth[0], mouth[6])
    vertical_dist_Y1 = dist.euclidean(mouth[2], mouth[10])
    vertical_dist_Y2 = dist.euclidean(mouth[4], mouth[8])
    vertical_dist_Y = (vertical_dist_Y1 + vertical_dist_Y2) / 2.0
    mar = vertical_dist_Y / horizontal_dist_X
    return mar

# Function to find face encodings from a list of images
def find_face_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list

# Function to authenticate
def authentication(name):
    with open('Authentication.csv', 'a') as f:
        now = datetime.now()
        timestamp = now.strftime('%d-%m-%Y    %H:%M:%S')
        f.write(f'{name}, {timestamp}\n')


pygame.mixer.init()
pygame.mixer.music.load('alarm.mp3')

path = 'Training_images'
images = []
class_names = []
image_files = os.listdir(path)

for file in image_files:
    image = cv2.imread(os.path.join(path, file))
    images.append(image)
    class_names.append(os.path.splitext(file)[0])

encode_list_known = find_face_encodings(images)


camera = cv2.VideoCapture(0)


EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 10
MOU_AR_THRESHOLD = 0.75

COUNTER = 0
yawn_status = False
yawn_count = 0
authentication_list = {}


detector = dlib.get_frontal_face_detector()
predictor_path = 'shape_predictor_68_face_landmarks.dat'
predictor = dlib.shape_predictor(predictor_path)


(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mouth_start, mouth_end) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]


while True:
    ret, frame = camera.read()
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    prev_yawn_status = yawn_status

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over the face detections
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]
        mouth = shape[mouth_start:mouth_end]

        left_ear = calculate_eye_aspect_ratio(left_eye)
        right_ear = calculate_eye_aspect_ratio(right_eye)
        mou_ear = calculate_mouth_aspect_ratio(mouth)

        ear = (left_ear + right_ear) / 2.0

        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        mouth_hull = cv2.convexHull(mouth)

        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 255), 1)
        cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESHOLD:
            COUNTER += 1
            cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                pygame.mixer.music.play(-1)
        else:
            COUNTER = 0
            pygame.mixer.music.stop()
            cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)

        if mou_ear > MOU_AR_THRESHOLD:
            cv2.putText(frame, "Yawning", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yawn_status = True
            yawn_count += 1
            cv2.putText(frame, f"Yawn Count: {yawn_count}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            if yawn_count>=10:
                pygame.mixer.music.play(-1)
        else:
            yawn_status = False
            yawn_count=0

        if prev_yawn_status == True and yawn_status == False:
            yawn_count += 1

        cv2.putText(frame, "MAR: {:.2f}".format(mou_ear), (480, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        cv2.putText(frame, "Batch-10 Capstone Project", (350, 470), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0,0,0), 1)  #153, 51, 102

    # Face recognition
    imgS = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    faces_cur_frame = face_recognition.face_locations(imgS)
    encodes_cur_frame = face_recognition.face_encodings(imgS, faces_cur_frame)

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(encode_list_known, encode_face)
        face_dis = face_recognition.face_distance(encode_list_known, encode_face)

        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = class_names[match_index].upper()
            
            if name not in authentication_list:
                authentication(name)
                authentication_list[name] = True

            y1, x2, y2, x1 = face_loc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
        break
pygame.mixer.music.stop()
camera.release()
cv2.destroyAllWindows()

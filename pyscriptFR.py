import cv2
import face_recognition
import numpy as np
import os
from mtcnn import MTCNN

DATABASE_PATH = "face_database"

# Load known faces by encoding all images every time (no cache)
def load_known_faces():
    print("Encoding faces (this may take a while)...")
    known_encodings = []
    known_names = []

    for person_name in os.listdir(DATABASE_PATH):
        person_folder = os.path.join(DATABASE_PATH, person_name)
        if os.path.isdir(person_folder):
            for filename in os.listdir(person_folder):
                if filename.endswith(('.jpg', '.png')):
                    image_path = os.path.join(person_folder, filename)
                    image = face_recognition.load_image_file(image_path)
                    encodings = face_recognition.face_encodings(image)

                    if encodings:
                        known_encodings.append(encodings[0])
                        known_names.append(person_name)

    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

cap = cv2.VideoCapture(0)
print("Starting camera. Press 'q' to exit.")

detector = MTCNN()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(rgb_frame)

    for detection in detections:
        x, y, width, height = detection['box']
        x, y = max(0, x), max(0, y)
        top = y
        right = x + width
        bottom = y + height
        left = x

        face_location = [(top, right, bottom, left)]
        encoding = face_recognition.face_encodings(rgb_frame, face_location)
        name = "Unknown"

        if encoding:
            matches = face_recognition.compare_faces(known_encodings, encoding[0], tolerance=0.5)
            if True in matches:
                matched_index = np.argmin(face_recognition.face_distance(known_encodings, encoding[0]))
                if matches[matched_index]:
                    name = known_names[matched_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

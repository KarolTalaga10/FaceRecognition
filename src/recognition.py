# Import libraries
import numpy as np
import os
import face_recognition
import cv2
from pathlib import Path

# Import input video
path = "../data/unknown/squad_presentation.mp4"
input_vid = cv2.VideoCapture(path)

# Save output video
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter("../data/squad_presentation_out.mp4", fourcc, 25.00, (1280, 720))

# Import images of known players
known_face_encodings = []
known_face_names = []
path2 = Path("../data/known")
players = os.listdir(path2)

for player in players:
    img = face_recognition.load_image_file(path2 / player)
    known_face_encodings.append(face_recognition.face_encodings(img)[0])
    known_face_names.append(player.split('.', 1)[0])

# Try to find faces on each frame of video
face_locations = []
face_encodings = []
face_names = []
num = 0

while True:
    # Grab a single frame of video
    ret, frame = input_vid.read()
    num += 1
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]
    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = 'unknown_player'
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        face_names.append(name)

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.25, (255, 255, 255), 1)
    print("Writing frame {}".format(num))
    out.write(frame)

out.release()
input_vid.release()
cv2.destroyAllWindows()






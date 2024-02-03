import time
import cv2
import numpy as np
import tensorflow as tf
import streamlit as st
import webbrowser

st.header('**FIND THE NEW WAY OF MUSIC**')

lang = st.text_input('language:')
singer = st.text_input('Singer:')
btn = st.button('find me a song')
if btn:
    st.markdown('your face is detecting......')
    st.markdown('Do not shake your head!')

    new_model = tf.keras.models.load_model("My_model.h5")

    path = 'haarcascade_frontalface_default.xml'
    font_scale = 1.5
    font = cv2.FONT_HERSHEY_PLAIN

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise IOError('cannot open webcam')

    # Dictionary to store emotion counts
    emotion_counts = {'Angry': 0, 'Disgust': 0, 'Fear': 0, 'Happy': 0, 'Neutral': 0, 'Sad': 0, 'Surprise': 0}

    # Timer for 20 seconds
    start_time = time.time()
    duration = 10  # seconds

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        face_roi = None  # Initialize face_roi outside the loop
        for x, y, w, h in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            faces_inside = face_cascade.detectMultiScale(roi_gray)
            if len(faces_inside) == 0:
                print('face not detected!!')
            else:
                for (ex, ey, ew, eh) in faces_inside:
                    face_roi = roi_color[ey:ey + eh, ex:ex + ew]

            if face_roi is not None:
                final_img = cv2.resize(face_roi, (224, 224))
                final_img = np.expand_dims(final_img, axis=0)

                font = cv2.FONT_HERSHEY_PLAIN

                predictions = new_model.predict(final_img)
                
                # Determine the dominant emotion
                dominant_emotion = max(emotion_counts, key=emotion_counts.get)

                if np.argmax(predictions) == 0:
                    status = 'Angry'
                elif np.argmax(predictions) == 1:
                    status = 'Disgust'
                elif np.argmax(predictions) == 2:
                    status = 'Fear'
                elif np.argmax(predictions) == 3:
                    status = 'Happy'
                elif np.argmax(predictions) == 4:
                    status = 'Neutral'
                elif np.argmax(predictions) == 5:
                    status = 'Sad'
                else:
                    status = 'Surprise'

                # Update emotion counts
                emotion_counts[status] += 1

                x1, y1, w1, h1 = 0, 0, 175, 75
                cv2.putText(frame, status, (100, 150), font, 3, (0, 0, 255), 2, cv2.LINE_4)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255))

        cv2.imshow("face emotion detector", frame)

        if cv2.waitKey(2) & 0xFF == ord('q'):
            break

    # Find the dominant emotion after 20 seconds
    dominant_emotion = max(emotion_counts, key=emotion_counts.get)


    if lang or singer:
            query = f"{lang} {singer} {dominant_emotion} emotion songs"
            youtube_search_url = f"https://www.youtube.com/results?search_query={query}"
            webbrowser.open(youtube_search_url)
    else:
        query = f'{dominant_emotion} songs'
        youtube_search_url = f"https://www.youtube.com/results?search_query={query}"
        webbrowser.open(youtube_search_url)

    cap.release()
    cv2.destroyAllWindows()
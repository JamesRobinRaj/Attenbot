import cv2
import numpy as np
import pickle
import os
import csv
import time
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
import streamlit as st

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


def register_face(name):
    video = cv2.VideoCapture(0)
    mugam_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    mugam_data = []
    i = 0
    capturing = True
    video_placeholder = st.empty() 
    while capturing:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mugam = mugam_detect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in mugam:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50))
            if len(mugam_data) < 100 and i % 10 == 0:
                mugam_data.append(resized_img)
            i += 1
            cv2.putText(frame, str(len(mugam_data)), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
        video_placeholder.image(frame, channels="BGR", use_column_width=True)
        k = cv2.waitKey(1)
        if k == ord('q') or len(mugam_data) == 100:
            capturing = False
    video.release()
    cv2.destroyAllWindows()
    mugam_data = np.asarray(mugam_data)
    mugam_data = mugam_data.reshape(100, -1)
    names_file = 'names.pkl'
    mugam_data_file = 'mugam_data.pkl'
    if not os.path.exists(names_file):
        names = [name] * 100
        with open(names_file, 'wb') as f:
            pickle.dump(names, f)
    else:
        with open(names_file, 'rb') as f:
            names = pickle.load(f)
        names = names + [name] * 100
        with open(names_file, 'wb') as f:
            pickle.dump(names, f)

    if not os.path.exists(mugam_data_file):
        with open(mugam_data_file, 'wb') as f:
            pickle.dump(mugam_data, f)
    else:
        with open(mugam_data_file, 'rb') as f:
            mugam = pickle.load(f)
        mugam = np.append(mugam, mugam_data, axis=0)
        with open(mugam_data_file, 'wb') as f:
            pickle.dump(mugam, f)

def take_attendance():
    st.header("Attendance Monitoring")
    mugam_detect = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    st.write("Monitoring attendance...")

    video_placeholder = st.empty()
    video = cv2.VideoCapture(0)

    with open('names.pkl', 'rb') as f:
        labels = pickle.load(f)
    with open('mugam_data.pkl', 'rb') as f:
        MUGAM = pickle.load(f)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(MUGAM, labels)
    col = ['Name', "Time"]

    attendance_taken = False  

    while True:
        ret, frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mugam = mugam_detect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in mugam:
            crop_img = frame[y:y+h, x:x+w, :]
            resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
            output = knn.predict(resized_img)
            ts = time.time()
            date = datetime.fromtimestamp(ts).strftime("%d-%m-%Y")
            timestamp = datetime.fromtimestamp(ts).strftime("%h:%M-%S")
            exist = os.path.isfile("attendance_"+date+".csv")
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.rectangle(frame, (x, y-40), (x+w, y), (255, 0, 0), -1)
            cv2.putText(frame, str(output[0]), (x, y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)
            attendance = [str(output[0]), str(timestamp)]

        video_placeholder.image(frame, channels="BGR", use_column_width=True)

        if not attendance_taken:
            if exist:
                with open("attendance_"+date+".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(attendance)
            else:
                with open("attendance_"+date+".csv", "+a") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(col)
                    writer.writerow(attendance)
            attendance_taken = True  

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

def run_chatbot():
    def preprocess_text(text):
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens

    def get_best_response(user_input, question_response_pairs):
        user_tokens = preprocess_text(user_input)
        similarities = {}
        for question, response in question_response_pairs.items():
            question_tokens = preprocess_text(question)
            intersection = set(user_tokens) & set(question_tokens)
            union = set(user_tokens) | set(question_tokens)

            if len(union) == 0:
                jaccard_similarity = 0
            else:
                jaccard_similarity = len(intersection) / len(union)

            similarities[question] = jaccard_similarity

        best_question = max(similarities, key=similarities.get)
        return question_response_pairs[best_question]

    with open('question_response_pairs.pickle', 'rb') as file:
        question_response_pairs = pickle.load(file)

    st.title("Chatbot App")

    user_input = st.text_input("You:")
    if user_input.lower() == 'exit':
        st.write("Bot: Goodbye! Have a great day.")
    else:
        response = get_best_response(user_input, question_response_pairs)
        st.write(f"Bot: {response}")

st.sidebar.title("Menu")
option = st.sidebar.selectbox("Choose an action", ["Store Face Data","Take Attendance","Chat Bot"])

if option == "Store Face Data":
    st.title("Face Registration")
    name = st.text_input('Enter the name:')
    if name:
        start_registration = st.button("Start Registration")
        if start_registration:
            register_face(name)
            st.write("Registration completed.")
elif option == "Take Attendance":
    take_attendance()
elif option =="Chat Bot":
    run_chatbot()
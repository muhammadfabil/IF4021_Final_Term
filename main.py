import cv2
import mediapipe as mp
import random
from utils import count_fingers, load_question_image
from questions import questions

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inisialisasi Kamera
cap = cv2.VideoCapture(0)

# Pilih pertanyaan secara acak
current_question = random.choice(questions)
score = 0  # Skor pemain

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera.")
        break

    # Konversi ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Tampilkan gambar pertanyaan
    question_image, position = load_question_image(
        current_question["image"], frame.shape)
    if question_image is not None:
        x, y, w, h = position
        frame[y:y+h, x:x+w] = question_image

    # Deteksi tangan dan hitung jari yang diangkat
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_count = count_fingers(hand_landmarks)

            if 1 <= fingers_count <= 5:
                cv2.putText(frame, f"Jawaban Anda: {chr(64 + fingers_count)}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                if fingers_count == current_question["answer"]:
                    cv2.putText(frame, "Benar!", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    score += 1
                    cv2.waitKey(2000)
                    current_question = random.choice(questions)

    cv2.imshow("FingerFacts: Game Kuis Pilihan Ganda", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"Skor akhir Anda: {score}")

import cv2
import mediapipe as mp
import random
import time
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
waiting_for_next_question = False  # Menandakan apakah menunggu pertanyaan berikutnya

# Muat gambar notifikasi dan ubah ukurannya agar sesuai dengan frame
correct_image = cv2.imread("correct.png", cv2.IMREAD_UNCHANGED)
if correct_image is not None:
    target_width = 300  # Lebar target untuk gambar
    target_height = 200  # Tinggi target untuk gambar
    correct_image = cv2.resize(correct_image, (target_width, target_height), interpolation=cv2.INTER_AREA)

overlay_x = 0  # Initialize overlay position variables
overlay_y = 0
overlay_w = 0
overlay_h = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera.")
        break

    # Konversi ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Tampilkan gambar pertanyaan
    question_image, position = load_question_image(current_question["image"], frame)


    # Tampilkan notifikasi jika jawaban benar dan menunggu user menekan 'N'
    if waiting_for_next_question:  # Hapus pengecekan waktu
        overlay_x = (frame.shape[1] - correct_image.shape[1]) // 2
        overlay_y = (frame.shape[0] - correct_image.shape[0]) // 2
        overlay_h, overlay_w = correct_image.shape[:2]
        
        # Tampilkan overlay
        if all(v > 0 for v in [overlay_x, overlay_y, overlay_w, overlay_h]):
            for c in range(0, 3):  # Warna (BGR)
                frame_slice = frame[overlay_y:overlay_y + overlay_h, overlay_x:overlay_x + overlay_w, c]
                alpha_channel = correct_image[..., 3] / 255.0  # Saluran transparansi
                frame[overlay_y:overlay_y + overlay_h, overlay_x:overlay_x + overlay_w, c] = \
                    frame_slice * (1 - alpha_channel) + correct_image[..., c] * alpha_channel
        
        # Tambahkan petunjuk untuk user
        cv2.putText(frame, "Tekan 'N' untuk pertanyaan selanjutnya", 
                   (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 255), 2)

    # Deteksi tangan dan hitung jari yang diangkat
    if results.multi_hand_landmarks and not waiting_for_next_question:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            fingers_count = count_fingers(hand_landmarks)

            if 1 <= fingers_count <= 5:
                cv2.putText(frame, f"Jawaban Anda: {chr(64 + fingers_count)}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                if fingers_count == current_question["answer"]:
                    waiting_for_next_question = True
                    score += 1

    cv2.imshow("FingerFacts: Game Kuis Pilihan Ganda", frame)

    # Tangani input tombol
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Keluar game jika tekan 'q'
        break
    elif key == ord('n') and waiting_for_next_question:  # Hapus pengecekan waktu
        current_question = random.choice(questions)  # Pilih pertanyaan baru
        waiting_for_next_question = False  # Reset mode tunggu

cap.release()
cv2.destroyAllWindows()
print(f"Skor akhir Anda: {score}")



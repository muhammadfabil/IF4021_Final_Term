# utils.py

import cv2

def count_fingers(hand_landmarks):
    """
    Menghitung jumlah jari yang diangkat berdasarkan landmark tangan.
    """
    finger_tips = [8, 12, 16, 20]  # Ujung jari
    finger_pips = [6, 10, 14, 18]  # Sendi bawah ujung jari

    count = 0
    for tip, pip in zip(finger_tips, finger_pips):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:  # Jika ujung jari lebih tinggi
            count += 1

    # Periksa ibu jari
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    if abs(thumb_tip.x - thumb_mcp.x) > 0.1:  # Threshold untuk ibu jari
        count += 1

    return count

def load_question_image(image_path, frame_shape, width=300, height=200, y_offset=50):
    """
    Memuat dan menempatkan gambar pertanyaan ke dalam frame.
    """
    question_image = cv2.imread(image_path)
    if question_image is None:
        return None

    # Resize gambar
    question_image = cv2.resize(question_image, (width, height))

    # Hitung posisi tengah atas
    x_offset = (frame_shape[1] - width) // 2
    return question_image, (x_offset, y_offset, width, height)

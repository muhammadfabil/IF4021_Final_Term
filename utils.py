# utils.py

import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

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



def load_question_image(image_path, frame, width=300, height=261, y_offset=50):
    """
    Memuat dan menempatkan gambar pertanyaan ke dalam frame, mendukung transparansi, dan memastikan warna tidak berubah.
    """
    # Memuat gambar menggunakan PIL untuk mendukung transparansi
    question_image = Image.open(image_path).convert("RGBA")
    question_image = question_image.resize((width, height))

    # Konversi ke format numpy array
    question_image_np = np.array(question_image)

    # Pisahkan saluran RGBA
    r, g, b, a = cv2.split(question_image_np)  # Pisahkan saluran
    bgr_image = cv2.merge([b, g, r])  # Gabungkan kembali dalam format BGR
    alpha_channel = a

    # Hitung posisi tengah atas
    x_offset = (frame.shape[1] - width) // 2
    y_offset = max(0, y_offset)

    # Pastikan area tempelan tidak keluar dari batas frame
    y_end = min(y_offset + height, frame.shape[0])
    x_end = min(x_offset + width, frame.shape[1])

    # Pastikan ukuran slice frame dan gambar cocok
    overlay_height = y_end - y_offset
    overlay_width = x_end - x_offset

    bgr_image = bgr_image[:overlay_height, :overlay_width]
    alpha_channel = alpha_channel[:overlay_height, :overlay_width]

    # Tempelkan gambar dengan transparansi
    for c in range(0, 3):  # Untuk setiap kanal warna (BGR)
        frame_slice = frame[y_offset:y_end, x_offset:x_end, c]
        alpha = alpha_channel / 255.0  # Normalisasi alpha ke 0-1
        frame[y_offset:y_end, x_offset:x_end, c] = \
            (1 - alpha) * frame_slice + alpha * bgr_image[..., c]

    return frame, (x_offset, y_offset, width, height)

def draw_custom_text(frame, text, position, font_path, font_size, color):
    """
    Menggambar teks dengan font khusus menggunakan Pillow.
    
    Args:
    - frame: Frame dari OpenCV (BGR).
    - text: Teks yang ingin ditampilkan.
    - position: Tuple (x, y) posisi teks.
    - font_path: Path ke file font .ttf.
    - font_size: Ukuran font.
    - color: Warna teks (RGB tuple).
    
    Returns:
    - frame dengan teks yang telah digambar.
    """
    # Konversi frame dari OpenCV (BGR) ke PIL (RGB)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    font = ImageFont.truetype(font_path, font_size)
    
    # Tambahkan teks ke frame
    draw.text(position, text, font=font, fill=color)
    
    # Konversi kembali ke OpenCV (BGR)
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

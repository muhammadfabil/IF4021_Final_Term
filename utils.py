# utils.py

import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

def count_fingers(hand_landmarks):
    """
    Menghitung jumlah jari yang diangkat berdasarkan landmark tangan.
    
    Args:
    - hand_landmarks: Landmark tangan yang dihasilkan oleh model deteksi tangan (misalnya, MediaPipe).

    Returns:
    - count (int): Jumlah jari yang terangkat.
    """
    # Indeks landmark ujung jari dan sendi di bawahnya
    finger_tips = [8, 12, 16, 20]  # Ujung jari
    finger_pips = [6, 10, 14, 18]  # Sendi di bawah ujung jari

    count = 0  # Inisialisasi jumlah jari terangkat

    # Periksa setiap jari selain ibu jari
    for tip, pip in zip(finger_tips, finger_pips):
        # Jika ujung jari lebih tinggi (koordinat y lebih kecil) dari sendi bawahnya
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[pip].y:
            count += 1

    # Periksa ibu jari (berdasarkan jarak horizontal x dari tip dan MCP)
    thumb_tip = hand_landmarks.landmark[4]
    thumb_mcp = hand_landmarks.landmark[2]
    if abs(thumb_tip.x - thumb_mcp.x) > 0.1:  # Threshold untuk menentukan ibu jari terangkat
        count += 1

    return count


def load_question_image(image_path, frame, width=300, height=261, y_offset=50):
    """
    Memuat dan menempatkan gambar pertanyaan pada frame OpenCV, mendukung transparansi.

    Args:
    - image_path (str): Path ke file gambar pertanyaan.
    - frame (numpy.ndarray): Frame video dari OpenCV.
    - width (int): Lebar gambar yang diinginkan setelah di-resize.
    - height (int): Tinggi gambar yang diinginkan setelah di-resize.
    - y_offset (int): Offset vertikal untuk posisi gambar.

    Returns:
    - frame (numpy.ndarray): Frame yang telah dimodifikasi dengan gambar pertanyaan.
    - position (tuple): Posisi gambar yang ditempel (x_offset, y_offset, width, height).
    """
    # Memuat gambar menggunakan PIL untuk mendukung transparansi
    question_image = Image.open(image_path).convert("RGBA")
    question_image = question_image.resize((width, height))

    # Konversi ke format numpy array
    question_image_np = np.array(question_image)

    # Pisahkan saluran warna dan alpha (transparansi)
    r, g, b, a = cv2.split(question_image_np)
    bgr_image = cv2.merge([b, g, r])  # Ubah ke format BGR (untuk OpenCV)
    alpha_channel = a  # Saluran alpha untuk transparansi

    # Hitung posisi tengah atas frame
    x_offset = (frame.shape[1] - width) // 2
    y_offset = max(0, y_offset)

    # Tentukan area tempelan dengan memastikan tidak keluar dari batas frame
    y_end = min(y_offset + height, frame.shape[0])
    x_end = min(x_offset + width, frame.shape[1])

    # Pastikan ukuran slice gambar sesuai area frame
    overlay_height = y_end - y_offset
    overlay_width = x_end - x_offset

    # Potong gambar dan alpha sesuai ukuran overlay
    bgr_image = bgr_image[:overlay_height, :overlay_width]
    alpha_channel = alpha_channel[:overlay_height, :overlay_width]

    # Tempelkan gambar ke frame dengan mempertimbangkan transparansi
    for c in range(3):  # Iterasi untuk setiap kanal warna (BGR)
        frame_slice = frame[y_offset:y_end, x_offset:x_end, c]
        alpha = alpha_channel / 255.0  # Normalisasi alpha ke rentang 0-1
        frame[y_offset:y_end, x_offset:x_end, c] = \
            (1 - alpha) * frame_slice + alpha * bgr_image[..., c]

    return frame, (x_offset, y_offset, width, height)


def draw_custom_text(frame, text, position, font_path, font_size, color):
    """
    Menggambar teks pada frame dengan font khusus menggunakan PIL.

    Args:
    - frame (numpy.ndarray): Frame video dari OpenCV (format BGR).
    - text (str): Teks yang akan digambar.
    - position (tuple): Posisi teks dalam bentuk (x, y).
    - font_path (str): Path ke file font .ttf yang digunakan.
    - font_size (int): Ukuran font.
    - color (tuple): Warna teks dalam format RGB.

    Returns:
    - frame (numpy.ndarray): Frame yang telah dimodifikasi dengan teks.
    """
    # Konversi frame dari OpenCV (BGR) ke PIL (RGB)
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)

    # Muat font khusus
    font = ImageFont.truetype(font_path, font_size)

    # Tambahkan teks ke frame menggunakan PIL
    draw.text(position, text, font=font, fill=color)

    # Konversi kembali frame ke format OpenCV (BGR)
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

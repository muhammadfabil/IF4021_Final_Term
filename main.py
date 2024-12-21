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

def reset_game():
    return {
        'remaining_questions': questions.copy(),
        'score': 0,
        'wrong_attempts': 0,
        'waiting_for_next_question': False,
        'game_state': 'opening'  # States: 'opening', 'playing', 'game_over'
    }

# Initialize game state
game_data = reset_game()

# Load images
logo_image = cv2.imread("asset/logo.png", cv2.IMREAD_UNCHANGED)
correct_image = cv2.imread("correct.png", cv2.IMREAD_UNCHANGED)
game_over_image = cv2.imread("game_over.png", cv2.IMREAD_UNCHANGED)

# Resize images
def resize_image(image, target_width, target_height):
    if image is not None:
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return None

logo_image = resize_image(logo_image, 400, 300)
correct_image = resize_image(correct_image, 300, 200)
game_over_image = resize_image(game_over_image, 400, 300)

# Pilih pertanyaan pertama secara acak
current_question = random.choice(game_data['remaining_questions'])
game_data['remaining_questions'].remove(current_question)

def overlay_image(frame, overlay, alpha_channel=True):
    if overlay is None:
        return frame
    
    overlay_x = (frame.shape[1] - overlay.shape[1]) // 2
    overlay_y = (frame.shape[0] - overlay.shape[0]) // 2
    overlay_h, overlay_w = overlay.shape[:2]
    
    if all(v > 0 for v in [overlay_x, overlay_y, overlay_w, overlay_h]):
        for c in range(0, 3):
            frame_slice = frame[overlay_y:overlay_y + overlay_h, overlay_x:overlay_x + overlay_w, c]
            if alpha_channel:
                alpha = overlay[..., 3] / 255.0
            else:
                alpha = 1.0
            frame[overlay_y:overlay_y + overlay_h, overlay_x:overlay_x + overlay_w, c] = \
                frame_slice * (1 - alpha) + overlay[..., c] * alpha
    
    return frame

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera.")
        break

    # Konversi ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if game_data['game_state'] == 'opening':
        frame = overlay_image(frame, logo_image)
        cv2.putText(frame, "Tekan SPACE untuk memulai permainan", 
                   (int(frame.shape[1]/2) - 200, frame.shape[0] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    elif game_data['game_state'] == 'playing':
        # Tampilkan gambar pertanyaan
        question_image, position = load_question_image(current_question["image"], frame)
        
        # Tampilkan score dan kesalahan
        cv2.putText(frame, f"Score: {game_data['score']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Kesalahan: {game_data['wrong_attempts']}/3", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        if game_data['waiting_for_next_question']:
            frame = overlay_image(frame, correct_image)
            cv2.putText(frame, "Tekan 'N' untuk pertanyaan selanjutnya", 
                       (10, frame.shape[0] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Deteksi tangan dan hitung jari yang diangkat
        if results.multi_hand_landmarks and not game_data['waiting_for_next_question']:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers_count = count_fingers(hand_landmarks)

                if 1 <= fingers_count <= 5:
                    cv2.putText(frame, f"Jawaban Anda: {chr(64 + fingers_count)}", 
                              (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if fingers_count == current_question["answer"]:
                        game_data['waiting_for_next_question'] = True
                        game_data['score'] += 1
                    else:
                        game_data['wrong_attempts'] += 1
                        if game_data['wrong_attempts'] >= 3:
                            game_data['game_state'] = 'game_over'

    elif game_data['game_state'] == 'game_over':
        frame = overlay_image(frame, game_over_image)
        cv2.putText(frame, f"Total Score: {game_data['score']}", 
                   (int(frame.shape[1]/2) - 100, frame.shape[0] - 80), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, "Tekan BACKSPACE untuk main lagi", 
                   (int(frame.shape[1]/2) - 200, frame.shape[0] - 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("FingerFacts: Game Kuis Pilihan Ganda", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and game_data['game_state'] == 'opening':
        game_data['game_state'] = 'playing'
    elif key == ord('n') and game_data['waiting_for_next_question']:
        if game_data['remaining_questions']:
            current_question = random.choice(game_data['remaining_questions'])
            game_data['remaining_questions'].remove(current_question)
            game_data['waiting_for_next_question'] = False
        else:
            game_data['game_state'] = 'game_over'
    elif key == 8 and game_data['game_state'] == 'game_over':  # Backspace key
        game_data = reset_game()
        current_question = random.choice(game_data['remaining_questions'])
        game_data['remaining_questions'].remove(current_question)

cap.release()
cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import random
import time
from utils import count_fingers, load_question_image, draw_custom_text
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
        'waiting_for_confirmation': False,
        'current_question_number': 1,
        'current_question': None,  # Store current question here
        'game_state': 'opening'
    }

# Initialize game state
game_data = reset_game()

# Load images
logo_image = cv2.imread("assets/logo.png", cv2.IMREAD_UNCHANGED)
correct_image = cv2.imread("correct.png", cv2.IMREAD_UNCHANGED)
game_over_image = cv2.imread("game_over.png", cv2.IMREAD_UNCHANGED)
error_images = {
    1: cv2.imread("assets/error_1.png", cv2.IMREAD_UNCHANGED),
    2: cv2.imread("assets/error_2.png", cv2.IMREAD_UNCHANGED)
}

# Resize images
def resize_image(image, target_width, target_height):
    if image is not None:
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return None

logo_image = resize_image(logo_image, 400, 300)
correct_image = resize_image(correct_image, 148, 59)
game_over_image = resize_image(game_over_image, 400, 300)
game_start_image = cv2.imread("assets/game_start.png", cv2.IMREAD_UNCHANGED)
font_path = "assets/fonts/Poppins-ExtraBold.ttf"
counter_images = []
for i in range(1, 11):  # 1/10 hingga 10/10
    image_path = f"assets/counter/counter_{i}.png"  # Path ke gambar
    counter_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    counter_images.append(counter_image)

def get_next_question(game_data, increment_number=False):
    if increment_number:
        game_data['current_question_number'] += 1
    
    if game_data['remaining_questions']:
        question = random.choice(game_data['remaining_questions'])
        game_data['remaining_questions'].remove(question)
        return question
    return None

# Get first question
game_data['current_question'] = get_next_question(game_data)

def overlay_image(frame, overlay, alpha_channel=True, x_pos=0, y_pos=0):
    if overlay is None:
        return frame
    
    overlay_h, overlay_w = overlay.shape[:2]

    # Tentukan posisi overlay di dalam frame
    overlay_x = x_pos
    overlay_y = y_pos

    if all(v > 0 for v in [overlay_x, overlay_y, overlay_w, overlay_h]):
        for c in range(0, 3):
            frame_slice = frame[overlay_y:overlay_y + overlay_h, overlay_x:overlay_x + overlay_w, c]
            if alpha_channel:
                alpha = overlay[..., 3] / 255.0  # Jika ada transparansi (alpha channel)
            else:
                alpha = 1.0  # Jika tidak ada transparansi (untuk correct_image_resized)
            frame[overlay_y:overlay_y + overlay_h, overlay_x:overlay_x + overlay_w, c] = \
                frame_slice * (1 - alpha) + overlay[..., c] * alpha
    
    return frame

start_time = time.time()
timer_active = True

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera.")
        break

    # Konversi ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if game_data['game_state'] == 'opening':
        x_pos = (frame.shape[1] - game_start_image.shape[1]) // 2
        y_pos = (frame.shape[0] - game_start_image.shape[0]) // 2

        # Overlay gambar game_start_image
        frame = overlay_image(frame, game_start_image, alpha_channel=True, x_pos=x_pos, y_pos=y_pos)

    elif game_data['game_state'] == 'playing':
        # Tampilkan soal dan elemen lainnya di sini

        # Tampilkan gambar counter soal (1/10 hingga 10/10)
        counter_image = counter_images[game_data['current_question_number'] - 1]  # Ambil gambar yang sesuai dengan soal
        frame = overlay_image(frame, counter_image, alpha_channel=True, x_pos=frame.shape[1] - counter_image.shape[1] - 10, y_pos=10)  # Menempatkan gambar di kanan atas

        if timer_active:
            elapsed_time = time.time() - start_time
            remaining_time = max(0, 5 - elapsed_time)

        # Tampilkan gambar pertanyaan
        question_image, position = load_question_image(game_data['current_question']["image"], frame)
        x_offset, y_offset, width, height = position
        score_text = f"Score: {game_data['score']}  |  Timer: {remaining_time:.1f}s"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        frame = draw_custom_text(frame, score_text, (text_x, 30), font_path, 16, (214, 202, 178)) 


        if remaining_time <= 0 and timer_active and not game_data['waiting_for_confirmation']:
            game_data['waiting_for_confirmation'] = True
            timer_active = False

        if game_data['waiting_for_confirmation']:
            remaining_attempts = 2 - game_data['wrong_attempts']
            error_image = error_images.get(remaining_attempts, None)

            if error_image is not None:
                question_image_height = position[3]  # Tinggi gambar pertanyaan
                y_pos = position[1] + question_image_height + 10  # Posisikan di bawah gambar pertanyaan
                frame = overlay_image(
                    frame, error_image, alpha_channel=True,
                    x_pos=(frame.shape[1] - error_image.shape[1]) // 2,  # Tempatkan di tengah horizontal
                    y_pos=y_pos
                )   
        # Periksa jika kesalahan sudah mencapai 3
            if game_data['wrong_attempts'] >= 3:
                game_data['game_state'] = 'game_over'

        if game_data['waiting_for_next_question']:
            # Ambil posisi gambar soal
            correct_image = cv2.imread('correct.png', cv2.IMREAD_UNCHANGED)
            question_image_height = position[3]
            y_pos = position[1] + question_image_height + 10
            frame = overlay_image(frame, correct_image, alpha_channel=True, 
                                x_pos=(frame.shape[1] - correct_image.shape[1]) // 2, 
                                y_pos=y_pos)

        if results.multi_hand_landmarks and not game_data['waiting_for_next_question'] and not game_data['waiting_for_confirmation']:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers_count = count_fingers(hand_landmarks)

                if 1 <= fingers_count <= 5:
                    cv2.putText(frame, f"Jawaban Anda: {chr(64 + fingers_count)}", 
                              (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                    if fingers_count == game_data['current_question']["answer"]:
                        game_data['waiting_for_next_question'] = True
                        game_data['score'] += 1
                        timer_active = False

    elif game_data['game_state'] == 'game_over':
        # Resize game_over_image agar pas
        game_over_image_resized = resize_image(game_over_image, target_width=265, target_height=272)
        
        # Tentukan posisi gambar
        x_pos = (frame.shape[1] - game_over_image_resized.shape[1]) // 2
        y_pos = (frame.shape[0] - game_over_image_resized.shape[0]) // 2  # Tempatkan di tengah
        
        # Overlay gambar game over
        frame = overlay_image(frame, game_over_image_resized, alpha_channel=True, x_pos=x_pos, y_pos=y_pos)
        

    cv2.imshow("FingerFacts: Game Kuis Pilihan Ganda", frame)

    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' ') and game_data['game_state'] == 'opening':
        game_data['game_state'] = 'playing'
        start_time = time.time()
        timer_active = True
    elif key == ord('n') and game_data['waiting_for_next_question']:
        next_question = get_next_question(game_data, increment_number=True)  # Increment only on correct answer
        if next_question:
            game_data['current_question'] = next_question
            game_data['waiting_for_next_question'] = False
            start_time = time.time()
            timer_active = True
        else:
            game_data['game_state'] = 'game_over'
    elif key == ord('l') and game_data['waiting_for_confirmation']:
        game_data['wrong_attempts'] += 1
        if game_data['wrong_attempts'] >= 3:
            game_data['game_state'] = 'game_over'
        else:
            next_question = get_next_question(game_data, increment_number=False)  # Don't increment on wrong answer
            if next_question:
                game_data['current_question'] = next_question
            else:
                game_data['game_state'] = 'game_over'
            game_data['waiting_for_confirmation'] = False
            start_time = time.time()
            timer_active = True
    elif key == 8 and game_data['game_state'] == 'game_over':  # Backspace key
        game_data = reset_game()
        game_data['current_question'] = get_next_question(game_data)
        start_time = time.time()
        timer_active = True

cap.release()
cv2.destroyAllWindows()
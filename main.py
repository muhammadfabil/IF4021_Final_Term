import cv2
import mediapipe as mp
import random
import time
from utils import count_fingers, load_question_image, draw_custom_text
from questions import questions

# Inisialisasi MediaPipe Hands untuk deteksi tangan
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Inisialisasi kamera untuk capture video
cap = cv2.VideoCapture(0)

def reset_game():
    """
    Menginisialisasi ulang state game ke kondisi awal.
    
    Returns:
        dict: Dictionary berisi state awal game dengan keys:
            - remaining_questions: List pertanyaan yang tersisa
            - score: Skor pemain
            - wrong_attempts: Jumlah jawaban salah
            - waiting_for_next_question: Status menunggu pertanyaan berikutnya
            - waiting_for_confirmation: Status menunggu konfirmasi
            - current_question_number: Nomor pertanyaan saat ini
            - current_question: Pertanyaan yang sedang aktif
            - game_state: State game (opening/playing/game_over)
    """
    return {
        'remaining_questions': questions.copy(),
        'score': 0,
        'wrong_attempts': 0,
        'waiting_for_next_question': False,
        'waiting_for_confirmation': False,
        'current_question_number': 0,
        'current_question': None,
        'game_state': 'opening'
    }

# Inisialisasi state game
game_data = reset_game()

# Load semua asset gambar yang diperlukan
logo_image = cv2.imread("assets/logo.png", cv2.IMREAD_UNCHANGED)
correct_image = cv2.imread("correct.png", cv2.IMREAD_UNCHANGED)
game_over_image = cv2.imread("assets/game_over.png", cv2.IMREAD_UNCHANGED)
error_images = {
    1: cv2.imread("assets/error_1.png", cv2.IMREAD_UNCHANGED),
    2: cv2.imread("assets/error_2.png", cv2.IMREAD_UNCHANGED)
}

def resize_image(image, target_width, target_height):
    """
    Mengubah ukuran gambar sesuai dengan dimensi target.
    
    Args:
        image: Gambar yang akan diubah ukurannya
        target_width: Lebar target
        target_height: Tinggi target
    
    Returns:
        Gambar yang telah diresize atau None jika input invalid
    """
    if image is not None:
        return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)
    return None

# Resize gambar-gambar utama
logo_image = resize_image(logo_image, 400, 300)
game_over_image = resize_image(game_over_image, 400, 300)
game_start_image = cv2.imread("assets/game_start.png", cv2.IMREAD_UNCHANGED)
font_path = "assets/fonts/Poppins-ExtraBold.ttf"

# Load gambar counter (1/10 hingga 10/10)
counter_images = []
for i in range(1, 11):
    image_path = f"assets/counter/counter_{i}.png"
    counter_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    counter_images.append(counter_image)

def get_next_question(game_data):
    """
    Mengambil pertanyaan berikutnya secara random dari sisa pertanyaan yang ada.
    
    Args:
        game_data: Dictionary berisi state game
    
    Returns:
        Dict pertanyaan berikutnya atau None jika sudah tidak ada pertanyaan
    """
    game_data['current_question_number'] += 1
    
    if game_data['remaining_questions']:
        question = random.choice(game_data['remaining_questions'])
        game_data['remaining_questions'].remove(question)
        return question
    return None

# Ambil pertanyaan pertama
game_data['current_question'] = get_next_question(game_data)

def overlay_image(frame, overlay, alpha_channel=True, x_pos=0, y_pos=0):
    """
    Menggabungkan gambar overlay ke frame dengan mempertahankan transparansi.
    
    Args:
        frame: Frame utama
        overlay: Gambar yang akan di-overlay
        alpha_channel: Boolean untuk menggunakan alpha channel
        x_pos: Posisi x overlay
        y_pos: Posisi y overlay
    
    Returns:
        Frame yang telah ditambahkan overlay
    """
    if overlay is None:
        return frame
    
    overlay_h, overlay_w = overlay.shape[:2]
    overlay_x = x_pos
    overlay_y = y_pos

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

# Inisialisasi waktu awal untuk timer
start_time = time.time()
timer_active = True

# Loop utama game
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Tidak dapat membaca frame dari kamera.")
        break

    # Konversi frame ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # State: Opening Screen
    if game_data['game_state'] == 'opening':
        # Tampilkan layar pembuka game
        x_pos = (frame.shape[1] - game_start_image.shape[1]) // 2
        y_pos = (frame.shape[0] - game_start_image.shape[0]) // 2
        frame = overlay_image(frame, game_start_image, alpha_channel=True, x_pos=x_pos, y_pos=y_pos)

    # State: Playing Game
    elif game_data['game_state'] == 'playing':
        # Tampilkan counter pertanyaan
        counter_image = counter_images[game_data['current_question_number'] - 1]
        frame = overlay_image(frame, counter_image, alpha_channel=True, 
                            x_pos=frame.shape[1] - counter_image.shape[1] - 10, y_pos=10)

        # Kelola timer dan kesempatan menjawab
        if timer_active:
            elapsed_time = time.time() - start_time
            remaining_time = max(0, 7 - elapsed_time)

            if remaining_time <= 0:
                game_data['wrong_attempts'] += 1
                timer_active = False
                
                if game_data['wrong_attempts'] >= 3:
                    game_data['game_state'] = 'game_over'
                else:
                    game_data['waiting_for_confirmation'] = True

        # Tampilkan gambar pertanyaan dan skor
        question_image, position = load_question_image(game_data['current_question']["image"], frame)
        x_offset, y_offset, width, height = position
        score_text = f"Score: {game_data['score']}  |  Timer: {remaining_time:.1f}s"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        frame = draw_custom_text(frame, score_text, (text_x, 30), font_path, 16, (17, 32, 40))

        # Tampilkan feedback jawaban salah
        if game_data['waiting_for_confirmation'] and game_data['wrong_attempts'] < 3:
            remaining_attempts = 3 - game_data['wrong_attempts']
            error_image = error_images.get(remaining_attempts, None)

            if error_image is not None:
                question_image_height = position[3]
                y_pos = position[1] + question_image_height + 10
                frame = overlay_image(
                    frame, error_image, alpha_channel=True,
                    x_pos=(frame.shape[1] - error_image.shape[1]) // 2,
                    y_pos=y_pos
                )

        # Tampilkan feedback jawaban benar
        if game_data['waiting_for_next_question']:
            correct_image = cv2.imread('assets/correct.png', cv2.IMREAD_UNCHANGED)
            question_image_height = position[3]
            y_pos = position[1] + question_image_height + 10
            frame = overlay_image(frame, correct_image, alpha_channel=True, 
                                x_pos=(frame.shape[1] - correct_image.shape[1]) // 2, 
                                y_pos=y_pos)

        # Proses deteksi jari dan jawaban
        if results.multi_hand_landmarks and not game_data['waiting_for_next_question'] and not game_data['waiting_for_confirmation']:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers_count = count_fingers(hand_landmarks)

                if 1 <= fingers_count <= 5:
                    answer_text = f"Jawaban Anda: {chr(64 + fingers_count)}"
                    frame = draw_custom_text(frame, answer_text, (10, frame.shape[0] - 30), font_path, 16, (17, 32, 40))

                    if fingers_count == game_data['current_question']["answer"]:
                        game_data['waiting_for_next_question'] = True
                        game_data['score'] += 10
                        timer_active = False

    # State: Game Over
    elif game_data['game_state'] == 'game_over':
        # Tampilkan layar selesai jika berhasil menyelesaikan semua pertanyaan
        if len(game_data['remaining_questions']) == 0 and game_data['wrong_attempts'] < 3:
            well_done_text = "Well Done!"
            score_text = f"Skor Anda: {game_data['score']}"
            restart_text = "Tekan 'Backspace' untuk main lagi"

            # Hitung posisi teks untuk centered alignment
            well_done_size = cv2.getTextSize(well_done_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            restart_size = cv2.getTextSize(restart_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]

            well_done_x = (frame.shape[1] - well_done_size[0]) // 2
            score_x = (frame.shape[1] - score_size[0]) // 2
            restart_x = (frame.shape[1] - restart_size[0]) // 2

            well_done_y = (frame.shape[0] // 2) - 40
            score_y = well_done_y + 30
            restart_y = score_y + 40

            # Render teks
            frame = draw_custom_text(frame, well_done_text, (well_done_x, well_done_y), font_path, 20, (214, 202, 178))
            frame = draw_custom_text(frame, score_text, (score_x, score_y), font_path, 16, (214, 202, 178))
            frame = draw_custom_text(frame, restart_text, (restart_x, restart_y), font_path, 14, (214, 202, 178))
        else:
            # Tampilkan layar game over jika gagal
            game_over_image_resized = resize_image(game_over_image, target_width=265, target_height=272)
            x_pos = (frame.shape[1] - game_over_image_resized.shape[1]) // 2
            y_pos = (frame.shape[0] - game_over_image_resized.shape[0]) // 2
            frame = overlay_image(frame, game_over_image_resized, alpha_channel=True, x_pos=x_pos, y_pos=y_pos)

    # Tampilkan window game
    cv2.imshow("FingerFacts: Game Kuis Pilihan Ganda", frame)

    # Handle input keyboard
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # Keluar dari game
        break
    elif key == ord(' ') and game_data['game_state'] == 'opening':  # Mulai game
        game_data['game_state'] = 'playing'
        start_time = time.time()
        timer_active = True
    elif key == ord('n') and game_data['waiting_for_next_question']:  # Lanjut ke pertanyaan berikutnya
        next_question = get_next_question(game_data)
        if next_question:
            game_data['current_question'] = next_question
            game_data['waiting_for_next_question'] = False
            start_time = time.time()
            timer_active = True
        else:
            game_data['game_state'] = 'game_over'
    elif key == ord('l') and game_data['waiting_for_confirmation']:  # Coba lagi setelah jawaban salah
        if game_data['wrong_attempts'] < 3:
            next_question = get_next_question(game_data)
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
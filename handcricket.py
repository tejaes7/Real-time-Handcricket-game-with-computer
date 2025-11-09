import cv2
import mediapipe as mp
import random
import time
from collections import deque

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# Finger landmarks
finger_tips = [8, 12, 16, 20]  # index to pinky
thumb_tip = 4
thumb_base = 2

# Function to detect score
def detect_score_right(hand, h):
    landmarks = hand.landmark
    count = 0
    thumb_up = landmarks[thumb_tip].x < landmarks[thumb_base].x
    for tip in finger_tips:
        if landmarks[tip].y * h < landmarks[tip - 2].y * h:
            count += 1
        
    if thumb_up and count > 0:
            count += 1
    
    if thumb_up and count == 0:
        return 6
    if count == 0:
        return 0
    else:
        return count
    
def detect_score_left(hand, h):
    landmarks = hand.landmark
    count = 0
    thumb_up = landmarks[thumb_tip].x > landmarks[thumb_base].x
    for tip in finger_tips:
        if landmarks[tip].y * h < landmarks[tip - 2].y * h:
            count += 1
        
    if thumb_up and count > 0:
            count += 1
    
    if thumb_up and count == 0:
        return 6
    if count == 0:
        return 0
    else:
        return count

# Debounce queue for stable gesture
gesture_queue = deque(maxlen=5)

# Game variables
player_score, computer_score = 0, 0
player_out, computer_out = False, False
turn = "player_batting"
computer_move = 0
player_move = 0
last_update = time.time()
game_over = False
show_moves = False
move_display_time = 0

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    score_text = ""
    info_text = ""

    user_score = 0
    if results.multi_hand_landmarks:
        for handLms,handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            hand_label = handedness.classification[0].label
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            if hand_label == "Right":
                user_score = detect_score_right(handLms, h)
            else:
                user_score = detect_score_left(handLms, h)

    # Add to gesture queue
    if user_score != 0:
        gesture_queue.append(user_score)
    if user_score == 0:
        gesture_queue.append(0)
    if len(gesture_queue) == gesture_queue.maxlen:
        stable_score = max(set(gesture_queue), key=gesture_queue.count)
    else:
        stable_score = 0

    # Update game every 1.5 seconds
    current_time = time.time()
    
    # Show moves for 2 seconds after each play
    if show_moves and current_time - move_display_time > 2:
        show_moves = False
    
    if stable_score != 0 and current_time - last_update > 1.5 and not game_over and not show_moves:
        computer_move = random.randint(1, 6)
        player_move = stable_score
        last_update = current_time
        show_moves = True
        move_display_time = current_time

        if turn == "player_batting":
            if player_move == computer_move:
                player_out = True
                turn = "computer_batting"
                score_text = f"OUT! Both showed {player_move}"
            else:
                player_score += player_move
                score_text = f"Player scores {player_move} runs!"
                
        elif turn == "computer_batting":
            if player_move == computer_move:
                computer_out = True
                game_over = True
                score_text = f"OUT! Both showed {player_move}"
            else:
                computer_score += computer_move
                score_text = f"Computer scores {computer_move} runs!"
                if computer_score > player_score:
                    game_over = True
                    score_text = "Computer Wins the Match!"

    # Display scores and game info
    cv2.putText(frame, f"Player Score: {player_score}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Computer Score: {computer_score}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    # Show current phase
    if turn == "player_batting":
        phase_text = "Player Batting | Computer Bowling"
        phase_color = (0, 255, 0)  # Green
    else:
        phase_text = "Computer Batting | Player Bowling"
        phase_color = (0, 255, 255)  # Yellow
    
    cv2.putText(frame, phase_text, (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, phase_color, 2)

    # Show moves when available
    if show_moves:
        # Create a semi-transparent overlay for moves display
        overlay = frame.copy()
        cv2.rectangle(overlay, (w//2-200, h//2-100), (w//2+200, h//2+100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Display moves based on who's batting/bowling
        if turn == "player_batting":
            cv2.putText(frame, "BATTING vs BOWLING", (w//2-180, h//2-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Player: {player_move}", (w//2-180, h//2-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
            cv2.putText(frame, f"Computer: {computer_move}", (w//2-180, h//2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
        else:
            cv2.putText(frame, "BOWLING vs BATTING", (w//2-180, h//2-60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, f"Player: {player_move}", (w//2-180, h//2-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
            cv2.putText(frame, f"Computer: {computer_move}", (w//2-180, h//2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # Show result of the move
        if player_move == computer_move:
            result_color = (0, 0, 255)  # Red for OUT
            result_text = "OUT!"
        else:
            result_color = (0, 255, 0)  # Green for runs
            if turn == "player_batting":
                result_text = f"{player_move} runs scored!"
            else:
                result_text = f"{computer_move} runs scored!"
        
        cv2.putText(frame, result_text, (w//2-180, h//2+60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, result_color, 3)

    # Show current gesture detection
    cv2.putText(frame, f"Detected: {stable_score if stable_score != 0 else 'No gesture'}", 
                (w-300, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Show game instructions
    instructions = [
        "Show fingers (1-5) or thumb (6) to play",
        "Hold gesture steady for 1.5 seconds",
        "Same number = OUT!",
        "Press R: Restart, Q: Quit"
    ]
    
    for i, instruction in enumerate(instructions):
        cv2.putText(frame, instruction, (10, h-120 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # Game over screen
    if game_over:
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Final result
        if player_score > computer_score:
            result = "PLAYER WINS! "
            color = (0, 255, 0)
        elif computer_score > player_score:
            result = "COMPUTER WINS! "
            color = (0, 255, 255)
        else:
            result = "MATCH TIED! "
            color = (255, 255, 255)
            
        cv2.putText(frame, "GAME OVER", (w//2-150, h//2-80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        cv2.putText(frame, result, (w//2-200, h//2-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, f"Final Score: Player {player_score} - {computer_score} Computer", 
                    (w//2-250, h//2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame, "Press R to Restart or Q to Quit", 
                    (w//2-200, h//2+60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    cv2.imshow("Hand Cricket", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset game
        player_score = computer_score = 0
        player_out = computer_out = False
        turn = "player_batting"
        game_over = False
        computer_move = 0
        player_move = 0
        show_moves = False
        gesture_queue.clear()

cap.release()
cv2.destroyAllWindows()
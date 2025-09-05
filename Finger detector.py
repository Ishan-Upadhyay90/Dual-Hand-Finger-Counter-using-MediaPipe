import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.8, 
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks
tip_ids = [4, 8, 12, 16, 20]

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)  # Mirror the image
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb_frame)
    total_fingers = 0

    if result.multi_hand_landmarks and result.multi_handedness:
        for idx, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[idx].classification[0].label  # "Left" or "Right"

            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                lm_list.append((int(lm.x * w), int(lm.y * h)))

            fingers = []

            # Thumb (x-coordinate depends on hand label)
            if hand_label == "Right":
                fingers.append(lm_list[tip_ids[0]][0] < lm_list[tip_ids[0] - 1][0])  # right thumb
            else:
                fingers.append(lm_list[tip_ids[0]][0] > lm_list[tip_ids[0] - 1][0])  # left thumb

            # Other fingers
            for i in range(1, 5):
                fingers.append(lm_list[tip_ids[i]][1] < lm_list[tip_ids[i] - 2][1])  # y-coordinate

            count = fingers.count(True)
            total_fingers += count

            # Draw landmarks & label
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x, y = lm_list[0]
            cv2.putText(frame, f'{hand_label} Hand: {count}', (x - 50, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    # Display total finger count
    cv2.putText(frame, f'Total Fingers: {total_fingers}', (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Accurate Both-Hand Finger Counter", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()

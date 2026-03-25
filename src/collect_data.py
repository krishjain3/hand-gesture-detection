import cv2
import mediapipe as mp
import os
import csv

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

GESTURES = ["open_palm", "fist", "thumbs_up", "peace", "pointing"]
SAMPLES_PER_GESTURE = 400   # increased from 200
DATA_FILE = "data/landmarks.csv"

os.makedirs("data", exist_ok=True)

hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.8)
cap = cv2.VideoCapture(0)

with open(DATA_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    header = [f"{axis}{i}" for i in range(21) for axis in ["x","y","z"]] + ["label"]
    writer.writerow(header)

    for gesture in GESTURES:

        # --- Instruction screen ---
        print(f"\n{'='*40}")
        print(f"  Next gesture: {gesture.upper()}")
        print(f"  Press SPACE when your hand is ready")
        print(f"{'='*40}")

        while True:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)

            # Black box at top
            cv2.rectangle(frame, (0,0), (640,80), (0,0,0), -1)
            cv2.putText(frame, f"Gesture: {gesture.upper()}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Position hand clearly, then press SPACE", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            cv2.imshow("Collect Data", frame)
            if cv2.waitKey(1) & 0xFF == ord(' '):
                break

        # --- Countdown 3 2 1 ---
        for i in range(3, 0, -1):
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            cv2.rectangle(frame, (0,0), (640,80), (0,0,0), -1)
            cv2.putText(frame, f"Starting in {i}...", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)
            cv2.imshow("Collect Data", frame)
            cv2.waitKey(1000)  # wait 1 second

        # --- Recording ---
        count = 0
        while count < SAMPLES_PER_GESTURE:
            ret, frame = cap.read()
            frame = cv2.flip(frame, 1)
            rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                lm  = result.multi_hand_landmarks[0]
                row = []
                for point in lm.landmark:
                    row.extend([point.x, point.y, point.z])
                row.append(gesture)
                writer.writerow(row)
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                count += 1

            # Progress bar
            progress = int((count / SAMPLES_PER_GESTURE) * 400)
            cv2.rectangle(frame, (0,0), (640,80), (0,0,0), -1)
            cv2.putText(frame, f"{gesture}  —  {count}/{SAMPLES_PER_GESTURE}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.rectangle(frame, (10, 50), (410, 70), (50,50,50), -1)
            cv2.rectangle(frame, (10, 50), (10 + progress, 70), (0, 255, 100), -1)
            cv2.putText(frame, "Hold gesture STEADY", (420, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)

            cv2.imshow("Collect Data", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"  Done! {count} samples recorded for {gesture}")

cap.release()
cv2.destroyAllWindows()
print("\nAll gestures recorded! Saved to", DATA_FILE)
import cv2
import mediapipe as mp
import joblib
import numpy as np

model = joblib.load("models/gesture_model.pkl")
le    = joblib.load("models/label_encoder.pkl")

mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands    = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75)

cap = cv2.VideoCapture(0)
print("Running... Press Q to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame  = cv2.flip(frame, 1)
    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    label = "No hand detected"
    color = (100, 100, 100)

    if result.multi_hand_landmarks:
        lm  = result.multi_hand_landmarks[0]
        row = []
        for point in lm.landmark:
            row.extend([point.x, point.y, point.z])

        pred  = model.predict([row])[0]
        prob  = model.predict_proba([row])[0].max()
        label = f"{le.inverse_transform([pred])[0]}  ({prob*100:.0f}%)"
        color = (0, 255, 100)

        mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    cv2.rectangle(frame, (0, 0), (420, 55), (0, 0, 0), -1)
    cv2.putText(frame, label, (10, 38),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    cv2.imshow("Hand Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
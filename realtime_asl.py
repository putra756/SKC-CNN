import cv2
import torch
import pickle
import mediapipe as mp
import torch.nn as nn


# =====================
# MODEL
# =====================

class GestureNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.Linear(63, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 28)

        )

    def forward(self, x):
        return self.model(x)


# =====================
# LOAD MODEL
# =====================

model = GestureNet()

model.load_state_dict(
    torch.load(
        "gesture_model.pth",
        map_location="cpu"
    )
)

model.eval()


# =====================
# LOAD ENCODER
# =====================

with open("label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)


# =====================
# MEDIAPIPE
# =====================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils


# =====================
# EXTRACT LANDMARK
# =====================

def extract_landmarks(hand):

    wrist_x = hand.landmark[0].x
    wrist_y = hand.landmark[0].y
    wrist_z = hand.landmark[0].z

    landmarks = []

    for lm in hand.landmark:

        landmarks.extend([
            lm.x - wrist_x,
            lm.y - wrist_y,
            lm.z - wrist_z
        ])

    return landmarks


# =====================
# WEBCAM
# =====================

cap = cv2.VideoCapture(0)

while True:

    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(
            frame,
            hand,
            mp_hands.HAND_CONNECTIONS
        )

        features = extract_landmarks(hand)

        X = torch.tensor(
            [features],
            dtype=torch.float32
        )

        with torch.no_grad():

            output = model(X)

            pred = torch.argmax(
                output,
                dim=1
            ).item()

        label = encoder.inverse_transform(
            [pred]
        )[0]

        cv2.putText(
            frame,
            f"Gesture : {label}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        "ASL Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == 27:
        break


cap.release()
cv2.destroyAllWindows()
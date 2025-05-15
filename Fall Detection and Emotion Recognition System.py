import cv2
import mediapipe as mp
import pygame
import time

# Initialize pygame for audio
pygame.mixer.init()
fall_alert_sound = 'siren.mp3'  # Use any alert sound file (e.g., beep.mp3)

# Load audio
def play_alert():
    pygame.mixer.music.load(fall_alert_sound)
    pygame.mixer.music.play()

# Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Posture detection logic
def detect_posture(landmarks):
    # Key points: shoulder (11,12), hip (23,24), knee (25,26)
    left_shoulder = landmarks[11]
    left_hip = landmarks[23]
    left_knee = landmarks[25]

    # Calculate vertical angles/position to detect fall
    shoulder_hip_y = abs(left_shoulder.y - left_hip.y)
    hip_knee_y = abs(left_hip.y - left_knee.y)

    if shoulder_hip_y < 0.1:
        return "FALL"
    elif shoulder_hip_y > 0.25 and hip_knee_y > 0.25:
        return "SITTING"
    else:
        return "STANDING"

# Start video capture
cap = cv2.VideoCapture('fall2.mp4')

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    fall_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert BGR to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = pose.process(image)

        # Draw pose annotations
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            posture = detect_posture(results.pose_landmarks.landmark)
            cv2.putText(image, f"Posture: {posture}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if posture == "FALL":
                if not fall_detected:
                    play_alert()
                    fall_detected = True
            else:
                fall_detected = False

        cv2.imshow("Posture Detection", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

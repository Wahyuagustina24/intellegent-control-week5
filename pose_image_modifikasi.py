from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Load model YOLOv8 Pose dan Object
pose_model = YOLO("yolov8n-pose.pt")
object_model = YOLO("yolov8n.pt")

# Warna dan ketebalan garis
COLOR = (0, 255, 0)
THICKNESS = 2

# Pasangan titik (skeleton) berdasarkan format COCO
SKELETON_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7),
    (0, 8), (8, 9), (9, 10),
    (0, 11), (11, 12), (12, 13),
    (1, 14), (14, 16)
]

# Nama-nama titik berdasarkan format COCO
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# === Baca gambar dari file lokal ===
image_path = "bus.jpg"  # Ganti dengan path gambar Anda
try:
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Tidak dapat membaca gambar dari {image_path}")
except FileNotFoundError as e:
    print(f"Error: {e}")
    exit()

# === Deteksi objek dengan YOLOv8 ===
object_results = object_model(frame)

for result in object_results:
    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy()
    confidences = result.boxes.conf.cpu().numpy()

    for box, class_id, conf in zip(boxes, class_ids, confidences):
        x1, y1, x2, y2 = map(int, box)
        # Ambil nama kelas objek dari model
        class_name = object_model.names[int(class_id)]
        # Buat kotak di sekitar objek
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Kotak biru

        # Buat background biru untuk teks
        text_bg_width = cv2.getTextSize(f'{class_name} {conf:.2f}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0] + 10
        text_bg_height = cv2.getTextSize(f'{class_name} {conf:.2f}', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][1] + 10
        cv2.rectangle(frame, (x1, y1 - text_bg_height), (x1 + text_bg_width, y1), (255, 0, 0), -1)  # Background biru

        # Tambahkan label nama kelas dan confidence dengan teks putih
        cv2.putText(frame, f'{class_name} {conf:.2f}', (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)  # Teks putih

# === Deteksi pose tubuh dengan YOLOv8 Pose ===
pose_results = pose_model(frame)

for result in pose_results:
    keypoints = result.keypoints.xy.cpu().numpy()

    if keypoints is not None:
        for person_keypoints in keypoints:
            # Gambar semua titik tubuh untuk setiap orang
            for i, (x, y) in enumerate(person_keypoints):
                if x > 0 and y > 0:
                    # Gambar titik tubuh
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # Titik merah
                    # Tambahkan keterangan nama titik
                    cv2.putText(frame, f'{KEYPOINT_NAMES[i]}', (int(x) + 5, int(y) - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    # Tambahkan keterangan angka di samping titik
                    cv2.putText(frame, f'{i}', (int(x) - 10, int(y) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

# === Deteksi pose tangan dengan MediaPipe Hands ===
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
hand_results = hands.process(rgb_frame)

# Tampilkan hasil deteksi tangan
if hand_results.multi_hand_landmarks:
    for hand_landmarks in hand_results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# === Simpan hasil ===
output_image = "result_pose_image_modifikasi.jpg"
cv2.imwrite(output_image, frame)
print(f"Hasil disimpan di: {output_image}")

# === Tampilkan hasil ===
cv2.imshow("YOLOv8 Pose and Object Detection", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
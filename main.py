import cv2
import time
from ultralytics import YOLO
from collections import defaultdict

# โหลดโมเดล YOLOv8 (กำหนด confidence)
model = YOLO("yolov8n.pt")

# เปิดกล้อง
cap = cv2.VideoCapture(0)

# เก็บเวลา count ล่าสุดของแต่ละ track
last_count_time = defaultdict(lambda: 0)

# เก็บตำแหน่งก่อนหน้า (เพื่อป้องกันการเปลี่ยน object ผิด)
last_positions = {}

# เก็บจำนวนเข้า-ออก
count_in, count_out = 0, 0

# กำหนดโซน (Zone A = ซ้าย, Zone B = ขวา)
zoneA_x, zoneB_x = 200, 400
cooldown_time = 2  # วินาที

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ตรวจจับวัตถุ (ใช้ conf 0.5)
    results = model.track(frame, persist=True, conf=0.5, iou=0.4)

    if results[0].boxes.id is not None:
        for box, track_id in zip(results[0].boxes.xyxy, results[0].boxes.id):
            x1, y1, x2, y2 = map(int, box[:4])
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            track_id = int(track_id)
            now = time.time()

            # ตำแหน่งก่อนหน้า
            prev_pos = last_positions.get(track_id, (cx, cy))
            last_positions[track_id] = (cx, cy)

            # คำนวณความใกล้เคียง (ถ้าเคลื่อนที่มากเกินไปในเฟรมเดียว ให้ข้าม)
            dx, dy = abs(cx - prev_pos[0]), abs(cy - prev_pos[1])
            if dx > 150 or dy > 150:  
                continue  # ป้องกันการเปลี่ยน object ID ผิด

            # ตรวจจับการข้ามโซน (มี cooldown)
            if (now - last_count_time[track_id]) > cooldown_time:
                if prev_pos[0] < zoneA_x and cx > zoneB_x:
                    count_in += 1
                    last_count_time[track_id] = now
                elif prev_pos[0] > zoneB_x and cx < zoneA_x:
                    count_out += 1
                    last_count_time[track_id] = now

            # วาดกรอบ
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # แสดงผล count
    cv2.putText(frame, f"IN: {count_in}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"OUT: {count_out}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # วาดเส้นโซน
    cv2.line(frame, (zoneA_x, 0), (zoneA_x, frame.shape[0]), (255, 0, 0), 2)
    cv2.line(frame, (zoneB_x, 0), (zoneB_x, frame.shape[0]), (0, 0, 255), 2)

    cv2.imshow("Counter", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # กด ESC เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()

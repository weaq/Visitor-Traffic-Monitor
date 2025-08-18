import os
import time
from datetime import datetime
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

# โหลดโมเดล
model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=30)

# กล้อง
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ตัวแปรนับ
count_in = 0
count_out = 0
counted_ids = set()
positions_dict = {}
zone_status = {}        # track_id -> zone ล่าสุด ("A","B","mid")
last_count_time = {}    # track_id -> เวลาที่นับล่าสุด

# กำหนดโซน
zone_A = 360  # โซนขวา
zone_B = 280  # โซนซ้าย
cooldown = 2.0  # วินาที

current_day = datetime.now().day

def get_today_filename():
    today_str = datetime.now().strftime('%Y-%m-%d')
    if not os.path.exists('logs'):
        os.makedirs('logs')
    return os.path.join('logs', f"count_{today_str}.csv")

def write_counts_to_file(in_count, out_count):
    filename = get_today_filename()
    file_exists = os.path.isfile(filename)
    with open(filename, 'a') as f:
        if not file_exists:
            f.write('datetime,count_in,count_out\n')
        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'{now},{in_count},{out_count}\n')

def get_zone(cx):
    if cx >= zone_A:
        return "A"
    elif cx <= zone_B:
        return "B"
    else:
        return "mid"

while True:
    # รีเซ็ตทุกวันใหม่
    if datetime.now().day != current_day:
        current_day = datetime.now().day
        counted_ids.clear()
        count_in = 0
        count_out = 0
        zone_status.clear()
        last_count_time.clear()

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.5)
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls_id = r
        if int(cls_id) == 0:  # ตรวจเฉพาะคน
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox, score, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r_, b = track.to_ltrb()
        cx, cy = int((l + r_) / 2), int((t + b) / 2)

        # zone ปัจจุบัน
        current_zone = get_zone(cx)
        prev_zone = zone_status.get(track_id, "mid")

        # เวลา ณ ตอนนี้
        now_time = time.time()
        last_time = last_count_time.get(track_id, 0)

        if (now_time - last_time) > cooldown:  # กันนับซ้ำภายใน cooldown
            # เข้า (จาก A → B)
            if prev_zone == "A" and current_zone == "B":
                count_in += 1
                counted_ids.add(track_id)
                last_count_time[track_id] = now_time
                write_counts_to_file(count_in, count_out)

            # ออก (จาก B → A)
            elif prev_zone == "B" and current_zone == "A":
                count_out += 1
                counted_ids.add(track_id)
                last_count_time[track_id] = now_time
                write_counts_to_file(count_in, count_out)

        # อัปเดต zone ล่าสุด
        zone_status[track_id] = current_zone
        positions_dict[track_id] = (cx, cy)

        # วาดกรอบ
        cv2.rectangle(frame, (int(l), int(t)), (int(r_), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # วาดเส้นโซน
    cv2.line(frame, (zone_A, 0), (zone_A, frame.shape[0]), (255, 0, 0), 2)
    cv2.line(frame, (zone_B, 0), (zone_B, frame.shape[0]), (0, 0, 255), 2)

    # แสดงผลนับ
    cv2.putText(frame, f'IN: {count_in}  OUT: {count_out}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    cv2.imshow('People Counter Zone A-B with Cooldown', frame)
    if cv2.waitKey(1) == 27:  # กด ESC เพื่อออก
        break

cap.release()
cv2.destroyAllWindows()

import os
import time
from datetime import datetime
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

# --- โหลดโมเดลและ tracker ---
model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=30)

# --- กล้อง ---
cap = cv2.VideoCapture(0)

# --- ตัวแปร ---
positions_dict = {}       # track_id -> (cx, cy)
zone_status = {}          # track_id -> zone ล่าสุด ("A","B","mid")
last_count_time = {}      # track_id -> เวลาที่นับล่าสุด
cooldown = 2.0            # วินาที

current_day = datetime.now().day

# --- ฟังก์ชันช่วย ---
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

def load_counts_today():
    """โหลดค่า count ล่าสุดจาก CSV ของวันนี้"""
    filename = get_today_filename()
    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:
                last_line = lines[-1].strip().split(',')
                try:
                    in_count = int(last_line[1])
                    out_count = int(last_line[2])
                    return in_count, out_count
                except:
                    return 0, 0
    return 0, 0

def get_zones(frame_width):
    zone_A = int(frame_width * 0.7)  # ขวา
    zone_B = int(frame_width * 0.3)  # ซ้าย
    return zone_A, zone_B

def get_zone(cx, zone_A, zone_B):
    if cx >= zone_A:
        return "A"
    elif cx <= zone_B:
        return "B"
    else:
        return "mid"

def check_cross(prev_zone, current_zone):
    if prev_zone=="mid" and current_zone=="B":
        return "in"
    elif prev_zone=="mid" and current_zone=="A":
        return "out"
    elif prev_zone=="A" and current_zone=="B":
        return "in"
    elif prev_zone=="B" and current_zone=="A":
        return "out"
    return None

# --- โหลดค่า count ล่าสุดตอนเริ่มระบบ ---
count_in, count_out = load_counts_today()
print(f"เริ่มระบบ count_in={count_in}, count_out={count_out}")

# --- main loop ---
while True:
    # รีเซ็ตทุกวันใหม่
    if datetime.now().day != current_day:
        current_day = datetime.now().day
        count_in, count_out = 0, 0
        zone_status.clear()
        last_count_time.clear()
        positions_dict.clear()

    ret, frame = cap.read()
    if not ret:
        break

    frame_width = frame.shape[1]
    zone_A, zone_B = get_zones(frame_width)

    # --- ตรวจจับ YOLOv8 ใหม่ ---
    results = model(frame, conf=0.5)
    detections = []

    if results:
        res = results[0]
        if len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            scores = res.boxes.conf.cpu().numpy()
            cls_ids = res.boxes.cls.int().cpu().numpy()

            for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, cls_ids):
                if int(cls_id) == 0:  # คน
                    bbox = [x1, y1, x2-x1, y2-y1]
                    detections.append((bbox, score, 'person'))

    # --- Update tracker ---
    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r_, b = track.to_ltrb()
        cx = int((l + r_) / 2)

        current_zone = get_zone(cx, zone_A, zone_B)
        prev_zone = zone_status.get(track_id)

        now_time = time.time()
        last_time = last_count_time.get(track_id, 0)

        # track ใหม่เริ่มจาก current_zone ไม่ถูกนับทันที
        if prev_zone is None:
            zone_status[track_id] = current_zone
            continue

        cross = check_cross(prev_zone, current_zone)
        if cross and (now_time - last_time) > cooldown:
            if cross == "in":
                count_in += 1
            else:
                count_out += 1
            last_count_time[track_id] = now_time
            write_counts_to_file(count_in, count_out)

        # อัปเดต zone ล่าสุด
        zone_status[track_id] = current_zone
        positions_dict[track_id] = (cx, int((t+b)/2))

        # วาดกรอบและ ID
        cv2.rectangle(frame, (int(l), int(t)), (int(r_), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # --- วาดเส้นโซน ---
    cv2.line(frame, (zone_A, 0), (zone_A, frame.shape[0]), (255, 0, 0), 2)
    cv2.line(frame, (zone_B, 0), (zone_B, frame.shape[0]), (0, 0, 255), 2)

    # --- แสดงผลนับ ---
    cv2.putText(frame, f'IN: {count_in}  OUT: {count_out}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    cv2.imshow('People Counter Zone A-B with Cooldown', frame)
    if cv2.waitKey(1) == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

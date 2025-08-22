import os, sys
import time
from datetime import datetime
import torch
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

import yaml

def resource_path(relative_path):
    """ หา path ของไฟล์เมื่อรันเป็น .exe """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# โหลด config.yaml
with open("config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# camera
camera_cfg = cfg.get("camera", {})
CAM_SOURCE = camera_cfg.get("source", 0)
CAM_WIDTH  = camera_cfg.get("width", 640)
CAM_HEIGHT = camera_cfg.get("height", 480)
CAM_FRAMERATE = camera_cfg.get("frame_rate", 30)
CAM_FLIPFRAME = camera_cfg.get("flip_frame", False)

# yolo
yolo_cfg = cfg.get("yolo", {})
MODEL_PATH = yolo_cfg.get("model", "yolov8n.pt")
CONF       = yolo_cfg.get("conf", 0.5)
IOU        = yolo_cfg.get("iou", 0.45)
CLASSES    = yolo_cfg.get("classes", [0])

zones_cfg = cfg.get("zones", {})
ZONE_A = zones_cfg.get("zone_a", 70)
ZONE_B = zones_cfg.get("zone_b", 30)

counter_cfg = cfg.get("counter", {})
COOLDOWN = counter_cfg.get("cooldown", 2.0)
COUNTER_DIRECTION = counter_cfg.get("direction", "horizontal")


# --- ตรวจสอบ device ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"✅ กำลังใช้งานบน {device.upper()}")

# --- โหลดโมเดลและ tracker ---
model = YOLO(MODEL_PATH).to(device)
tracker = DeepSort(max_age=15, n_init=3, nn_budget=100)

# --- กล้อง ---
cap = cv2.VideoCapture(CAM_SOURCE)
cap.set(cv2.CAP_PROP_FPS, CAM_FRAMERATE)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)


if not cap.isOpened():
    print("❌ ไม่สามารถเปิดกล้องได้")
    exit()

# --- ตัวแปร ---
positions_dict = {}       # track_id -> (cx, cy)
zone_status = {}          # track_id -> zone ล่าสุด ("A","B","mid")
last_count_time = {}      # track_id -> เวลาที่นับล่าสุด
cooldown = COOLDOWN            # วินาที

track_state = {}  # track_id -> 'toward_A', 'toward_B', 'none'

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

def get_zones(frame_width, frame_height):
    if COUNTER_DIRECTION == "vertical":
        zone_A = int(frame_height * (ZONE_A / 100))  # ล่าง
        zone_B = int(frame_height * (ZONE_B / 100))  # บน
    else:
        zone_A = int(frame_width * (ZONE_A / 100))   # ขวา
        zone_B = int(frame_width * (ZONE_B / 100))   # ซ้าย
    return zone_A, zone_B


def get_zone(cx, cy, zone_A, zone_B, direction):
    if direction == "vertical":
        if cy >= zone_A:
            return "A"   # ล่าง
        elif cy <= zone_B:
            return "B"   # บน
        else:
            return "mid"
    else:
        if cx >= zone_A:
            return "A"   # ขวา
        elif cx <= zone_B:
            return "B"   # ซ้าย
        else:
            return "mid"


def check_cross(prev_zone, current_zone, track_id):
    """
    คืนค่า 'in' / 'out' หรือ None
    ใช้ track_state[track_id] เก็บทิศทาง
    """
    if prev_zone == "mid":
        if current_zone == "B" and track_state.get(track_id) == "toward_B":
            track_state[track_id] = "none"
            return "in"
        elif current_zone == "A" and track_state.get(track_id) == "toward_A":
            track_state[track_id] = "none"
            return "out"
    elif prev_zone == "A" and current_zone == "mid":
        track_state[track_id] = "toward_B"
    elif prev_zone == "B" and current_zone == "mid":
        track_state[track_id] = "toward_A"
    
    return None



# --- โหลดค่า count ล่าสุดตอนเริ่มระบบ ---
count_in, count_out = load_counts_today()
print(f"เริ่มระบบ count_in={count_in}, count_out={count_out}")

# --- main loop ---
try:
    while True:
        # รีเซ็ตทุกวันใหม่
        if datetime.now().day != current_day:
            current_day = datetime.now().day
            count_in, count_out = 0, 0
            zone_status.clear()
            last_count_time.clear()
            positions_dict.clear()
            print("🔄 เริ่มนับใหม่วันใหม่")

        ret, frame = cap.read()
        if not ret:
            print("⚠️ ไม่สามารถอ่านภาพจากกล้องได้")
            break

        # --- Flip frame horizontal (mirror) ---
        if CAM_FLIPFRAME :
            frame = cv2.flip(frame, 1)  # 1 = horizontal, 0 = vertical, -1 = both

        frame_height, frame_width = frame.shape[:2]
        zone_A, zone_B = get_zones(frame_width, frame_height)


        # --- ตรวจจับ YOLOv8 ---
        results = model.predict(frame, conf=CONF, iou=IOU, verbose=False, device=device)
        detections = []

        if results:
            res = results[0]
            if len(res.boxes) > 0:
                boxes = res.boxes.xyxy.cpu().numpy()
                scores = res.boxes.conf.cpu().numpy()
                cls_ids = res.boxes.cls.int().cpu().numpy()

                for (x1, y1, x2, y2), score, cls_id in zip(boxes, scores, cls_ids):
                    if int(cls_id) == 0:      # คน
                        label = 'person'
                    elif int(cls_id) == 2:    # รถยนต์
                        label = 'car'
                    else:
                        continue

                    bbox = [x1, y1, x2-x1, y2-y1]
                    detections.append((bbox, score, label))

        # --- Update tracker ---
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            l, t, r_, b = track.to_ltrb()

            cx = int((l + r_) / 2)
            cy = int((t + b) / 2)

            current_zone = get_zone(cx, cy, zone_A, zone_B, COUNTER_DIRECTION)

            prev_zone = zone_status.get(track_id)

            now_time = time.time()
            last_time = last_count_time.get(track_id, 0)

            # track ใหม่เริ่มจาก current_zone ไม่ถูกนับทันที
            if prev_zone is None:
                zone_status[track_id] = current_zone
                continue

            cross = check_cross(prev_zone, current_zone, track_id)
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
        if COUNTER_DIRECTION == "vertical":
            cv2.line(frame, (0, zone_A), (frame.shape[1], zone_A), (255, 0, 0), 2)
            cv2.line(frame, (0, zone_B), (frame.shape[1], zone_B), (0, 0, 255), 2)
        else:
            cv2.line(frame, (zone_A, 0), (zone_A, frame.shape[0]), (255, 0, 0), 2)
            cv2.line(frame, (zone_B, 0), (zone_B, frame.shape[0]), (0, 0, 255), 2)

        # --- แสดงผล ---
        text_in = f'IN: {count_in}'
        (text_w, text_h), baseline = cv2.getTextSize(text_in, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)

        # วาง IN
        x_in, y_in = 10, 30
        cv2.putText(frame, text_in, (x_in, y_in), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

        # วาง OUT ต่อจาก IN + ระยะห่าง 20 px
        x_out = x_in + text_w + 20
        cv2.putText(frame, f'OUT: {count_out}', (x_out, y_in), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        
         # --- แสดงเวลา ---
        cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

        # แสดงผลแบบเต็มจอ
        cv2.namedWindow('People Counter Zone A-B with Cooldown', cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty('People Counter Zone A-B with Cooldown',
                              cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.imshow('People Counter Zone A-B with Cooldown', frame)

        # ออกด้วยปุ่ม q หรือ Esc
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # q หรือ Esc
            break

        # --- เคลียร์ dictionary ป้องกัน memory leak ---
        if len(zone_status) > 1000:
            zone_status.clear()
            last_count_time.clear()
            positions_dict.clear()

except KeyboardInterrupt:
    print("🛑 หยุดการทำงานด้วย Ctrl+C")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("🚪 ปิดระบบเรียบร้อย")

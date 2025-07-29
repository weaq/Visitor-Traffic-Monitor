import os
from datetime import datetime
from ultralytics import YOLO
import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO('yolov8n.pt')
tracker = DeepSort(max_age=30)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

count_in = 0
count_out = 0
counted_ids = set()
positions_dict = {}

line_position = 300
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

while True:
    # รีเซ็ตทุกวันใหม่
    if datetime.now().day != current_day:
        current_day = datetime.now().day
        counted_ids.clear()
        count_in = 0
        count_out = 0

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]
    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, cls_id = r
        if int(cls_id) == 0:
            bbox = [x1, y1, x2 - x1, y2 - y1]
            detections.append((bbox, score, 'person'))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, r_, b = track.to_ltrb()
        cx, cy = int((l + r_) / 2), int((t + b) / 2)

        prev_pos = positions_dict.get(track_id, None)

        if prev_pos is not None:
            prev_cy = prev_pos[1]

            if track_id not in counted_ids:
                if prev_cy < line_position and cy >= line_position:
                    count_in += 1
                    counted_ids.add(track_id)
                    write_counts_to_file(count_in, count_out)
                elif prev_cy > line_position and cy <= line_position:
                    count_out += 1
                    counted_ids.add(track_id)
                    write_counts_to_file(count_in, count_out)

        positions_dict[track_id] = (cx, cy)

        cv2.rectangle(frame, (int(l), int(t)), (int(r_), int(b)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.line(frame, (0, line_position), (frame.shape[1], line_position), (0, 0, 255), 2)
    cv2.putText(frame, f'IN: {count_in}  OUT: {count_out}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, datetime.now().strftime('%H:%M:%S'), (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    cv2.imshow('People Counter', frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()

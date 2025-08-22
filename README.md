# Visitor-Traffic-Monitor

# direction 
## horizontal

เข้า (IN) = เคลื่อนจากขวาไปซ้าย

ออก (OUT) = เคลื่อนจากซ้ายไปขวา

## vertical

เข้า (IN) = เคลื่อนจากบนลงล่าง

ออก (OUT) = เคลื่อนจากล่างขึ้นบน

# Install Library

$ pip install ultralytics opencv-python deep-sort-realtime numpy


# Create EXE
project/

 ├─ people_counter.py      

 ├─ config.yaml 

$ pip install pyinstaller

$ pyinstaller --onefile --add-data "config.yaml;." people_counter.py




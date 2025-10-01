# 🚗 Traffic Congestion Predictor

ระบบพยากรณ์การจราจรติด/ไม่ติด และวันหยุด/วันทำงาน โดยใช้ Machine Learning

## 🎯 ฟีเจอร์หลัก

### 1. การพยากรณ์การจราจรติด/ไม่ติด
- **เกณฑ์การประเมิน:**
  - v/c > 1 (อัตราส่วนปริมาณการจราจรต่อความจุถนน)
  - speed < 20 km/h (ความเร็วต่ำกว่า 20 กม./ชม.)
- **โมเดลที่ใช้:** Decision Tree, Random Forest, XGBoost

### 2. การพยากรณ์วันหยุด/วันทำงาน
- วิเคราะห์จากรูปแบบการจราจร (density, volume, speed)
- วันหยุด: เสาร์-อาทิตย์
- วันทำงาน: จันทร์-ศุกร์

## 📊 ฟีเจอร์ที่ใช้ (Features)

1. **density** - ความหนาแน่นของการจราจร
2. **volume** - ปริมาณการจราจร
3. **capacity** - ความจุถนน
4. **speed** - ความเร็ว (km/h)
5. **time_hour** - ชั่วโมง (0-23)
6. **day_of_week** - วันในสัปดาห์ (1=จันทร์, 7=อาทิตย์)

## 🚀 การติดตั้งและใช้งาน

### 1. ติดตั้ง Dependencies
```bash
pip install -r requirements.txt
pip install xgboost
```

### 2. เริ่มเซิร์ฟเวอร์
```bash
python -m uvicorn traffic_main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. เปิดเว็บเบราว์เซอร์
ไปที่: `http://127.0.0.1:8000`

## 🔧 การใช้งาน

### 1. การพยากรณ์แบบ Single
- กรอกข้อมูลการจราจรในฟอร์ม
- เลือกโมเดล (Decision Tree, Random Forest, XGBoost)
- กดปุ่ม "ทำนายการจราจร"

### 2. การประมวลผลแบบ Batch (Excel)
- อัปโหลดไฟล์ Excel ที่มีคอลัมน์: density, volume, capacity, speed, time_hour, day_of_week
- ระบบจะประมวลผลและให้ดาวน์โหลดผลลัพธ์

### 3. ฝึกโมเดลใหม่
- กดปุ่ม "ฝึกโมเดลใหม่" เพื่อสร้างโมเดลจากข้อมูลตัวอย่าง

## 📈 ตัวอย่างข้อมูล

### ข้อมูลการจราจรไม่ติด
```json
{
  "density": 50,
  "volume": 1500,
  "capacity": 2000,
  "speed": 45,
  "time_hour": 8,
  "day_of_week": 2
}
```

### ข้อมูลการจราจรติด
```json
{
  "density": 80,
  "volume": 2500,
  "capacity": 2000,
  "speed": 15,
  "time_hour": 17,
  "day_of_week": 2
}
```

## 🎨 UI Features

- **Cinematic Background:** เปลี่ยนตามสถานะการจราจร
- **Real-time Prediction:** แสดงผลลัพธ์ทันที
- **Multiple Models:** เลือกโมเดลได้
- **Batch Processing:** ประมวลผลไฟล์ Excel
- **Visual Indicators:** แสดงเกณฑ์การประเมิน

## 📁 โครงสร้างไฟล์

```
ML_Model_Predictor/
├── traffic_main.py          # ไฟล์หลัก FastAPI
├── templates/
│   └── traffic_index.html   # หน้าเว็บ
├── static/
│   └── traffic_style.css    # สไตล์ CSS
├── models/                   # โมเดลที่ฝึกแล้ว
│   ├── dt_traffic_model.pkl
│   ├── rf_traffic_model.pkl
│   └── xgb_traffic_model.pkl
├── test_traffic.py          # ไฟล์ทดสอบ
└── TRAFFIC_README.md        # คู่มือนี้
```

## 🔍 API Endpoints

- `GET /` - หน้าเว็บหลัก
- `GET /health` - ตรวจสอบสถานะ
- `POST /predict-traffic` - ทำนายการจราจร
- `POST /upload-traffic-excel` - อัปโหลดไฟล์ Excel
- `POST /train-models` - ฝึกโมเดลใหม่
- `GET /download/{filename}` - ดาวน์โหลดผลลัพธ์

## 🧪 การทดสอบ

```bash
python test_traffic.py
```

## 📊 ตัวอย่างผลลัพธ์

```json
{
  "traffic_prediction": 1,
  "traffic_label": "การจราจรติด (Congested)",
  "traffic_probabilities": {
    "ไม่ติด": "15.23%",
    "ติด": "84.77%"
  },
  "day_type_prediction": 0,
  "day_type_label": "วันทำงาน (Weekday)",
  "actual_congested": 1,
  "vc_ratio": 1.25,
  "criteria_met": {
    "vc_ratio_over_1": true,
    "speed_under_20": true
  }
}
```

## 🎯 เกณฑ์การประเมิน

### การจราจรติด (Congested)
- v/c > 1.0 **หรือ** speed < 20 km/h

### การจราจรไม่ติด (Free Flow)
- v/c ≤ 1.0 **และ** speed ≥ 20 km/h

### วันหยุด/วันทำงาน
- วันหยุด: เสาร์(6), อาทิตย์(7)
- วันทำงาน: จันทร์(1) - ศุกร์(5)

## 🚀 การพัฒนาต่อ

1. **เพิ่มฟีเจอร์:** weather, events, holidays
2. **ปรับปรุงโมเดล:** Neural Networks, Ensemble Methods
3. **Real-time Data:** เชื่อมต่อข้อมูลจริง
4. **Mobile App:** แอปมือถือ
5. **API Integration:** เชื่อมต่อระบบอื่น

---

**พัฒนาโดย:** AI Assistant  
**เวอร์ชัน:** 1.0.0  
**วันที่:** 2024

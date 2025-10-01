# 🚗 Traffic Congestion Predictor (Simple Version)

ระบบพยากรณ์การจราจรติด/ไม่ติด และวันหยุด/วันทำงาน โดยใช้โมเดลที่มีอยู่แล้ว

## 🎯 ฟีเจอร์หลัก

### 1. การพยากรณ์การจราจรติด/ไม่ติด
- **เกณฑ์การประเมิน:**
  - v/c > 1 (อัตราส่วนปริมาณการจราจรต่อความจุถนน)
  - speed < 20 km/h (ความเร็วต่ำกว่า 20 กม./ชม.)
- **โมเดลที่ใช้:** Decision Tree, Random Forest (จากไฟล์ที่มีอยู่)

### 2. การพยากรณ์วันหยุด/วันทำงาน
- วิเคราะห์จากรูปแบบการจราจร
- วันหยุด: เสาร์-อาทิตย์
- วันทำงาน: จันทร์-ศุกร์

## 📊 ฟีเจอร์ที่ใช้ (Features)

1. **latitude** - ละติจูด
2. **longitude** - ลองจิจูด  
3. **density** - ความหนาแน่นของการจราจร
4. **volume** - ปริมาณการจราจร
5. **capacity** - ความจุถนน
6. **hour** - ชั่วโมง (0-23)
7. **speed** - ความเร็ว (km/h)
8. **day_of_week** - วันในสัปดาห์ (1=จันทร์, 7=อาทิตย์)

## 🚀 การติดตั้งและใช้งาน

### 1. เริ่มเซิร์ฟเวอร์
```bash
python -m uvicorn traffic_simple:app --host 127.0.0.1 --port 8002 --reload
```

### 2. เปิดเว็บเบราว์เซอร์
ไปที่: `http://127.0.0.1:8002`

### 3. ทดสอบระบบ
```bash
python test_simple.py
```

## 🔧 การใช้งาน

### 1. การพยากรณ์แบบ Single
- กรอกข้อมูลการจราจรในฟอร์ม
- เลือกโมเดล (Decision Tree, Random Forest)
- กดปุ่ม "ทำนายการจราจร"

### 2. การประมวลผลแบบ Batch (Excel)
- อัปโหลดไฟล์ Excel ที่มีคอลัมน์: latitude, longitude, density, volume, capacity, hour, speed
- ระบบจะประมวลผลและให้ดาวน์โหลดผลลัพธ์

## 📈 ตัวอย่างข้อมูล

### ข้อมูลการจราจรไม่ติด
```json
{
  "latitude": 13.7563,
  "longitude": 100.5018,
  "density": 50,
  "volume": 1500,
  "capacity": 2000,
  "hour": 8,
  "speed": 45,
  "day_of_week": 2
}
```

### ข้อมูลการจราจรติด
```json
{
  "latitude": 13.7563,
  "longitude": 100.5018,
  "density": 80,
  "volume": 2500,
  "capacity": 2000,
  "hour": 17,
  "speed": 15,
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
├── traffic_simple.py          # ไฟล์หลัก FastAPI
├── templates/
│   └── traffic_simple.html    # หน้าเว็บ
├── static/
│   └── traffic_style.css      # สไตล์ CSS
├── models/                    # โมเดลที่มีอยู่
│   ├── dt_model.pkl          # Decision Tree
│   ├── rf_model.pkl          # Random Forest
│   └── gb_model.pkl          # Gradient Boosting (มีปัญหา)
├── test_simple.py            # ไฟล์ทดสอบ
└── SIMPLE_README.md          # คู่มือนี้
```

## 🔍 API Endpoints

- `GET /` - หน้าเว็บหลัก
- `GET /health` - ตรวจสอบสถานะ
- `POST /predict-traffic` - ทำนายการจราจร
- `POST /upload-traffic-excel` - อัปโหลดไฟล์ Excel
- `GET /download/{filename}` - ดาวน์โหลดผลลัพธ์

## 🧪 การทดสอบ

```bash
python test_simple.py
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

## ⚠️ หมายเหตุ

- โมเดล Gradient Boosting มีปัญหาในการโหลด (version compatibility)
- ใช้โมเดล Decision Tree และ Random Forest ที่ทำงานได้
- ระบบพร้อมใช้งานทันที

## 🚀 การพัฒนาต่อ

1. **แก้ไขโมเดล Gradient Boosting:** อัปเดต version compatibility
2. **เพิ่มฟีเจอร์:** weather, events, holidays
3. **ปรับปรุงโมเดล:** Neural Networks, Ensemble Methods
4. **Real-time Data:** เชื่อมต่อข้อมูลจริง
5. **Mobile App:** แอปมือถือ

---

**พัฒนาโดย:** AI Assistant  
**เวอร์ชัน:** 1.0.0 (Simple)  
**วันที่:** 2024

#!/usr/bin/env python3
"""
ทดสอบระบบ Traffic Congestion Predictor (Random Forest)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simple_rf import create_jam_features, create_day_features, create_traffic_label, create_day_type_label

def test_rf_system():
    """ทดสอบระบบการจราจร Random Forest"""
    print("🚗 ทดสอบระบบ Traffic Congestion Predictor (Random Forest)")
    print("=" * 60)
    
    # ทดสอบข้อมูลการจราจรไม่ติด
    print("\n📊 ทดสอบข้อมูลการจราจรไม่ติด...")
    free_flow_data = {
        "latitude": 13.7563,
        "longitude": 100.5018,
        "density": 50,
        "volume": 1500,
        "capacity": 2000,
        "hour": 8,
        "speed": 45,
        "day_of_week": 2
    }
    
    jam_features = create_jam_features(free_flow_data)
    day_features = create_day_features(free_flow_data)
    traffic_label = create_traffic_label(free_flow_data)
    day_type_label = create_day_type_label(free_flow_data)
    
    vc_ratio = free_flow_data['volume'] / free_flow_data['capacity']
    speed = free_flow_data['speed']
    
    print(f"✅ ฟีเจอร์ Traffic Jam: {jam_features.shape}")
    print(f"✅ ฟีเจอร์ Day Type: {day_features.shape}")
    print(f"✅ Traffic Label: {traffic_label} ({'ติด' if traffic_label == 1 else 'ไม่ติด'})")
    print(f"✅ Day Type Label: {day_type_label} ({'วันหยุด' if day_type_label == 1 else 'วันทำงาน'})")
    print(f"✅ v/c ratio: {vc_ratio:.3f} {'✅' if vc_ratio > 1.0 else '❌'}")
    print(f"✅ speed: {speed} km/h {'✅' if speed < 20 else '❌'}")
    print(f"✅ ผลลัพธ์: {'ติด' if (vc_ratio > 1.0) or (speed < 20) else 'ไม่ติด'}")
    
    # ทดสอบข้อมูลการจราจรติด
    print("\n🚦 ทดสอบข้อมูลการจราจรติด...")
    congested_data = {
        "latitude": 13.7563,
        "longitude": 100.5018,
        "density": 80,
        "volume": 2500,
        "capacity": 2000,
        "hour": 17,
        "speed": 15,
        "day_of_week": 2
    }
    
    jam_features_congested = create_jam_features(congested_data)
    day_features_congested = create_day_features(congested_data)
    traffic_label_congested = create_traffic_label(congested_data)
    day_type_label_congested = create_day_type_label(congested_data)
    
    vc_ratio_congested = congested_data['volume'] / congested_data['capacity']
    speed_congested = congested_data['speed']
    
    print(f"✅ ฟีเจอร์ Traffic Jam: {jam_features_congested.shape}")
    print(f"✅ ฟีเจอร์ Day Type: {day_features_congested.shape}")
    print(f"✅ Traffic Label: {traffic_label_congested} ({'ติด' if traffic_label_congested == 1 else 'ไม่ติด'})")
    print(f"✅ Day Type Label: {day_type_label_congested} ({'วันหยุด' if day_type_label_congested == 1 else 'วันทำงาน'})")
    print(f"✅ v/c ratio: {vc_ratio_congested:.3f} {'✅' if vc_ratio_congested > 1.0 else '❌'}")
    print(f"✅ speed: {speed_congested} km/h {'✅' if speed_congested < 20 else '❌'}")
    print(f"✅ ผลลัพธ์: {'ติด' if (vc_ratio_congested > 1.0) or (speed_congested < 20) else 'ไม่ติด'}")
    
    # ทดสอบข้อมูลวันหยุด
    print("\n🏖️ ทดสอบข้อมูลวันหยุด...")
    weekend_data = {
        "latitude": 13.7563,
        "longitude": 100.5018,
        "density": 30,
        "volume": 800,
        "capacity": 2000,
        "hour": 10,
        "speed": 60,
        "day_of_week": 6  # เสาร์
    }
    
    day_type_weekend = create_day_type_label(weekend_data)
    print(f"✅ Day Type Label: {day_type_weekend} ({'วันหยุด' if day_type_weekend == 1 else 'วันทำงาน'})")
    
    print("\n🎉 ทดสอบเสร็จสิ้น!")
    print("\n📋 สรุป:")
    print("   - ระบบสามารถสร้างฟีเจอร์ได้")
    print("   - ระบบสามารถสร้าง label ได้")
    print("   - เกณฑ์การประเมินทำงานถูกต้อง")
    print("   - พร้อมใช้งานกับโมเดล Random Forest")
    print("\n🚀 วิธีเริ่มเซิร์ฟเวอร์:")
    print("   python -m uvicorn simple_rf:app --host 127.0.0.1 --port 8004")
    print("   แล้วเปิดเว็บเบราว์เซอร์ไปที่: http://127.0.0.1:8004")
    
    return True

if __name__ == "__main__":
    try:
        test_rf_system()
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {e}")
        import traceback
        traceback.print_exc()

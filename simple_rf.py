import os
import json
import joblib
import numpy as np
import pandas as pd
from io import BytesIO
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ===== FastAPI setup =====
app = FastAPI(title="Traffic Congestion Predictor", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "jam_model": "loaded" if jam_model is not None else "not_loaded",
        "day_model": "loaded" if day_model is not None else "not_loaded"
    }

# ===== Static & Templates =====
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ===== Traffic Labels =====
traffic_labels = {
    "0": "การจราจรไม่ติด (Free Flow)",
    "1": "การจราจรติด (Congested)"
}

day_type_labels = {
    "0": "วันทำงาน (Weekday)",
    "1": "วันหยุด (Weekend/Holiday)"
}

# ===== Load Models =====
# Traffic Jam Model
jam_model = None
try:
    jam_model = joblib.load("models/rf_model.pkl")
    print("✅ โหลดโมเดล Traffic Jam (Random Forest) สำเร็จ")
except Exception as e:
    print(f"❌ ไม่สามารถโหลดโมเดล Traffic Jam: {e}")

# Day Type Model (ใช้โมเดลเดียวกันสำหรับตอนนี้)
day_model = None
try:
    day_model = joblib.load("models/rf_model.pkl")
    print("✅ โหลดโมเดล Day Type (Random Forest) สำเร็จ")
except Exception as e:
    print(f"❌ ไม่สามารถโหลดโมเดล Day Type: {e}")

# ===== Feature Engineering =====
def create_jam_features(data):
    """สร้างฟีเจอร์สำหรับการทำนายการจราจรติด/ไม่ติด"""
    # Basic features for traffic jam prediction
    latitude = float(data.get("latitude", 13.7563))
    longitude = float(data.get("longitude", 100.5018))
    density = float(data.get("density", 0))
    volume = float(data.get("volume", 0))
    capacity = float(data.get("capacity", 1))
    hour = int(data.get("hour", 12))
    speed = float(data.get("speed", 0))
    vc_ratio = float(data.get("vc_ratio", 0.75))
    
    # Create feature array for traffic jam model
    features = [
        latitude, longitude, density, volume, capacity, 
        hour, speed, vc_ratio
    ]
    
    return np.array(features).reshape(1, -1)

def create_day_features(data):
    """สร้างฟีเจอร์สำหรับการทำนายวันทำงาน/วันหยุด"""
    # Basic features for day type prediction
    latitude = float(data.get("latitude", 13.7563))
    longitude = float(data.get("longitude", 100.5018))
    density = float(data.get("density", 0))
    volume = float(data.get("volume", 0))
    capacity = float(data.get("capacity", 1))
    hour = int(data.get("hour", 12))
    speed = float(data.get("speed", 0))
    vc_ratio = float(data.get("vc_ratio", 0.75))
    
    # Create feature array for day type model
    features = [
        latitude, longitude, density, volume, capacity, 
        hour, speed, vc_ratio
    ]
    
    return np.array(features).reshape(1, -1)

def create_traffic_label(data):
    """สร้าง label สำหรับการจราจรติด/ไม่ติด"""
    vc_ratio = float(data.get("vc_ratio", 0.75))
    speed = float(data.get("speed", 0))
    
    # เกณฑ์: v/c > 1 หรือ speed < 20 km/h
    if vc_ratio > 1.0 or speed < 20:
        return 1  # ติด
    else:
        return 0  # ไม่ติด

def create_day_type_label(data):
    """สร้าง label สำหรับวันหยุด/วันทำงาน"""
    day_of_week = int(data.get("day_of_week", 2))
    
    # วันหยุด: เสาร์(6), อาทิตย์(7)
    if day_of_week >= 6:
        return 1  # วันหยุด
    else:
        return 0  # วันทำงาน

# ===== Routes =====
@app.get("/", response_class=HTMLResponse)
def root(request: Request):
    return templates.TemplateResponse("simple_rf.html", {"request": request})

@app.post("/predict-traffic")
def predict_traffic(data: dict):
    """ทำนายการจราจรติด/ไม่ติด และวันทำงาน/วันหยุด"""
    try:
        if jam_model is None:
            return {"error": "ไม่พบโมเดล Traffic Jam"}
        if day_model is None:
            return {"error": "ไม่พบโมเดล Day Type"}
        
        # Create features for both models
        jam_features = create_jam_features(data)
        day_features = create_day_features(data)
        
        # Predict traffic congestion using jam model
        traffic_pred = int(jam_model.predict(jam_features)[0])
        
        # Get traffic probabilities
        try:
            traffic_proba = jam_model.predict_proba(jam_features)[0]
            traffic_proba_dict = {
                "ไม่ติด": f"{traffic_proba[0]*100:.2f}%",
                "ติด": f"{traffic_proba[1]*100:.2f}%"
            }
        except:
            traffic_proba_dict = {
                "ไม่ติด": "N/A",
                "ติด": "N/A"
            }
        
        # Predict day type using day model
        day_pred = int(day_model.predict(day_features)[0])
        
        # Get day type probabilities
        try:
            day_proba = day_model.predict_proba(day_features)[0]
            day_proba_dict = {
                "วันทำงาน": f"{day_proba[0]*100:.2f}%",
                "วันหยุด": f"{day_proba[1]*100:.2f}%"
            }
        except:
            day_proba_dict = {
                "วันทำงาน": "N/A",
                "วันหยุด": "N/A"
            }
        
        # Calculate actual congestion based on criteria
        vc_ratio = float(data.get("vc_ratio", 0.75))
        speed = float(data.get("speed", 0))
        actual_congested = 1 if (vc_ratio > 1.0) or (speed < 20) else 0
        
        return {
            "traffic_prediction": traffic_pred,
            "traffic_label": traffic_labels[str(traffic_pred)],
            "traffic_probabilities": traffic_proba_dict,
            "day_type_prediction": day_pred,
            "day_type_label": day_type_labels[str(day_pred)],
            "day_type_probabilities": day_proba_dict,
            "actual_congested": actual_congested,
            "vc_ratio": vc_ratio,
            "criteria_met": {
                "vc_ratio_over_1": vc_ratio > 1.0,
                "speed_under_20": speed < 20
            }
        }
        
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload-traffic-excel")
async def upload_traffic_excel(file: UploadFile = File(...)):
    """อัปโหลดไฟล์ Excel สำหรับการจราจร"""
    try:
        if jam_model is None:
            return {"error": "ไม่พบโมเดล Traffic Jam"}
        if day_model is None:
            return {"error": "ไม่พบโมเดล Day Type"}
            
        if not file.filename.endswith(('.xlsx', '.xls')):
            return {"error": "กรุณาอัปโหลดไฟล์ Excel (.xlsx หรือ .xls)"}
        
        contents = await file.read()
        df = pd.read_excel(BytesIO(contents))
        
        # Required columns
        required_columns = ["latitude", "longitude", "density", "volume", "capacity", "hour", "speed", "v/c"]
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            return {"error": f"คอลัมน์ที่ขาดหายไป: {', '.join(missing_columns)}"}
        
        results = []
        for index, row in df.iterrows():
            try:
                # Create features for both models
                jam_features = create_jam_features(row.to_dict())
                day_features = create_day_features(row.to_dict())
                
                # Predict traffic jam
                jam_pred = int(jam_model.predict(jam_features)[0])
                try:
                    jam_proba = jam_model.predict_proba(jam_features)[0]
                    jam_proba_dict = {
                        "ไม่ติด": f"{jam_proba[0]*100:.2f}%",
                        "ติด": f"{jam_proba[1]*100:.2f}%"
                    }
                except:
                    jam_proba_dict = {"ไม่ติด": "N/A", "ติด": "N/A"}
                
                # Predict day type
                day_pred = int(day_model.predict(day_features)[0])
                try:
                    day_proba = day_model.predict_proba(day_features)[0]
                    day_proba_dict = {
                        "วันทำงาน": f"{day_proba[0]*100:.2f}%",
                        "วันหยุด": f"{day_proba[1]*100:.2f}%"
                    }
                except:
                    day_proba_dict = {"วันทำงาน": "N/A", "วันหยุด": "N/A"}
                
                # Actual congestion
                vc_ratio = float(row.get("v/c", 0.75))
                speed = float(row.get("speed", 0))
                actual_congested = 1 if (vc_ratio > 1.0) or (speed < 20) else 0
                
                results.append({
                    "row": index + 1,
                    "input_data": row.to_dict(),
                    "traffic_prediction": jam_pred,
                    "traffic_label": traffic_labels[str(jam_pred)],
                    "traffic_probabilities": jam_proba_dict,
                    "day_type_prediction": day_pred,
                    "day_type_label": day_type_labels[str(day_pred)],
                    "day_type_probabilities": day_proba_dict,
                    "actual_congested": actual_congested,
                    "vc_ratio": vc_ratio,
                    "criteria_met": {
                        "vc_ratio_over_1": vc_ratio > 1.0,
                        "speed_under_20": speed < 20
                    }
                })
                
            except Exception as e:
                results.append({
                    "row": index + 1,
                    "error": str(e),
                    "input_data": row.to_dict()
                })
        
        # Save results
        output_df = pd.DataFrame(results)
        output_filename = f"traffic_results_{file.filename}"
        output_path = f"static/{output_filename}"
        output_df.to_excel(output_path, index=False)
        
        return {
            "message": "ประมวลผลข้อมูลการจราจรสำเร็จ",
            "total_rows": len(df),
            "successful_predictions": len([r for r in results if "error" not in r]),
            "errors": len([r for r in results if "error" in r]),
            "download_url": f"/download/{output_filename}",
            "results": results[:10]
        }
        
    except Exception as e:
        return {"error": f"เกิดข้อผิดพลาด: {str(e)}"}

@app.get("/download/{filename}")
async def download_file(filename: str):
    """ดาวน์โหลดไฟล์ผลลัพธ์"""
    file_path = f"static/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, filename=filename)
    else:
        return {"error": "ไฟล์ไม่พบ"}

# ===== Initialize =====
if __name__ == "__main__":
    # Create directories
    os.makedirs("static", exist_ok=True)
    
    print("🌐 ระบบ Traffic Congestion Predictor (Dual Models) พร้อมใช้งาน!")
    if jam_model is not None:
        print("✅ โมเดล Traffic Jam โหลดสำเร็จ")
    else:
        print("❌ ไม่พบโมเดล Traffic Jam")
    
    if day_model is not None:
        print("✅ โมเดล Day Type โหลดสำเร็จ")
    else:
        print("❌ ไม่พบโมเดล Day Type")

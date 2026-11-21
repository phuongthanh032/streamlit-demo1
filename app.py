import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Hệ thống Đánh giá Sinh lý ECG/HRV", layout="wide")

# Định nghĩa chuẩn y tế sinh lý
STANDARDS = {
    "ecg_hr_calc": {"min": 60, "max": 100, "name": "Nhịp tim (BPM)", "unit": "BPM"},
    "ecg_sdnn_ms": {"min": 30, "max": 999, "name": "SDNN (ms)", "unit": "ms"},
    "ecg_rmssd_ms": {"min": 15, "max": 999, "name": "RMSSD (ms)", "unit": "ms"},
    "ecg_lf_hf_ratio": {"min": 0.5, "max": 2.0, "name": "LF/HF Ratio", "unit": ""}
}

def evaluate_metric(value, std_min, std_max):
    """Hàm đánh giá chỉ số dựa trên ngưỡng chuẩn"""
    if std_min <= value <= std_max:
        return "Bình thường ✅", "normal"
    elif value < std_min:
        return f"Thấp (Dưới {std_min}) ⚠️", "low"
    else:
        return f"Cao (Trên {std_max if std_max != 999 else ''}) ⚠️", "high"

st.title("Hệ thống Đánh giá Chỉ số Y Sinh học - Căng thẳng & Vận động")

# ==========================================
# TÍNH NĂNG 1: NHẬP LIỆU THỦ CÔNG (1 MẪU)
# ==========================================
st.sidebar.header("🔍 Nhập thông số thủ công")
st.sidebar.write("Nhập số liệu của 1 cá nhân để đánh giá nhanh:")

with st.sidebar.form("manual_input_form"):
    input_hr = st.number_input("Nhịp tim (BPM)", min_value=30, max_value=200, value=75)
    input_sdnn = st.number_input("SDNN (ms)", min_value=0.0, max_value=300.0, value=40.0)
    input_rmssd = st.number_input("RMSSD (ms)", min_value=0.0, max_value=300.0, value=30.0)
    input_lfhf = st.number_input("LF/HF Ratio", min_value=0.0, max_value=10.0, value=1.5)
    
    submit_manual = st.form_submit_button("Đánh giá ngay")

if submit_manual:
    st.subheader("Kết quả Đánh giá Cá nhân")
    col1, col2, col3, col4 = st.columns(4)
    
    metrics = [
        (input_hr, "ecg_hr_calc", col1),
        (input_sdnn, "ecg_sdnn_ms", col2),
        (input_rmssd, "ecg_rmssd_ms", col3),
        (input_lfhf, "ecg_lf_hf_ratio", col4)
    ]
    
    for val, key, col in metrics:
        std = STANDARDS[key]
        status, condition = evaluate_metric(val, std["min"], std["max"])
        col.metric(label=std["name"], value=f"{val} {std['unit']}")
        if condition == "normal":
            col.success(status)
        else:
            col.warning(status)
            
    st.markdown("---")

# ==========================================
# TÍNH NĂNG 2: XỬ LÝ DỮ LIỆU TỪ FILE CSV
# ==========================================
st.subheader("📁 Tải file CSV để phân tích hàng loạt")
uploaded_file = st.file_uploader("Tải lên file dữ liệu (ví dụ: qol_stress_merged_features_3.csv)", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    
    # Tiền xử lý dữ liệu
    with st.spinner('Hệ thống đang tự động lọc nhiễu và xử lý dữ liệu thiếu...'):
        df_clean = df_raw.copy()
        
        # 1. Điền dữ liệu thiếu bằng trung vị (Median)
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].apply(lambda x: x.fillna(x.median()))
        
        # 2. Lọc nhiễu ngoại lai bằng Z-score (Ngưỡng = 3)
        z_scores = np.abs(stats.zscore(df_clean[numeric_cols]))
        df_clean = df_clean[(z_scores < 3).all(axis=1)]
        
    st.success(f"Tiền xử lý hoàn tất! Đã giữ lại {len(df_clean)} mẫu hợp lệ (từ {len(df_raw)} mẫu gốc).")
    
    st.write("**Bảng dữ liệu sau khi làm sạch:**")
    st.dataframe(df_clean.head())
    
    # Phân tích tổng quan theo chuẩn
    st.subheader("📊 Đối chiếu Chỉ số Tổng quan")
    
    cols_to_analyze = [col for col in STANDARDS.keys() if col in df_clean.columns]
    
    if not cols_to_analyze:
        st.error("Không tìm thấy các cột chuẩn (ecg_hr_calc, ecg_sdnn_ms, ecg_rmssd_ms, ecg_lf_hf_ratio) trong tệp của bạn.")
    else:
        # Xác định cột phân nhóm (Phase hoặc Label)
        group_col = 'phase' if 'phase' in df_clean.columns else ('label' if 'label' in df_clean.columns else None)
        
        for col in cols_to_analyze:
            std = STANDARDS[col]
            mean_val = df_clean[col].mean()
            
            st.markdown(f"#### {std['name']}")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(label="Giá trị Trung bình Toàn cục", value=f"{mean_val:.2f} {std['unit']}")
                status, condition = evaluate_metric(mean_val, std["min"], std["max"])
                if condition == "normal":
                    st.success(f"Quần thể chung: {status}")
                else:
                    st.warning(f"Quần thể chung: {status}")
                    
            with col2:
                fig, ax = plt.subplots(figsize=(6, 3))
                if group_col:
                    sns.boxplot(data=df_clean, x=group_col, y=col, ax=ax, palette="muted")
                else:
                    sns.histplot(df_clean[col], kde=True, ax=ax, color="teal")
                
                # Vẽ đường ranh giới chuẩn
                ax.axhline(std["min"], color='red', linestyle='--', label=f'Ngưỡng dưới ({std["min"]})')
                if std["max"] != 999:
                    ax.axhline(std["max"], color='orange', linestyle='--', label=f'Ngưỡng trên ({std["max"]})')
                
                ax.legend()
                st.pyplot(fig)

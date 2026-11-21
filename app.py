import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Phân tích ECG & HRV", layout="wide")

st.title("Hệ thống Phân tích & Đánh giá Chỉ số ECG / HRV")
st.write("Tải lên dữ liệu CSV của bạn để tự động tiền xử lý và đối chiếu với chuẩn y tế sinh lý.")

# Tiêu chuẩn y tế (Short-term resting)
STANDARDS = {
    "ecg_hr_calc": {"min": 60, "max": 100, "name": "Nhịp tim (BPM)"},
    "ecg_sdnn_ms": {"min": 30, "max": 999, "name": "SDNN (ms) - Sức khỏe tổng thể"},
    "ecg_rmssd_ms": {"min": 15, "max": 999, "name": "RMSSD (ms) - Mức độ thư giãn"},
    "ecg_lf_hf_ratio": {"min": 0.5, "max": 2.0, "name": "LF/HF Ratio - Cân bằng thần kinh"}
}

uploaded_file = st.file_uploader("Tải lên tệp CSV chứa dữ liệu ECG/HRV", type=["csv"])

if uploaded_file is not None:
    # 1. Đọc dữ liệu
    df_raw = pd.read_csv(uploaded_file)
    st.subheader("1. Dữ liệu thô ban đầu")
    st.write(df_raw.head())

    # 2. Tiền xử lý dữ liệu
    st.subheader("2. Tiền xử lý dữ liệu")
    with st.spinner('Đang xử lý dữ liệu (Xóa NaN, Lọc nhiễu)...'):
        # Tạo bản sao để xử lý
        df_clean = df_raw.copy()
        
        # Bước A: Xử lý giá trị thiếu (Missing values) - Điền bằng giá trị trung vị của cột
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].apply(lambda x: x.fillna(x.median()))
        
        # Bước B: Lọc nhiễu ngoại lai (Outliers) bằng phương pháp Z-score (Ngưỡng = 3)
        # Nếu giá trị chênh lệch quá 3 lần độ lệch chuẩn, xem như là nhiễu do cử động/lỗi cảm biến
        z_scores = np.abs(stats.zscore(df_clean[numeric_cols]))
        df_clean = df_clean[(z_scores < 3).all(axis=1)]
        
        st.success(f"Hoàn thành! Đã giữ lại {len(df_clean)} mẫu dữ liệu hợp lệ (từ {len(df_raw)} mẫu ban đầu).")
    
    # 3. Phân tích và Đối chiếu chuẩn y tế
    st.subheader("3. Đánh giá chỉ số theo Chuẩn Y Tế Sinh Lý")
    
    # Lấy các cột có trong dataset trùng khớp với các chuẩn y tế đã định nghĩa
    cols_to_analyze = [col for col in STANDARDS.keys() if col in df_clean.columns]
    
    if not cols_to_analyze:
        st.warning("Không tìm thấy các cột chỉ số (ecg_hr_calc, ecg_sdnn_ms, ecg_rmssd_ms, ecg_lf_hf_ratio) trong tệp của bạn.")
    else:
        # Nhóm theo trạng thái/pha (nếu có cột label hoặc phase)
        group_col = None
        if 'label' in df_clean.columns:
            group_col = 'label'
        elif 'phase' in df_clean.columns:
            group_col = 'phase'

        for col in cols_to_analyze:
            std_min = STANDARDS[col]['min']
            std_max = STANDARDS[col]['max']
            std_name = STANDARDS[col]['name']
            
            mean_val = df_clean[col].mean()
            
            st.markdown(f"#### {std_name}")
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.metric(label=f"Giá trị trung bình", value=f"{mean_val:.2f}")
                if std_min <= mean_val <= std_max:
                    st.info(f"✅ Đạt chuẩn sinh lý bình thường (Khoảng chuẩn: {std_min} - {std_max if std_max != 999 else 'Trở lên'})")
                elif mean_val < std_min:
                    st.error(f"⚠️ Cảnh báo: Thấp hơn chuẩn ({std_min})")
                else:
                    st.error(f"⚠️ Cảnh báo: Cao hơn chuẩn ({std_max})")
            
            with col2:
                fig, ax = plt.subplots(figsize=(6, 3))
                if group_col:
                    sns.boxplot(data=df_clean, x=group_col, y=col, ax=ax, palette="Set2")
                else:
                    sns.histplot(df_clean[col], kde=True, ax=ax, color="skyblue")
                
                # Vẽ đường ranh giới chuẩn y tế
                ax.axhline(std_min, color='red', linestyle='--', label='Ngưỡng dưới')
                if std_max != 999:
                    ax.axhline(std_max, color='orange', linestyle='--', label='Ngưỡng trên')
                ax.legend()
                st.pyplot(fig)

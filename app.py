import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

st.set_page_config(page_title="Phân tích ECG & HRV", layout="wide")

# ==========================================
# CẤU HÌNH CHUẨN Y TẾ SINH LÝ
# ==========================================
STANDARDS = {
    "ecg_hr_calc": {"min": 60, "max": 100, "name": "Nhịp tim (BPM)", "unit": "BPM"},
    "ecg_sdnn_ms": {"min": 30, "max": 999, "name": "SDNN - Sức khỏe tổng thể", "unit": "ms"},
    "ecg_rmssd_ms": {"min": 15, "max": 999, "name": "RMSSD - Phản xạ thư giãn", "unit": "ms"},
    "ecg_lf_hf_ratio": {"min": 0.5, "max": 2.0, "name": "Tỷ lệ LF/HF", "unit": ""}
}

def evaluate_metric(value, std_min, std_max):
    """Hàm đánh giá chỉ số dựa trên ngưỡng chuẩn"""
    if std_min <= value <= std_max:
        return "Bình thường ✅", "normal"
    elif value < std_min:
        return f"Thấp (Dưới mức {std_min}) ⚠️", "low"
    else:
        return f"Cao (Vượt mức {std_max if std_max != 999 else ''}) ⚠️", "high"

st.title("Hệ thống Đánh giá Sinh lý học: Căng thẳng & Vận động")
st.info("⚠️ Lưu ý: Kết quả dựa trên biến thiên nhịp tim (HRV) dùng để đánh giá mức độ căng thẳng thần kinh tự chủ, không thay thế chẩn đoán bệnh lý tim mạch lâm sàng (như nhồi máu cơ tim, rối loạn nhịp tim).")

# ==========================================
# PHẦN 1: GIẢI THÍCH CÁC THÔNG SỐ (YÊU CẦU MỚI)
# ==========================================
with st.expander("📚 BẤM VÀO ĐÂY ĐỂ XEM GIẢI THÍCH CÁC THÔNG SỐ Y SINH CẦN NHẬP", expanded=False):
    st.markdown("""
    *   **Nhịp tim (BPM - Beats Per Minute):** Tốc độ đập của tim. Lúc nghỉ ngơi bình thường dao động từ 60-100 BPM. Dưới 60 là nhịp chậm (hoặc thể lực tốt), trên 100 là nhịp nhanh (dấu hiệu hồi hộp, stress hoặc đang vận động).
    *   **SDNN (ms):** Độ lệch chuẩn của khoảng cách giữa các nhịp tim. Đây là thước đo **sức khỏe tổng thể và khả năng chống chịu stress**. Dưới 30ms cho thấy cơ thể đang kiệt sức hoặc stress nặng; trên 50ms là rất tốt.
    *   **RMSSD (ms):** Thước đo hoạt động của hệ thần kinh phó giao cảm (hệ thống giúp cơ thể nghỉ ngơi, tiêu hóa). Chỉ số này phản ánh **mức độ thư giãn**. Càng cao càng tốt (>15ms). Nếu rất thấp, bạn đang ở trạng thái căng thẳng hoặc thiếu ngủ.
    *   **Tỷ lệ LF/HF:** Tỷ lệ giữa dải tần số thấp và cao của nhịp tim, đại diện cho cán cân căng thẳng/thư giãn. Ngưỡng bình thường là 0.5 - 2.0. Nếu **vượt mức 2.0**, cơ thể đang ở trạng thái "chiến đấu hoặc bỏ chạy" (stress nặng hoặc đang vận động mạnh).
    """)

# ==========================================
# PHẦN 2: NHẬP LIỆU THỦ CÔNG & CHẨN ĐOÁN
# ==========================================
st.header("1. Đánh giá nhanh từ bảng thông số mới")
st.write("Nhập thông số của một đối tượng để hệ thống tự động đối chiếu với chuẩn y khoa.")

with st.form("manual_input_form"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        input_hr = st.number_input("Nhịp tim (ecg_hr_calc)", value=85.0)
    with col2:
        input_sdnn = st.number_input("SDNN (ecg_sdnn_ms)", value=25.0)
    with col3:
        input_rmssd = st.number_input("RMSSD (ecg_rmssd_ms)", value=12.0)
    with col4:
        input_lfhf = st.number_input("Tỷ lệ LF/HF (ecg_lf_hf_ratio)", value=2.5)
    
    submit_manual = st.form_submit_button("Tiến hành Chẩn đoán/Đánh giá")

if submit_manual:
    st.subheader("Báo cáo Chẩn đoán Trạng thái Sinh lý")
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    
    metrics = [
        (input_hr, "ecg_hr_calc", r_col1),
        (input_sdnn, "ecg_sdnn_ms", r_col2),
        (input_rmssd, "ecg_rmssd_ms", r_col3),
        (input_lfhf, "ecg_lf_hf_ratio", r_col4)
    ]
    
    # Biến cờ để đánh giá stress tổng thể
    stress_markers = 0 
    
    for val, key, col in metrics:
        std = STANDARDS[key]
        status, condition = evaluate_metric(val, std["min"], std["max"])
        col.metric(label=std["name"], value=f"{val} {std['unit']}")
        
        if condition == "normal":
            col.success(status)
        else:
            col.error(status)
            if key in ["ecg_sdnn_ms", "ecg_rmssd_ms"] and condition == "low":
                stress_markers += 1
            if key == "ecg_lf_hf_ratio" and condition == "high":
                stress_markers += 1

    # Kết luận hệ thống
    if stress_markers >= 2:
        st.warning("⚠️ **KẾT LUẬN:** Dựa trên các chỉ số HRV, hệ thần kinh tự chủ của đối tượng đang mất cân bằng. Có dấu hiệu suy giảm thể lực, mệt mỏi hoặc đang chịu căng thẳng (stress) mức độ cao. Khuyến nghị nghỉ ngơi và theo dõi thêm.")
    else:
        st.success("✅ **KẾT LUẬN:** Các chỉ số sinh lý nằm trong ngưỡng an toàn. Hệ thần kinh tự chủ hoạt động ổn định, khả năng phục hồi và chống chịu stress tốt.")

st.markdown("---")

# ==========================================
# PHẦN 3: XỬ LÝ HÀNG LOẠT TỪ FILE CSV
# ==========================================
st.header("2. Phân tích & Xử lý nhiễu hàng loạt từ file dữ liệu")
uploaded_file = st.file_uploader("Tải lên tệp CSV chứa dữ liệu của bạn", type=["csv"])

if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    
    with st.spinner('Hệ thống đang tự động tiền xử lý (Điền khuyết, Lọc nhiễu Z-score)...'):
        df_clean = df_raw.copy()
        
        # Tiền xử lý 1: Điền dữ liệu thiếu (Missing values) bằng trung vị
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].apply(lambda x: x.fillna(x.median()))
        
        # Tiền xử lý 2: Lọc nhiễu (Outliers) bằng Z-score (loại bỏ dữ liệu lệch > 3 độ lệch chuẩn)
        z_scores = np.abs(stats.zscore(df_clean[numeric_cols]))
        df_clean = df_clean[(z_scores < 3).all(axis=1)]
        
    st.success(f"Đã xử lý xong! Dữ liệu làm sạch giữ lại {len(df_clean)}/{len(df_raw)} mẫu (đã loại bỏ các đoạn nhiễu do cử động tay).")
    
    # Đối chiếu biểu đồ
    st.subheader("📊 Biểu đồ phân bố so với Chuẩn Y khoa")
    cols_to_analyze = [col for col in STANDARDS.keys() if col in df_clean.columns]
    
    if not cols_to_analyze:
        st.error("File CSV của bạn không chứa các cột chuẩn như ecg_hr_calc, ecg_sdnn_ms...")
    else:
        for col in cols_to_analyze:
            std = STANDARDS[col]
            mean_val = df_clean[col].mean()
            
            st.markdown(f"#### {std['name']} (Trung bình toàn nhóm: {mean_val:.2f})")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df_clean[col], kde=True, ax=ax, color="cornflowerblue")
            
            # Vẽ ranh giới chuẩn y khoa
            ax.axvline(std["min"], color='red', linestyle='--', linewidth=2, label=f'Ngưỡng dưới ({std["min"]})')
            if std["max"] != 999:
                ax.axvline(std["max"], color='orange', linestyle='--', linewidth=2, label=f'Ngưỡng trên ({std["max"]})')
            
            ax.set_title(f"Phân bố {std['name']} của tập dữ liệu")
            ax.legend()
            st.pyplot(fig)

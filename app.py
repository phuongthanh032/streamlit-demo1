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

def generate_doctor_advice(conditions):
    """Hàm tự động đưa ra lời khuyên dựa trên trạng thái bất thường của các chỉ số"""
    advices = []
    
    # Đánh giá nhịp tim
    if conditions.get("ecg_hr_calc") == "high":
        advices.append("🫀 **Nhịp tim lúc nghỉ quá cao (>100 BPM):** Hạn chế ngay các chất kích thích như caffeine (cà phê, trà đặc, nước tăng lực). Hãy uống đủ nước và nằm nghỉ ngơi. Nếu nhịp tim vẫn cao liên tục kèm theo khó thở, tức ngực hoặc chóng mặt, bạn cần đến bệnh viện gặp bác sĩ tim mạch để đo ECG 12 chuyển đạo ngay.")
    elif conditions.get("ecg_hr_calc") == "low":
        advices.append("🫀 **Nhịp tim thấp (<60 BPM):** Nếu bạn là người thường xuyên tập thể thao cường độ cao (vận động viên), đây có thể là biểu hiện của một trái tim khỏe. Tuy nhiên, nếu bạn ít vận động mà nhịp tim thấp kèm theo hay xây xẩm, ngất xỉu, mệt mỏi, đây là dấu hiệu của nhịp chậm bệnh lý, cần thăm khám sớm.")

    # Đánh giá SDNN
    if conditions.get("ecg_sdnn_ms") == "low":
        advices.append("🔋 **Sức khỏe tổng thể & Sức bền kém (SDNN thấp):** Cơ thể bạn đang chịu áp lực mãn tính hoặc kiệt sức. Cần ưu tiên ngủ đủ 7-8 tiếng/đêm. Hãy bắt đầu tập thể dục nhịp điệu (đi bộ, đạp xe, bơi lội) ở mức độ nhẹ nhàng để nâng cao dần thể lực hệ tim mạch.")

    # Đánh giá RMSSD
    if conditions.get("ecg_rmssd_ms") == "low":
        advices.append("🧘 **Khả năng thư giãn kém (RMSSD thấp):** Hệ phó giao cảm đang hoạt động yếu, cơ thể bạn đang không được phục hồi đúng cách. Lời khuyên: Hãy thực hành các bài tập hít thở sâu (như phương pháp thở 4-7-8), thiền định, hoặc yoga. Tuyệt đối không tập thể dục cường độ cao (HIIT/tập tạ nặng) trong 24-48 giờ tới để cơ thể nghỉ ngơi.")

    # Đánh giá LF/HF
    if conditions.get("ecg_lf_hf_ratio") == "high":
        advices.append("🧠 **Căng thẳng thần kinh cao (LF/HF cao):** Hệ thần kinh giao cảm (trạng thái 'chiến đấu hoặc bỏ chạy') đang bị kích thích quá mức. Cần cắt giảm khối lượng công việc ngay lập tức, rời xa màn hình thiết bị điện tử ít nhất 1 giờ trước khi ngủ và tìm không gian yên tĩnh để thả lỏng trí óc.")

    # Nếu tất cả bình thường
    if not advices:
        advices.append("🌟 **Tuyệt vời!** Các chỉ số sinh lý của bạn đều nằm trong ngưỡng an toàn và lý tưởng. Khả năng phục hồi, chống chịu stress và sức khỏe tim mạch của bạn đang rất tốt. Hãy tiếp tục duy trì chế độ sinh hoạt, dinh dưỡng và tập luyện hiện tại!")
        
    return advices

st.title("Hệ thống Đánh giá Sinh lý học: Căng thẳng & Vận động")
st.info("⚠️ Lưu ý: Kết quả dựa trên biến thiên nhịp tim (HRV) dùng để đánh giá mức độ căng thẳng thần kinh tự chủ, không thay thế chẩn đoán bệnh lý tim mạch lâm sàng.")

# ==========================================
# PHẦN 1: GIẢI THÍCH CÁC THÔNG SỐ
# ==========================================
with st.expander("📚 BẤM VÀO ĐÂY ĐỂ XEM GIẢI THÍCH CÁC THÔNG SỐ Y SINH CẦN NHẬP", expanded=False):
    st.markdown("""
    *   **Nhịp tim (BPM):** Tốc độ đập của tim. Lúc nghỉ ngơi bình thường dao động từ 60-100 BPM.
    *   **SDNN (ms):** Thước đo sức khỏe tổng thể. Dưới 30ms cho thấy cơ thể đang kiệt sức hoặc stress nặng; trên 50ms là rất tốt.
    *   **RMSSD (ms):** Thước đo mức độ thư giãn. Càng cao càng tốt (>15ms).
    *   **Tỷ lệ LF/HF:** Ngưỡng bình thường là 0.5 - 2.0. Vượt mức 2.0, cơ thể đang ở trạng thái căng thẳng hoặc vận động mạnh.
    """)

# ==========================================
# PHẦN 2: NHẬP LIỆU THỦ CÔNG & CHẨN ĐOÁN
# ==========================================
st.header("1. Đánh giá & Khuyến nghị Lâm sàng")
st.write("Nhập thông số của một đối tượng để hệ thống tự động đối chiếu và đưa ra lời khuyên.")

with st.form("manual_input_form"):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        input_hr = st.number_input("Nhịp tim (ecg_hr_calc)", value=105.0)
    with col2:
        input_sdnn = st.number_input("SDNN (ecg_sdnn_ms)", value=25.0)
    with col3:
        input_rmssd = st.number_input("RMSSD (ecg_rmssd_ms)", value=12.0)
    with col4:
        input_lfhf = st.number_input("Tỷ lệ LF/HF (ecg_lf_hf_ratio)", value=2.5)
    
    submit_manual = st.form_submit_button("Tiến hành Đánh giá")

if submit_manual:
    st.subheader("📊 Kết quả Đánh giá Sinh lý")
    r_col1, r_col2, r_col3, r_col4 = st.columns(4)
    
    metrics = [
        (input_hr, "ecg_hr_calc", r_col1),
        (input_sdnn, "ecg_sdnn_ms", r_col2),
        (input_rmssd, "ecg_rmssd_ms", r_col3),
        (input_lfhf, "ecg_lf_hf_ratio", r_col4)
    ]
    
    # Lưu trữ trạng thái (low, normal, high) để đưa vào hàm lời khuyên
    patient_conditions = {}
    stress_markers = 0 
    
    for val, key, col in metrics:
        std = STANDARDS[key]
        status, condition = evaluate_metric(val, std["min"], std["max"])
        
        # Ghi nhận trạng thái
        patient_conditions[key] = condition
        
        # Hiển thị số liệu
        col.metric(label=std["name"], value=f"{val} {std['unit']}")
        
        if condition == "normal":
            col.success(status)
        else:
            col.error(status)
            if key in ["ecg_sdnn_ms", "ecg_rmssd_ms"] and condition == "low":
                stress_markers += 1
            if key == "ecg_lf_hf_ratio" and condition == "high":
                stress_markers += 1

    st.markdown("---")
    
    # IN RA LỜI KHUYÊN TỪ BÁC SĨ
    st.subheader("👨‍⚕️ Lời khuyên & Chỉ định từ Chuyên gia")
    
    # Kết luận tổng thể
    if stress_markers >= 2:
        st.error("🚨 **KẾT LUẬN TỔNG QUAN:** Cơ thể đang ở trạng thái báo động do mất cân bằng thần kinh tự chủ, suy kiệt thể lực hoặc căng thẳng tột độ.")
    else:
        st.success("✅ **KẾT LUẬN TỔNG QUAN:** Hệ thần kinh tự chủ hoạt động cơ bản ổn định.")

    # In chi tiết từng lời khuyên
    advices = generate_doctor_advice(patient_conditions)
    for advice in advices:
        st.info(advice)

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
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        df_clean[numeric_cols] = df_clean[numeric_cols].apply(lambda x: x.fillna(x.median()))
        z_scores = np.abs(stats.zscore(df_clean[numeric_cols]))
        df_clean = df_clean[(z_scores < 3).all(axis=1)]
        
    st.success(f"Đã xử lý xong! Dữ liệu làm sạch giữ lại {len(df_clean)}/{len(df_raw)} mẫu.")
    
    st.subheader("📊 Biểu đồ phân bố so với Chuẩn Y khoa")
    cols_to_analyze = [col for col in STANDARDS.keys() if col in df_clean.columns]
    
    if not cols_to_analyze:
        st.error("File CSV của bạn không chứa các cột chuẩn.")
    else:
        for col in cols_to_analyze:
            std = STANDARDS[col]
            mean_val = df_clean[col].mean()
            
            st.markdown(f"#### {std['name']} (Trung bình toàn nhóm: {mean_val:.2f})")
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.histplot(df_clean[col], kde=True, ax=ax, color="cornflowerblue")
            ax.axvline(std["min"], color='red', linestyle='--', linewidth=2, label=f'Ngưỡng dưới ({std["min"]})')
            if std["max"] != 999:
                ax.axvline(std["max"], color='orange', linestyle='--', linewidth=2, label=f'Ngưỡng trên ({std["max"]})')
            ax.legend()
            st.pyplot(fig)

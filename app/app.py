import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.predict import predict_text
from src.risk_assessment import assess_risk, get_random_message

st.set_page_config(
    page_title="Mental Health Detector",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    :root {
        --primary-color: #6366f1;
        --secondary-color: #8b5cf6;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --danger-color: #ef4444;
    }
    
    /* Animated Background */
    @keyframes gradientShift {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-20px); }
    }
    
    @keyframes glistening {
        0% {
            opacity: 0;
            transform: translateX(-100%);
        }
        50% {
            opacity: 1;
        }
        100% {
            opacity: 0;
            transform: translateX(100%);
        }
    }
    
    @keyframes pulse-glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(99, 102, 241, 0.3), 0 4px 15px rgba(99, 102, 241, 0.2);
        }
        50% {
            box-shadow: 0 0 40px rgba(139, 92, 246, 0.5), 0 4px 25px rgba(99, 102, 241, 0.3);
        }
    }
    
    /* Main page background */
    .stApp {
        background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
        min-height: 100vh;
    }
    
    .main-container {
        max-width: 900px;
        margin: 0 auto;
    }
    
    .header-section {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
        background-size: 200% 200%;
        animation: gradientShift 8s ease infinite, float 3s ease-in-out infinite, pulse-glow 3s ease-in-out infinite;
        padding: 40px 20px;
        border-radius: 20px;
        color: white;
        margin-bottom: 30px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    
    /* Glistening effect on header */
    .header-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, 
            transparent, 
            rgba(255, 255, 255, 0.3), 
            transparent);
        animation: glistening 3s ease-in-out infinite;
    }
    
    .header-section h1 {
        position: relative;
        margin: 0;
        font-size: 2.5em;
        font-weight: 700;
        z-index: 1;
    }
    
    .header-section p {
        position: relative;
        margin: 10px 0 0 0;
        font-size: 1.1em;
        opacity: 0.95;
        z-index: 1;
    }
    
    .header-section h1 {
        margin: 0;
        font-size: 2.5em;
    }
    
    .input-section {
        background: rgba(248, 250, 252, 0.95);
        backdrop-filter: blur(10px);
        padding: 25px;
        border-radius: 15px;
        border: 2px solid rgba(226, 232, 240, 0.5);
        margin-bottom: 25px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.1);
        animation: float 4s ease-in-out infinite;
    }
    
    .result-container {
        background: rgba(255, 255, 255, 0.97);
        backdrop-filter: blur(10px);
        padding: 30px;
        border-radius: 15px;
        border-left: 5px solid #6366f1;
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
        animation: float 5s ease-in-out infinite;
    }
    
    .result-high {
        border-left-color: #ef4444;
        background: rgba(254, 242, 242, 0.97);
        box-shadow: 0 8px 32px rgba(239, 68, 68, 0.15);
    }
    
    .result-medium {
        border-left-color: #f59e0b;
        background: rgba(255, 251, 235, 0.97);
        box-shadow: 0 8px 32px rgba(245, 158, 11, 0.15);
    }
    
    .result-low {
        border-left-color: #10b981;
        background: rgba(240, 253, 244, 0.97);
        box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15);
    }
    
    .metric-row {
        display: flex;
        gap: 20px;
        margin: 20px 0;
    }
    
    .metric-box {
        flex: 1;
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(10px);
        padding: 15px;
        border-radius: 12px;
        border: 2px solid rgba(226, 232, 240, 0.5);
        text-align: center;
        transition: all 0.3s ease;
        animation: float 6s ease-in-out infinite;
    }
    
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(99, 102, 241, 0.2);
        border-color: #6366f1;
    }
    
    .metric-label {
        font-size: 0.85em;
        color: #64748b;
        margin-bottom: 8px;
        font-weight: 600;
    }
    
    .metric-value {
        font-size: 1.8em;
        font-weight: bold;
        color: #1e293b;
    }
    </style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
    <div class="header-section">
        <h1>💬 Tâm An </h1>
        <p>Hệ thống phát hiện rủi ro cảm xúc qua ngôn ngữ viết dựa trên AI</p>
    </div>
""", unsafe_allow_html=True)

st.markdown("""
<p style="text-align: center; color: #64748b; margin-bottom: 30px; font-size: 0.95em;">
🔍 Nhập đoạn văn để phân tích mức rủi ro | ⚠️ Không thay thế tư vấn chuyên gia tâm lý
</p>
""", unsafe_allow_html=True)

text = st.text_area("Nhập đoạn văn (50–150 chữ):", height=200)

if st.button("Phân tích"):
    if not text.strip():
        st.warning("Vui lòng nhập đoạn văn.")
    else:
        with st.spinner("Đang xử lý..."):
            try:
                result = predict_text(text)
                label = result["label"]
                prob = result["probability"]

                level = assess_risk(prob)
                advice = get_random_message(level)

                # HIỂN THỊ
                st.subheader("Kết quả phân tích")
                st.write(f"**Nhãn mô hình:** {label}")
                st.write(f"**Xác suất self-harm:** `{prob:.3f}`")

                if level == "high":
                    st.error("Mức rủi ro: **CAO**")
                elif level == "medium":
                    st.warning("Mức rủi ro: **TRUNG BÌNH**")
                else:
                    st.success("Mức rủi ro: **THẤP**")

                st.subheader("💡 Lời khuyên")
                st.info(advice)

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

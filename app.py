import streamlit as st
import pandas as pd
import joblib

#Load Model
@st.cache_resource
def load_model():
    # Load model
    model = joblib.load('best_model_rf.pkl')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Gagal memuat model. Error: {e}")
#Judul dan Deskripsi
st.title("UAS BENGKEL KODING")
st.write("""
Aplikasi untuk demonstrasi
""")

#Form Input Fitur
col1, col2 = st.columns(2)

with col1:
    st.header("Profil Pelanggan")
    gender = st.selectbox("Gender", ['Male', 'Female'])
    senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak")
    partner = st.selectbox("Partner", ['Yes', 'No'])
    dependents = st.selectbox("Dependents", ['Yes', 'No'])
    tenure = st.number_input("Lama Berlangganan (Bulan)", min_value=0, max_value=100, value=12)

    st.header("Info Tagihan")
    contract = st.selectbox("Contract Type", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
    payment_method = st.selectbox("Payment Method", [
        'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
    ])
    monthly_charges = st.number_input("Biaya Bulanan (Monthly Charges)", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Biaya (Total Charges)", min_value=0.0, value=500.0)

with col2:
    st.header("Layanan Pelanggan")
    phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    

    online_security = st.selectbox("Online Security", ['No internet service', 'No', 'Yes'])
    online_backup = st.selectbox("Online Backup", ['No internet service', 'No', 'Yes'])
    device_protection = st.selectbox("Device Protection", ['No internet service', 'No', 'Yes'])
    tech_support = st.selectbox("Tech Support", ['No internet service', 'No', 'Yes'])
    streaming_tv = st.selectbox("Streaming TV", ['No internet service', 'No', 'Yes'])
    streaming_movies = st.selectbox("Streaming Movies", ['No internet service', 'No', 'Yes'])

#Proses Prediksi
if st.button("Prediksi Churn"):
    # Nama kolom harus sama dengan dataset awal (X_train)
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Prediksi
    try:
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("Hasil Prediksi:")
        
        if prediction[0] == 1:
            st.error(f"⚠️ Pelanggan diprediksi akan **CHURN** (Berhenti).")
        else:
            st.success(f"✅ Pelanggan diprediksi **TIDAK CHURN** (Setia).")
            
        st.write(f"Probabilitas Churn: {prediction_proba[0][1]*100:.2f}%")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
        st.write("Pastikan format input sesuai dengan data training.")

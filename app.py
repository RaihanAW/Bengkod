import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# 1. Load Model
model = joblib.load('model_only.pkl')

st.title("Telco Customer Churn Prediction")

def rerun_app():
    st.cache_resource.clear()

# 2. Form Input
col1, col2 = st.columns(2)

with st.expander("ℹ️ Penjelasan Singkat Fitur"):
    st.write("""
    Penjelasan Fitur:
    
    **Data Demografis & Profil:**
    * **Gender**: Jenis kelamin pelanggan (Male/Female).
    * **Senior Citizen**: Apakah pelanggan berusia lanjut (1: Ya, 0: Tidak).
    * **Partner**: Apakah pelanggan memiliki pasangan (Yes/No).
    * **Dependents**: Apakah pelanggan memiliki tanggungan seperti anak/orang tua (Yes/No).
    * **Tenure**: Lama berlangganan dalam hitungan bulan.

    **Layanan Utama & Tambahan:**
    * **Phone Service**: Menggunakan layanan telepon (Yes/No).
    * **Multiple Lines**: Memiliki lebih dari satu jalur telepon.
    * **Internet Service**: Jenis provider internet (DSL, Fiber optic, No).
    * **Online Security**: Layanan keamanan cybersecurity tambahan.
    * **Online Backup** : Layanan cadangan data
    * **Device Protection**: Perlindungan perangkat.
    * **Tech Support**: Layanan dukungan teknis khusus.
    * **Streaming TV / Movies**: Layanan hiburan streaming TV dan Film.

    **Akun & Pembayaran:**
    * **Contract**: Jenis kontrak (Month-to-month, One year, Two year).
    * **Paperless Billing**: Tagihan dikirim secara digital/paperless (Yes/No).
    * **Payment Method**: Metode pembayaran yang digunakan.
    * **Monthly Charges**: Biaya yang dibayar setiap bulan.
    * **Total Charges**: Total biaya yang telah dibayar selama berlangganan.
    """)

with col1:
    st.header("Profil & Tagihan")
    gender = st.selectbox("Gender", ['Female', 'Male'])
    partner = st.selectbox("Partner", ['No', 'Yes'])
    dependents = st.selectbox("Dependents", ['No', 'Yes'])
    tenure = st.number_input("Tenure (Bulan)", min_value=0.0, value=12.0)
    # Ordinal untuk Contract
    contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", ['No', 'Yes'])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)

with col2:
    st.header("Layanan")
    phone_service = st.selectbox("Phone Service", ['No', 'Yes'])
    multiple_lines = st.selectbox("Multiple Lines", ['No', 'No phone service', 'Yes'])
    internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", ['No', 'No internet service', 'Yes'])
    online_backup = st.selectbox("Online Backup", ['No', 'No internet service', 'Yes'])
    device_protection = st.selectbox("Device Protection", ['No', 'No internet service', 'Yes'])
    tech_support = st.selectbox("Tech Support", ['No', 'No internet service', 'Yes'])
    streaming_tv = st.selectbox("Streaming TV", ['No', 'No internet service', 'Yes'])
    streaming_movies = st.selectbox("Streaming Movies", ['No', 'No internet service', 'Yes'])
    payment_method = st.selectbox("Payment Method", [
        'Bank transfer (automatic)', 'Credit card (automatic)', 
        'Electronic check', 'Mailed check'
    ])

st.button("Reset Input", on_click=rerun_app)

# 3. Proses Prediksi
if st.button("Prediksi"):
    contract_map = {'Month-to-month': 0, 'One year': 1, 'Two year': 2}
    contract_encoded = contract_map[contract]

    # B. Menyiapkan Dictionary untuk One-Hot Encoding
    input_dict = {
        'tenure': tenure,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract': contract_encoded,
        
        # One-Hot Columns (Set default 0)
        'gender_Female': 0, 'gender_Male': 0,
        'Partner_No': 0, 'Partner_Yes': 0,
        'Dependents_No': 0, 'Dependents_Yes': 0,
        'PhoneService_No': 0, 'PhoneService_Yes': 0,
        'MultipleLines_No': 0, 'MultipleLines_No phone service': 0, 'MultipleLines_Yes': 0,
        'InternetService_DSL': 0, 'InternetService_Fiber optic': 0, 'InternetService_No': 0,
        'OnlineSecurity_No': 0, 'OnlineSecurity_No internet service': 0, 'OnlineSecurity_Yes': 0,
        'OnlineBackup_No': 0, 'OnlineBackup_No internet service': 0, 'OnlineBackup_Yes': 0,
        'DeviceProtection_No': 0, 'DeviceProtection_No internet service': 0, 'DeviceProtection_Yes': 0,
        'TechSupport_No': 0, 'TechSupport_No internet service': 0, 'TechSupport_Yes': 0,
        'StreamingTV_No': 0, 'StreamingTV_No internet service': 0, 'StreamingTV_Yes': 0,
        'StreamingMovies_No': 0, 'StreamingMovies_No internet service': 0, 'StreamingMovies_Yes': 0,
        'PaperlessBilling_No': 0, 'PaperlessBilling_Yes': 0,
        'PaymentMethod_Bank transfer (automatic)': 0, 
        'PaymentMethod_Credit card (automatic)': 0,
        'PaymentMethod_Electronic check': 0, 
        'PaymentMethod_Mailed check': 0
    }

    # C. Mengubah nilai 0 menjadi 1 sesuai pilihan user
    input_dict[f'gender_{gender}'] = 1
    input_dict[f'Partner_{partner}'] = 1
    input_dict[f'Dependents_{dependents}'] = 1
    input_dict[f'PhoneService_{phone_service}'] = 1
    input_dict[f'MultipleLines_{multiple_lines}'] = 1
    input_dict[f'InternetService_{internet_service}'] = 1
    input_dict[f'OnlineSecurity_{online_security}'] = 1
    input_dict[f'OnlineBackup_{online_backup}'] = 1
    input_dict[f'DeviceProtection_{device_protection}'] = 1
    input_dict[f'TechSupport_{tech_support}'] = 1
    input_dict[f'StreamingTV_{streaming_tv}'] = 1
    input_dict[f'StreamingMovies_{streaming_movies}'] = 1
    input_dict[f'PaperlessBilling_{paperless_billing}'] = 1
    input_dict[f'PaymentMethod_{payment_method}'] = 1

    # D. Membuat DataFrame dan Memastikan URUTAN KOLOM SAMA PERSIS
    df_input = pd.DataFrame([input_dict])
    
    # Daftar kolom sesuai urutan fit model Anda
    feature_order = [
        'tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'gender_Female', 'gender_Male', 
        'Partner_No', 'Partner_Yes', 'Dependents_No', 'Dependents_Yes', 'PhoneService_No', 
        'PhoneService_Yes', 'MultipleLines_No', 'MultipleLines_No phone service', 'MultipleLines_Yes', 
        'InternetService_DSL', 'InternetService_Fiber optic', 'InternetService_No', 'OnlineSecurity_No', 
        'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 'OnlineBackup_No', 
        'OnlineBackup_No internet service', 'OnlineBackup_Yes', 'DeviceProtection_No', 
        'DeviceProtection_No internet service', 'DeviceProtection_Yes', 'TechSupport_No', 
        'TechSupport_No internet service', 'TechSupport_Yes', 'StreamingTV_No', 
        'StreamingTV_No internet service', 'StreamingTV_Yes', 'StreamingMovies_No', 
        'StreamingMovies_No internet service', 'StreamingMovies_Yes', 'PaperlessBilling_No', 
        'PaperlessBilling_Yes', 'PaymentMethod_Bank transfer (automatic)', 
        'PaymentMethod_Credit card (automatic)', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
    ]
    
    df_input = df_input[feature_order]

    # E. Prediksi dan Tampilan Hasil
    prediction = model.predict(df_input)
    proba = model.predict_proba(df_input)[0][1]

    st.divider()
    if prediction[0] == 1:
        st.error(f"Hasil Prediksi: **CHURN (1)**")
        st.write(f"Probabilitas pelanggan berhenti: {proba:.2%}")
    else:
        st.success(f"Hasil Prediksi: **TIDAK CHURN (0)**")
        st.write(f"Probabilitas pelanggan tidak berhenti: {proba:.2%}")
    


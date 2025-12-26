import streamlit as st
import pandas as pd
import joblib
import numpy as np

st.set_page_config(page_title="UAS Bengkod", layout="wide")

@st.cache_resource
def load_assets():
    try:
        scaler = joblib.load('scaler.pkl')
        ord_enc = joblib.load('ordinal_encoder.pkl')
        oh_enc = joblib.load('onehot_encoder.pkl')
        model = joblib.load('model_only.pkl') 
        return scaler, ord_enc, oh_enc, model
    except Exception as e:
        st.error(f"Gagal memuat file .pkl. Pastikan file ada di folder yang sama. Error: {e}")
        return None, None, None, None


scaler, ord_enc, oh_enc, model = load_assets()


st.title("UAS Bengkod")
st.markdown("Aplikasi untuk memprediksi potensi churn pelanggan.")


if model is not None:
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Profil Data")
            gender = st.selectbox("Gender", ['Male', 'Female'])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
            partner = st.selectbox("Partner", ['Yes', 'No'])
            dependents = st.selectbox("Dependents", ['Yes', 'No'])
            tenure = st.number_input("Tenure (Bulan)", min_value=0, max_value=100, value=12)

        with col2:
            st.subheader("Layanan")
            phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
            multiple_lines = st.selectbox("Multiple Lines", ['No phone service', 'No', 'Yes'])
            internet_service = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
            online_security = st.selectbox("Online Security", ['No internet service', 'No', 'Yes'])
            online_backup = st.selectbox("Online Backup", ['No internet service', 'No', 'Yes'])
            device_protection = st.selectbox("Device Protection", ['No internet service', 'No', 'Yes'])
        
        with col3:
            st.subheader("Akun & Tagihan")
            tech_support = st.selectbox("Tech Support", ['No internet service', 'No', 'Yes'])
            streaming_tv = st.selectbox("Streaming TV", ['No internet service', 'No', 'Yes'])
            streaming_movies = st.selectbox("Streaming Movies", ['No internet service', 'No', 'Yes'])
            contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
            paperless_billing = st.selectbox("Paperless Billing", ['Yes', 'No'])
            payment_method = st.selectbox("Payment Method", ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=500.0)


        submitted = st.form_submit_button("Prediksi Sekarang")

    if submitted:
        raw_data = pd.DataFrame({
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

        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        ord_cols = ['Contract']
        nom_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 
                    'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen']

        try:
            # C. Transformasi
            
            # 1. Numerik (MinMax)
            num_transformed = scaler.transform(raw_data[num_cols])
            df_num = pd.DataFrame(num_transformed, columns=num_cols)

            # 2. Ordinal (Contract)
            ord_transformed = ord_enc.transform(raw_data[ord_cols])
            df_ord = pd.DataFrame(ord_transformed, columns=ord_cols)

            # 3. Nominal (OneHot)
            nom_transformed = oh_enc.transform(raw_data[nom_cols])
            # Ambil nama kolom baru dari encoder
            nom_feature_names = oh_enc.get_feature_names_out(nom_cols)
            df_nom = pd.DataFrame(nom_transformed, columns=nom_feature_names)

            # 4. Passthrough (SeniorCitizen)
            df_pass = raw_data[pass_cols].reset_index(drop=True)

            # D. Gabungkan Semua (Concatenate)
            # Urutan penggabungan ini SANGAT PENTING. 
            # Biasanya urutannya: [Numerik, Ordinal, Nominal, Passthrough] 
            # Sesuaikan urutan ini dengan urutan X_train.columns akhir di Notebook Anda.
            # Jika di notebook Anda menggunakan pd.concat([num, ord, nom], axis=1), maka ikuti urutan itu.
            
            X_final = pd.concat([df_num, df_ord, df_nom, df_pass], axis=1)

            # Debugging (Opsional: Tampilkan data yang akan masuk model)
            # st.write("Data yang masuk ke model:", X_final)

            # E. Prediksi
            prediction = model.predict(X_final)
            prob = model.predict_proba(X_final)

            # F. Tampilkan Hasil
            st.divider()
            st.subheader("Hasil Prediksi")
            
            col_res1, col_res2 = st.columns(2)
            
            if prediction[0] == 1:
                col_res1.error("🔴 CHURN (Berhenti Berlangganan)")
                col_res1.write("Pelanggan ini berisiko tinggi untuk pindah.")
            else:
                col_res1.success("🟢 NOT CHURN (Tetap Berlangganan)")
                col_res1.write("Pelanggan ini diprediksi setia.")

            col_res2.metric("Probabilitas Churn", f"{prob[0][1]*100:.2f}%")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat pemrosesan data: {e}")
            st.write("Detail Error:", e)

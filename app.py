import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# ---------------------------------------------------------
# STYLING & CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Neural Networks - Henri Ilham",
    page_icon="🧠",
    layout="wide",
)

# Custom CSS untuk mempercantik kartu identitas & UI
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0px;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 20px;
    }
    .profile-card {
        background-color: #F0F9FF;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #0284C7;
        margin-bottom: 20px;
    }
    </style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# SIDEBAR (Identitas & Navigasi)
# ---------------------------------------------------------
st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=80
)
st.sidebar.title("Dashboard Tugas NN")

# Identitas Diri
st.sidebar.markdown(
    """
<div class="profile-card">
    <small><b>Penyusun Tugas:</b></small><br>
    <b style="color: #0369A1; font-size: 1.05rem;">Henri Ilham</b><br>
    <small>Mata Kuliah: Neural Networks</small><br>
    <small>Status: 🟢 Active Deployment</small>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.header("📌 Navigasi Aplikasi")
pilihan_app = st.sidebar.radio(
    "Pilih Modul Pembelajaran:",
    [
        "Aplikasi 1: Regresi (Prediksi Tarif)",
        "Aplikasi 2: Klasifikasi Biner",
        "Aplikasi 3: Klasifikasi Multikelas & Callbacks",
    ],
)

st.sidebar.divider()
st.sidebar.caption("© 2026 Henri Ilham • Streamlit Deployment")

# ---------------------------------------------------------
# HEADER UTAMA
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">🧠 Dashboard Neural Networks (Keras & TensorFlow)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Simulasi & Visualisasi Model Deep Learning | Disusun oleh <b>Henri Ilham</b></div>',
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------
# APLIKASI 1: REGRESI
# ---------------------------------------------------------
if pilihan_app == "Aplikasi 1: Regresi (Prediksi Tarif)":
    st.subheader("1. Aplikasi Regresi - Prediksi Nilai Kontinu")
    st.write(
        "Model regresi memprediksi variabel kontinu (seperti tarif taksi) tanpa menggunakan fungsi aktivasi pada layer output."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### ⚙️ Parameter Input Simulasi")
        f1 = st.slider("Jarak Tempuh (km)", 1.0, 50.0, 10.0)
        f2 = st.slider("Waktu Tempuh (menit)", 5.0, 120.0, 25.0)
        f3 = st.slider("Tingkat Kemacetan (1-5)", 1.0, 5.0, 2.0)

        run_btn = st.button("🚀 Train Model & Prediksi", type="primary")

    with col2:
        st.markdown("### 💻 Arsitektur Model")
        kode_regresi = """
model = Sequential([
    Dense(512, activation='relu', input_dim=3),
    Dense(512, activation='relu'),
    Dense(1) # Output tanpa aktivasi (Regresi)
])
model.compile(optimizer='adam', loss='mae')
        """
        st.code(kode_regresi, language="python")

    if run_btn:
        with st.spinner("Melatih Neural Network di latar belakang..."):
            X_train = np.random.rand(1000, 3) * [50, 120, 5]
            y_train = (
                X_train[:, 0] * 3.5 + X_train[:, 1] * 1.2 + X_train[:, 2] * 4.0
            )

            model = Sequential(
                [
                    Dense(512, activation="relu", input_dim=3),
                    Dense(512, activation="relu"),
                    Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mae", metrics=["mae"])
            history = model.fit(
                X_train, y_train, epochs=20, batch_size=32, verbose=0
            )

            sample_input = np.array([[f1, f2, f3]])
            pred_val = model.predict(sample_input)[0][0]
            y_pred = model.predict(X_train)
            r2 = r2_score(y_train, y_pred)

        st.success("✅ Pelatihan Model Selesai!")
        res_col1, res_col2 = st.columns(2)
        res_col1.metric("Hasil Prediksi Estimasi Tarif", f"Rp {pred_val*1000:,.0f}")
        res_col2.metric("Akurasi Model (R² Score)", f"{r2:.4f}")

        # Visualisasi Training Loss
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(history.history["loss"], color="#0284C7", linewidth=2)
        ax.set_title("Grafik Konvergensi Loss (MAE) per Epoch")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        st.pyplot(fig)

# ---------------------------------------------------------
# APLIKASI 2: KLASIFIKASI BINER
# ---------------------------------------------------------
elif pilihan_app == "Aplikasi 2: Klasifikasi Biner":
    st.subheader("2. Aplikasi Klasifikasi Biner")
    st.write(
        "Memisahkan data ke dalam 2 kategori pilihan menggunakan fungsi aktivasi Sigmoid pada layer output."
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### ⚙️ Parameter Input")
        val1 = st.number_input("Fitur 1 (Range -1.0 s/d 1.0)", -1.0, 1.0, -0.3)
        val2 = st.number_input("Fitur 2 (Range -1.0 s/d 1.0)", -1.0, 1.0, 0.4)
        run_btn2 = st.button("🚀 Jalankan Klasifikasi Biner", type="primary")

    with col2:
        st.markdown("### 💻 Arsitektur Model")
        kode_biner = """
model = Sequential([
    Dense(128, activation='relu', input_dim=2),
    Dense(1, activation='sigmoid') # Sigmoid untuk 0/1
])
model.compile(optimizer='adam', loss='binary_crossentropy')
        """
        st.code(kode_biner, language="python")

    if run_btn2:
        with st.spinner("Proses klasifikasi berlangsung..."):
            X_train = np.random.uniform(-1, 1, (1000, 2))
            y_train = ((X_train[:, 0] ** 2 + X_train[:, 1] ** 2) < 0.5).astype(
                int
            )

            model = Sequential(
                [
                    Dense(128, activation="relu", input_dim=2),
                    Dense(1, activation="sigmoid"),
                ]
            )
            model.compile(
                optimizer="adam",
                loss="binary_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(X_train, y_train, epochs=15, batch_size=16, verbose=0)

            sample = np.array([[val1, val2]])
            prob = model.predict(sample)[0][0]
            pred_class = 1 if prob > 0.5 else 0

        st.success("✅ Klasifikasi Selesai!")
        st.info(
            f"Probabilitas Positif: **{prob*100:.2f}%** ➔ Hasil Kategori: **Kelas {pred_class}**"
        )

# ---------------------------------------------------------
# APLIKASI 3: KLASIFIKASI MULTIKELAS
# ---------------------------------------------------------
elif pilihan_app == "Aplikasi 3: Klasifikasi Multikelas & Callbacks":
    st.subheader("3. Aplikasi Klasifikasi Multikelas & Callbacks")
    st.write(
        "Memilah data ke dalam 4 kategori ($N > 2$) menggunakan aktivasi Softmax dan teknik Regularisasi Dropout."
    )

    run_btn3 = st.button("🚀 Jalankan Simulasi Multikelas", type="primary")

    if run_btn3:
        with st.spinner("Melatih Model Multikelas..."):
            X_train = np.random.rand(1000, 2)
            y_train = np.random.randint(0, 4, 1000)

            model = Sequential(
                [
                    Dense(128, activation="relu", input_dim=2),
                    Dropout(0.2),
                    Dense(4, activation="softmax"),
                ]
            )
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

            sample = np.array([[0.2, 0.8]])
            probs = model.predict(sample)[0]
            pred_class = np.argmax(probs)

        st.success("✅ Simulasi Selesai!")
        st.write(f"Kategori Prediksi Tertinggi: **Kelas {pred_class}**")

        # Visualisasi Distribusi Probabilitas
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(
            ["Kelas 0", "Kelas 1", "Kelas 2", "Kelas 3"],
            probs,
            color="#0284C7",
        )
        ax.set_ylabel("Probabilitas")
        ax.set_title("Distribusi Output Softmax per Kelas")
        st.pyplot(fig)

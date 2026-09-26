import numpy as np
import streamlit as st
from sklearn.metrics import r2_score
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential

# ---------------------------------------------------------
# STYLING & JUDUL APLIKASI
# ---------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Neural Networks Chapter 9", layout="wide"
)
st.title("🤖 Dashboard Aplikasi Neural Networks (Keras & TensorFlow)")
st.write(
    "Pilih aplikasi pada sidebar di sebelah kiri untuk melihat kode dan menjalankan simulasi."
)

# ---------------------------------------------------------
# SIDEBAR MENU (Pilihan 3 Aplikasi Utama)
# ---------------------------------------------------------
st.sidebar.header("Navigasi Aplikasi")
pilihan_app = st.sidebar.radio(
    "Pilih Jenis Aplikasi Neural Network:",
    [
        "Aplikasi 1: Regresi (Prediksi Tarif)",
        "Aplikasi 2: Klasifikasi Biner",
        "Aplikasi 3: Klasifikasi Multikelas & Callbacks",
    ],
)

# ---------------------------------------------------------
# APLIKASI 1: REGRESI
# ---------------------------------------------------------
if pilihan_app == "Aplikasi 1: Regresi (Prediksi Tarif)":
    st.header("1. Aplikasi Regresi - Prediksi Nilai Kontinu")
    st.write(
        "Aplikasi ini memprediksi nilai kontinu (seperti tarif taksi) menggunakan output layer tanpa fungsi aktivasi."
    )

    kode_regresi = """
# Arsitektur Model Regresi
model = Sequential()
model.add(Dense(512, activation='relu', input_dim=3))
model.add(Dense(512, activation='relu'))
model.add(Dense(1)) # Output 1 neuron tanpa aktivasi

model.compile(optimizer='adam', loss='mae', metrics=['mae'])
model.fit(X_train, y_train, epochs=20, batch_size=32)
    """
    st.code(kode_regresi, language="python")

    if st.button("🚀 Jalankan Proses Training & Prediksi"):
        with st.spinner("Melatih model Neural Network..."):
            X_train = np.random.rand(1000, 3) * 10
            y_train = (
                X_train[:, 0] * 2 + X_train[:, 1] * 1.5 + X_train[:, 2] * 5.0
            )

            model = Sequential(
                [
                    Dense(512, activation="relu", input_dim=3),
                    Dense(512, activation="relu"),
                    Dense(1),
                ]
            )
            model.compile(optimizer="adam", loss="mae", metrics=["mae"])
            model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)

            y_pred = model.predict(X_train)
            r2 = r2_score(y_train, y_pred)

            sample_input = np.array([[4.0, 17.0, 2.0]])
            pred_val = model.predict(sample_input)[0][0]

        st.success("Proses Berhasil Selesai!")
        st.metric(
            label="Hasil Evaluasi (R² Score)", value=f"{r2:.4f}"
        )
        st.info(f"Hasil Prediksi Sampel Input [4.0, 17.0, 2.0]: **{pred_val:.2f}**")

# ---------------------------------------------------------
# APLIKASI 2: KLASIFIKASI BINER
# ---------------------------------------------------------
elif pilihan_app == "Aplikasi 2: Klasifikasi Biner":
    st.header("2. Aplikasi Klasifikasi Biner")
    st.write(
        "Mengelompokkan data ke dalam 2 kelas (0 atau 1) menggunakan fungsi aktivasi Sigmoid."
    )

    kode_biner = """
# Model Biner dengan Aktivasi Sigmoid
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    """
    st.code(kode_biner, language="python")

    if st.button("🚀 Jalankan Proses Training & Prediksi"):
        with st.spinner("Melatih model Klasifikasi Biner..."):
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

            sample = np.array([[-0.5, 0.0]])
            prob = model.predict(sample)[0][0]
            pred_class = 1 if prob > 0.5 else 0

        st.success("Proses Berhasil Selesai!")
        st.write(f"Probabilitas Positif: **{prob*100:.2f}%**")
        st.info(f"Hasil Prediksi Kategori: **Kelas {pred_class}**")

# ---------------------------------------------------------
# APLIKASI 3: KLASIFIKASI MULTIKELAS
# ---------------------------------------------------------
elif pilihan_app == "Aplikasi 3: Klasifikasi Multikelas & Callbacks":
    st.header("3. Aplikasi Klasifikasi Multikelas")
    st.write(
        "Memilah data ke >2 kategori menggunakan Softmax dan teknik pencegahan overfitting (Dropout)."
    )

    kode_multi = """
# Model Multikelas dengan Softmax & Dropout
model = Sequential()
model.add(Dense(128, activation='relu', input_dim=2))
model.add(Dropout(0.2))
model.add(Dense(4, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    """
    st.code(kode_multi, language="python")

    if st.button("🚀 Jalankan Proses Training & Prediksi"):
        with st.spinner("Melatih model Multikelas..."):
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

        st.success("Proses Berhasil Selesai!")
        st.write("Distribusi Probabilitas per Kelas:", probs)
        st.info(f"Hasil Prediksi Kategori Terbesar: **Kelas {pred_class}**")
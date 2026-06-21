import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Penyakit Daun Tomat",
    page_icon="🍅",
    layout="centered"
)

# --- DOWNLOAD MODEL DARI GOOGLE DRIVE ---
# Menggunakan ekstensi .tflite dan File ID barumu
MODEL_FILE = "model_klasifikasi_tomat.tflite"
FILE_ID = "1n7AEsxODnX9_63Rwt5m5kCmZ-M3STAK9"

@st.cache_resource
def load_model():
    # Download model jika belum ada
    if not os.path.exists(MODEL_FILE):
        with st.spinner("Mengunduh model AI (versi ringan), mohon tunggu beberapa saat..."):
            url = f"https://drive.google.com/uc?id={FILE_ID}"
            gdown.download(url, MODEL_FILE, quiet=False)

    # Load model menggunakan TFLite Interpreter (Sangat hemat RAM)
    interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# --- DAFTAR KELAS PENYAKIT ---
class_names = [
    'Tomato Bacterial spot',
    'Tomato Early blight',
    'Tomato Late_blight',
    'Tomato Leaf_mold',
    'Tomato Septoria_leaf_spot',
    'Tomato Spider mites Two spotted spider mite',
    'Tomato Target Spot',
    'Tomato Tomato Yellow Leaf Curl Virus',
    'Tomato Tomato mosaic virus',
    'Tomato healthy'
]

# --- ANTARMUKA WEBSITE ---
st.title("🍅 Sistem Deteksi Penyakit Daun Tomat")

st.write("""
Selamat datang! Unggah foto daun tomat Anda ke sini, dan sistem AI kami
akan membantu mendeteksi apakah daun tersebut sehat atau terindikasi penyakit.
""")

st.markdown("---")

# --- FITUR UPLOAD GAMBAR ---
uploaded_file = st.file_uploader(
    "Pilih file gambar daun tomat (JPG/PNG)...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Gambar yang Diunggah:**")
        st.image(image, use_container_width=True)

    with col2:
        st.write("**Hasil Analisis:**")

        with st.spinner("Sedang menganalisis gambar..."):

            # Resize sesuai ukuran training
            img = image.resize((256, 256))

            # Konversi ke array
            img_array = tf.keras.preprocessing.image.img_to_array(img)

            # Tambah dimensi batch dan pastikan tipe data Float32 (Wajib untuk TFLite)
            img_array = np.expand_dims(img_array, axis=0).astype(np.float32)

            # Normalisasi
            img_array = img_array / 255.0

            # --- PREDIKSI MENGGUNAKAN TFLITE INTERPRETER ---
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Masukkan gambar ke dalam mesin
            interpreter.set_tensor(input_details[0]['index'], img_array)
            
            # Jalankan prediksi
            interpreter.invoke()
            
            # Ambil hasil prediksi
            predictions = interpreter.get_tensor(output_details[0]['index'])

            predicted_class_index = np.argmax(predictions)
            predicted_class = class_names[predicted_class_index]
            confidence = np.max(predictions) * 100

            st.success(f"**Prediksi:** {predicted_class}")
            st.info(f"**Tingkat Keyakinan:** {confidence:.2f}%")

            if predicted_class == "Tomato healthy":
                st.balloons()
                st.write("🌿 Yeay! Tanaman tomat Anda terlihat sangat sehat.")
            else:
                st.warning(
                    "⚠️ Perhatian: Daun ini terindikasi memiliki penyakit. "
                    "Segera lakukan penanganan yang tepat!"
                )
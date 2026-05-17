import streamlit as st
import numpy as np
import joblib
from PIL import Image
from streamlit_drawable_canvas import st_canvas
import io

st.set_page_config(
    page_title="MNIST Rakam Tanıma",
    page_icon="✏️",
    layout="centered"
)

st.title("✏️ MNIST El Yazısı Rakam Tanıma")
st.write("Gerçek MNIST veri seti ile eğitilmiş Random Forest modeli kullanılır.")

st.info("""
📌 Model Bilgileri

• Algoritma: Random Forest Classifier  
• Veri Seti: Gerçek MNIST Dataset  
• Eğitim Verisi: 60.000 görüntü  
• Test Verisi: 10.000 görüntü  
• Görüntü Boyutu: 28x28 piksel  
• Test Doğruluğu: %96.89
""")


@st.cache_resource
def load_model():
    return joblib.load("mnist_rf_model.pkl")

# Tahmin geçmişi için session state
if "prediction_history" not in st.session_state:
    st.session_state.prediction_history = []

def preprocess_canvas_image(image):
    """
    Canvas'tan gelen çizimi MNIST formatına uygun hale getirir.
    """

    # Görüntüyü gri tona çevir
    image = image.convert("L")

    # Görüntüyü numpy array formatına çevir
    img_array = np.array(image)

    # Beyaz pikselleri bul
    coords = np.argwhere(img_array > 20)

    # Eğer çizim algılanmadıysa boş değer döndür
    if coords.size == 0:
        return None, None

    # Çizilen rakamın sınırlarını bul
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Sadece rakamın olduğu alanı kırp
    cropped = img_array[y_min:y_max + 1, x_min:x_max + 1]

    # Rakamı kare bir alana yerleştir
    h, w = cropped.shape
    size = max(h, w)

    square = np.zeros((size, size), dtype=np.uint8)

    y_offset = (size - h) // 2
    x_offset = (size - w) // 2

    square[y_offset:y_offset + h, x_offset:x_offset + w] = cropped

    # Kenarlara boşluk ekle
    square = np.pad(square, pad_width=20, mode="constant", constant_values=0)

    # 28x28 MNIST boyutuna çevir
    processed_image = Image.fromarray(square)
    processed_image = processed_image.resize((28, 28), Image.Resampling.LANCZOS)

    # Piksel değerlerini 0-1 aralığına getir
    final_array = np.array(processed_image).astype("float32") / 255.0

    # Model için 784 özellikli hale getir
    img_flat = final_array.reshape(1, -1)

    return img_flat, processed_image


try:
    model = load_model()
except:
    st.error("Model bulunamadı. Önce `python model_train.py` komutunu çalıştır.")
    st.stop()


st.subheader("Rakam Çiz")

canvas_result = st_canvas(
    fill_color="black",
    stroke_width=12,
    stroke_color="white",
    background_color="black",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

st.caption("🗑️ Çizimi temizlemek için canvas altındaki çöp kutusunu kullanabilirsiniz.")

if st.button("Tahmin Et"):
    if canvas_result.image_data is not None:

        # Canvas görüntüsünü PIL formatına çevir
        image = Image.fromarray(canvas_result.image_data.astype("uint8"))

        # Görüntüyü MNIST formatına uygun hale getir
        img_flat, processed_image = preprocess_canvas_image(image)

        # Kullanıcı çizim yapmadıysa uyarı göster ve işlemi durdur
        if img_flat is None:
            st.warning("Lütfen önce bir rakam çizin.")
            st.stop()

        # Tahmin yap
        prediction = model.predict(img_flat)[0]
        probabilities = model.predict_proba(img_flat)[0]

        confidence = probabilities[prediction] * 100

        # Tahmin geçmişine ekle
        st.session_state.prediction_history.insert(
            0,
            f"Rakam {prediction} → %{confidence:.2f}"
        )

        # Maksimum 5 kayıt tut
        st.session_state.prediction_history = st.session_state.prediction_history[:5]

        st.markdown("---")
        st.subheader("Tahmin Sonucu")
        st.markdown(f"# {prediction}")
        st.write(f"Güven oranı: %{confidence:.2f}")

        # Güven oranı düşükse uyarı ver
        if confidence < 50:
            st.warning(
                "Model bu tahminden çok emin değil. "
                "Rakamı daha ortalı ve net çizmeyi deneyin."
            )

        st.subheader("Modelin Gördüğü 28x28 Görüntü")
        st.image(
            processed_image.resize((280, 280), Image.Resampling.NEAREST),
            width=280
        )

        # Görüntüyü indirilebilir hale getir
        buffer = io.BytesIO()
        processed_image.save(buffer, format="PNG")

        st.download_button(
            label="📥 28x28 Görüntüyü İndir",
            data=buffer.getvalue(),
            file_name="mnist_digit.png",
            mime="image/png"
        )

        # En yüksek 3 tahmini bul
        top_3 = np.argsort(probabilities)[::-1][:3]

        st.subheader("🔍 En Olası Tahminler")

        for idx, digit in enumerate(top_3):
            prob = probabilities[digit] * 100

            if idx == 0:
                medal = "🥇"
            elif idx == 1:
                medal = "🥈"
            else:
                medal = "🥉"

            st.write(f"{medal} Rakam {digit} → %{prob:.2f}")

        st.subheader("Tüm Rakam Olasılıkları")
        prob_dict = {str(i): float(probabilities[i] * 100) for i in range(10)}
        st.bar_chart(prob_dict)

        st.subheader("🕘 Tahmin Geçmişi")

        for item in st.session_state.prediction_history:
            st.write(item)
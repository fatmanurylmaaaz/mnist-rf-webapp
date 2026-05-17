import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt

print("=" * 50)
print("MNIST RANDOM FOREST MODEL EĞİTİMİ")
print("=" * 50)

# 1. Gerçek MNIST veri setini yükle
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Veri seti yüklendi.")
print("Eğitim verisi:", X_train.shape)
print("Test verisi:", X_test.shape)

# 2. 28x28 görüntüleri 784 özellikli vektöre çevir
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# 3. Normalize et: 0-255 arası pikselleri 0-1 arasına getir
X_train = X_train / 255.0
X_test = X_test / 255.0

# 4. Random Forest modeli oluştur
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)

print("Model eğitiliyor...")
model.fit(X_train, y_train)

# 5. Test et
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Test doğruluğu: %{accuracy * 100:.2f}")

# 6. Confusion matrix hesapla
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# 7. Confusion Matrix görseli oluştur ve kaydet
plt.figure(figsize=(10, 8))
plt.imshow(cm, cmap="Blues")

plt.title("Confusion Matrix")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")

plt.colorbar()

for i in range(10):
    for j in range(10):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            color="black"
        )

plt.savefig("model_analysis.png")
plt.close()

print("Confusion Matrix görseli kaydedildi: model_analysis.png")

# 8. Modeli kaydet
joblib.dump(model, "mnist_rf_model.pkl")

print("Model kaydedildi: mnist_rf_model.pkl")
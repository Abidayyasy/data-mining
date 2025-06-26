import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix

st.title("Naive Bayes Classifier - Breast Cancer Classification")

# Load Dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model
st.subheader("Inisialisasi Model & Training")
nb = GaussianNB()
nb.fit(X_train_scaled, y_train)
st.success("Model berhasil dilatih dengan data training.")

# Prediksi Probabilitas
st.subheader("Prediksi Probabilitas Data Testing")
y_proba_nb = nb.predict_proba(X_test_scaled)
st.write("Probabilitas prediksi (kolom pertama = Benign, kolom kedua = Malignant):")
st.write(y_proba_nb)

# Visualisasi Probabilitas
st.subheader("Visualisasi Probabilitas Prediksi")
fig, ax = plt.subplots(figsize=(15, 5))
n_data = len(y_proba_nb)
x = np.arange(n_data)
width = 0.35

ax.bar(x - width/2, y_proba_nb[:, 0], width, label='Benign (0)', color='blue')
ax.bar(x + width/2, y_proba_nb[:, 1], width, label='Malignant (1)', color='red')
ax.set_title("Probabilitas Prediksi Naive Bayes")
ax.set_xlabel("Index Data")
ax.set_ylabel("Probabilitas")
ax.set_ylim(0, 1.05)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot(fig)

# Prediksi Akhir
st.subheader("Prediksi Akhir & Evaluasi")
y_pred_nb = nb.predict(X_test_scaled)
st.write("Hasil prediksi akhir:", y_pred_nb)
st.write("Data aktual:", y_test.values)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_nb)
st.write("Confusion Matrix:")
st.write(cm)

# Perhitungan Manual
TN, FP, FN, TP = cm.ravel()
total_data = len(y_test)
pred_benar = TP + TN
pred_salah = FP + FN

st.write(f"Total data uji: {total_data}")
st.write(f"Prediksi benar : {pred_benar}")
st.write(f"Prediksi salah : {pred_salah}")

st.write("### Perhitungan Akurasi")
st.write(f"Akurasi = ({pred_benar} / {total_data}) x 100")
acc = (pred_benar / total_data) * 100
st.write(f"Akurasi : **{acc:.2f}%**")

st.write("### Perhitungan Precision")
st.write(f"Precision = ({TP} / ({TP} + {FP})) x 100")
prec = (TP / (TP + FP)) * 100
st.write(f"Precision : **{prec:.2f}%**")

st.write("### Perhitungan Recall")
st.write(f"Recall = ({TP} / ({TP} + {FN})) x 100")
rec = (TP / (TP + FN)) * 100
st.write(f"Recall : **{rec:.2f}%**")

st.write("### Perhitungan F1-Score")
f1 = 2 * (prec * rec) / (prec + rec)
st.write(f"F1-Score : **{f1:.2f}%**")

# Visualisasi Confusion Matrix
st.subheader("Visualisasi Confusion Matrix")
cm_sum = cm.sum()
cm_percent = cm / cm_sum * 100

annot = np.empty_like(cm).astype(str)
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        c = cm[i, j]
        p = cm_percent[i, j]
        annot[i, j] = f"{c}\n({p:.1f}%)"

fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=annot, fmt='', cmap="YlGnBu", cbar=False,
            xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"],
            linewidths=0.5, linecolor='black', ax=ax)
ax.set_title("Confusion Matrix - Naive Bayes")
ax.set_ylabel("Actual Class")
ax.set_xlabel("Predicted Class")
st.pyplot(fig)

# Rangkuman
st.subheader("Rangkuman Akhir Naive Bayes")
st.write(f"**Akurasi   : {acc:.2f}%**")
st.write(f"**Precision : {prec:.2f}%**")
st.write(f"**Recall    : {rec:.2f}%**")
st.write(f"**F1-Score  : {f1:.2f}%**")
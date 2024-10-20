# Langkah  pertama lakukan import library yang dibutuhkan baik itu untuk visualisasi ataupun menampilkan output 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns

# Kode dibawah Menambahkan source dataset didalam file 
file_path = 'C:\\user_behavior_dataset.csv'
data = pd.read_csv(file_path)

# Menampilkan informasi dataset
print(data.info())

# Menampilkan informasi guna untuk mencari penulisan kata yang salah didalam dataset ini
print(data.columns)  

#  Tahap ini kami melakukan terlebih dahulu Prosessing data untuk menghilangkan missing value
data['Gender'] = data['Gender'].fillna(data['Gender'].mode()[0])
data['Screen On Time (hours/day)'] = data['Screen On Time (hours/day)'].fillna(data['Screen On Time (hours/day)'].median())
data['Device Model'] = data['Device Model'].fillna(data['Device Model'].mode()[0])
data['Operating System'] = data['Operating System'].fillna(data['Operating System'].mode()[0])
data['App Usage Time (min/day)'] = data['App Usage Time (min/day)'].fillna(data['App Usage Time (min/day)'].median())
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Number of Apps Installed'] = data['Number of Apps Installed'].fillna(data['Number of Apps Installed'].median())

# Kode merupakan Label Kategori dari Inputan yang kami akan  masukan (Gender, Device Model, and Operating System)
label_encoder_gender = LabelEncoder()
label_encoder_device = LabelEncoder()
label_encoder_os = LabelEncoder()
data['Gender'] = label_encoder_gender.fit_transform(data['Gender'])
data['Device Model'] = label_encoder_device.fit_transform(data['Device Model'])
data['Operating System'] = label_encoder_os.fit_transform(data['Operating System'])


""" Pemilihan fitur dan label Variabel X berisi fitur-fitur yang digunakan untuk melatih model. Fitur-fitur ini diambil dari dataset data seperti 'Gender', 'Screen On Time', 'Device Model', 'Operating System', dan 'App Usage Time'.
Variabel y berisi target atau label yang akan diprediksi oleh model, yaitu 'User Behavior Class'."""

X = data[['Gender', 'Screen On Time (hours/day)', 'Device Model', 'Operating System', 'App Usage Time (min/day)', 'Age', 'Number of Apps Installed']]
y = data['User Behavior Class']  

"""Standarisasi Data yang mana StandardScaler digunakan untuk menstandarkan fitur-fitur X sehingga memiliki nilai rata-rata 0 dan standar deviasi 1. 
Ini penting agar model bekerja dengan baik, terutama pada algoritma yang sensitif terhadap skala data. Pembagian data dan uji :
Data dibagi menjadi data latih dan data uji menggunakan fungsi train_test_split(), di mana 70% data digunakan untuk pelatihan (train) dan 30% untuk pengujian (test). 
Parameter random_state=42 memastikan pembagian data yang konsisten di setiap eksekusi.
"""
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)


""" Disini kami menggunakan metode pelatihan model random forest Model Random Forest Classifier dengan 100 pohon keputusan (decision trees)
dilatih menggunakan data latih X_train dan y_train. Parameter random_state=42 memastikan hasil yang konsisten.
 """  
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Melakukan Prediksi yaitu Setelah model dilatih, hasil prediksi untuk data uji (X_test) disimpan dalam variabel y_pred.
y_pred = rf.predict(X_test)

""" Melakukan sebuah Evaluasi akurasi Fungsi accuracy_score() digunakan untuk menghitung akurasi model, 
  yang merupakan perbandingan antara jumlah prediksi benar dengan total prediksi. 
  Akurasi ditampilkan dalam persentase dengan dua angka di belakang koma.
 """
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")


# Laporan klasifikasi dan confusion matriks 
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


"""visualisasi feature importances, untuk memvisualisasikan tingkat pentingnya (importance)
  setiap fitur yang digunakan dalam model Random Forest. Model akan memberikan bobot penting untuk setiap fitur, 
  yang membantu menjelaskan seberapa besar kontribusi masing-masing fitur dalam prediksi model """ 
  
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
features = ['Gender', 'Screen On Time (hours/day)', 'Device Model', 'Operating System', 'App Usage Time (min/day)', 'Age', 'Number of Apps Installed']
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), [features[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()

# Bagian kode ini menampilkan confusion matrix sebagai heatmap. Confusion matrix berguna untuk memahami bagaimana kinerja model dalam hal memprediksi kelas yang benar dan salah
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

"""Pada bagian ini, kode menghasilkan laporan mendetail untuk setiap kelas dalam dataset (1 sampai 5, yang menggambarkan tingkat kecanduan). 
  Laporan ini mencakup jumlah prediksi untuk setiap kelas, jumlah data aktual dalam kelas tersebut, serta akurasi prediksi untuk masing-masing kelas.""" 
class_labels = np.unique(y_test)
for label in class_labels:
    print(f"\nBehavior for class {label}:")
    print(f"Count of predictions: {(y_pred == label).sum()}")
    print(f"True instances: {(y_test == label).sum()}")
    print(f"Accuracy for this class: {accuracy_score(y_test[y_test == label], y_pred[y_test == label]) * 100:.2f}%")



import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split

# Load the dataset
def load_data():
    # Load the original dataset (replace with actual dataset)
    data = pd.read_excel("dataori.xlsx")  # Replace this with the path to your dataset
    return data

# Load dataset
data = load_data()

# Data Preprocessing
data_dropped = data.drop(columns=['NO', 'NAMA BALITA', 'TINGGI(M)', 'TINGGI M2'])
encode = LabelEncoder()
data_encoded = data_dropped.copy()
data_encoded['JENIS KELAMIN'] = encode.fit_transform(data_encoded['JENIS KELAMIN'])
data_encoded['STATUS GIZI'] = encode.fit_transform(data_encoded['STATUS GIZI'])

# Feature and label separation
X = data_encoded.drop('STATUS GIZI', axis=1)
y = data_encoded['STATUS GIZI']

# Normalize the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Sidebar Navigation with Custom Styling
st.markdown("""
    <style>
    .sidebar .sidebar-content {
        background-color: #2C3E50; /* Dark Blue Color */
        color: white;
        padding: 20px;
    }
    .sidebar .sidebar-content a {
        text-decoration: none;
        color: white;
        font-size: 16px;
        padding: 10px 15px;
        display: block;
        border-radius: 5px;
        margin-bottom: 5px;
    }
    .sidebar .sidebar-content a:hover {
        background-color: #34495E; /* Hover Effect */
        transform: scale(1.05);
        transition: all 0.3s ease;
    }
    </style>
""", unsafe_allow_html=True)

st.sidebar.title("Navigasi")
option = st.sidebar.radio("Pilih Menu", ['Data Asli dan Preprocessing', 'Model KNN', 'Model Naive Bayes', 'Model SVM'])

# Tampilkan Data Asli dan yang Sudah DiPreprocessing
if option == 'Data Asli dan Preprocessing':
    st.header("Klasifikasi Status Gizi Balita")
    
    # Memberikan pengertian tentang klasifikasi status gizi balita
    st.write("""
    Klasifikasi status gizi balita adalah proses penilaian terhadap status gizi seorang anak berdasarkan berat badan, 
    tinggi badan, dan parameter kesehatan lainnya. Dalam konteks ini, status gizi balita digunakan untuk mengidentifikasi 
    apakah seorang balita memiliki status gizi yang normal, kekurangan gizi, atau kelebihan gizi. Proses ini sangat penting 
    untuk mencegah masalah kesehatan yang lebih serius di kemudian hari dan untuk memastikan bahwa balita menerima 
    gizi yang cukup untuk tumbuh dan berkembang dengan baik.
    """)
    
    st.subheader("Data Asli")
    st.dataframe(data)

    # Tampilkan Data Asli setelah Seleksi Fitur
    st.subheader("Data Setelah Seleksi Fitur")
    st.dataframe(data_encoded)  # Tampilkan seluruh data secara lengkap

    # Data Setelah Normalisasi
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    st.subheader("Data Setelah Normalisasi")
    st.dataframe(X_scaled_df)  # Tampilkan data yang telah dinormalisasi secara lengkap

# Model KNN
elif option == 'Model KNN':
    st.header("Model KNN")
    
    # Split the data for KNN
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train KNN Model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    
    # Confusion Matrix
    cm_knn = confusion_matrix(y_test, y_pred_knn)
    cm_knn_df = pd.DataFrame(cm_knn, columns=encode.classes_, index=encode.classes_)
    st.subheader("Confusion Matrix - KNN")
    st.dataframe(cm_knn_df)  # Tampilkan confusion matrix dalam tabel

    # Data Actual vs Prediction
    result_df_knn = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_knn})
    st.subheader("Data Aktual dan Prediksi - KNN")
    st.dataframe(result_df_knn)  # Tampilkan data aktual vs prediksi

    # Metrics
    st.write(f"Akurasi KNN: {accuracy_score(y_test, y_pred_knn):.2f}")
    classification_rep_knn = classification_report(y_test, y_pred_knn, target_names=encode.classes_, output_dict=True)
    classification_rep_knn_df = pd.DataFrame(classification_rep_knn).transpose()  # Convert to DataFrame
    st.subheader("Classification Report - KNN")
    st.dataframe(classification_rep_knn_df)  # Tampilkan classification report dalam tabel

# Model Naive Bayes
elif option == 'Model Naive Bayes':
    st.header("Model Naive Bayes")
    
    # Split the data for Naive Bayes
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train Naive Bayes Model
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    y_pred_nb = nb_model.predict(X_test)
    
    # Confusion Matrix
    cm_nb = confusion_matrix(y_test, y_pred_nb)
    cm_nb_df = pd.DataFrame(cm_nb, columns=encode.classes_, index=encode.classes_)
    st.subheader("Confusion Matrix - Naive Bayes")
    st.dataframe(cm_nb_df)  # Tampilkan confusion matrix dalam tabel

    # Data Actual vs Prediction
    result_df_nb = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_nb})
    st.subheader("Data Aktual dan Prediksi - Naive Bayes")
    st.dataframe(result_df_nb)  # Tampilkan data aktual vs prediksi

    # Metrics
    st.write(f"Akurasi Naive Bayes: {accuracy_score(y_test, y_pred_nb):.2f}")
    classification_rep_nb = classification_report(y_test, y_pred_nb, target_names=encode.classes_, output_dict=True)
    classification_rep_nb_df = pd.DataFrame(classification_rep_nb).transpose()  # Convert to DataFrame
    st.subheader("Classification Report - Naive Bayes")
    st.dataframe(classification_rep_nb_df)  # Tampilkan classification report dalam tabel

# Model SVM
elif option == 'Model SVM':
    st.header("Model SVM")
    
    # Split the data for SVM
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Train SVM Model
    svm_model = SVC(kernel='linear', random_state=42)
    svm_model.fit(X_train, y_train)
    y_pred_svm = svm_model.predict(X_test)
    
    # Confusion Matrix
    cm_svm = confusion_matrix(y_test, y_pred_svm)
    cm_svm_df = pd.DataFrame(cm_svm, columns=encode.classes_, index=encode.classes_)
    st.subheader("Confusion Matrix - SVM")
    st.dataframe(cm_svm_df)  # Tampilkan confusion matrix dalam tabel

    # Data Actual vs Prediction
    result_df_svm = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred_svm})
    st.subheader("Data Aktual dan Prediksi - SVM")
    st.dataframe(result_df_svm)  # Tampilkan data aktual vs prediksi

    # Metrics
    st.write(f"Akurasi SVM: {accuracy_score(y_test, y_pred_svm):.2f}")
    classification_rep_svm = classification_report(y_test, y_pred_svm, target_names=encode.classes_, output_dict=True)
    classification_rep_svm_df = pd.DataFrame(classification_rep_svm).transpose()  # Convert to DataFrame
    st.subheader("Classification Report - SVM")
    st.dataframe(classification_rep_svm_df)  # Tampilkan classification report dalam tabel

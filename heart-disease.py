# Import Required Libraries
import pandas as pd
import numpy as np
import pickle
import time
import streamlit as st
from PIL import Image

# Page Configuration
st.set_page_config(layout="wide", page_title="Capstone Project DQLab", page_icon=":heart:")
st.sidebar.title("Navigation")
nav = st.sidebar.selectbox("Go to", ("Home", "Dataset", "Exploratory Data Analysis","Modelling", "Prediction", "About"))

# Dataset Page
url = "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
df = pd.read_csv(url)

# Function Heart Disease Prediction
def heart():
    st.write("""
        This app predicts the **Heart Disease**

        Data obtained from the [Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease) by UCIML. 
        """)
    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_manual():
            st.sidebar.header("Manual Input")
            age = st.sidebar.slider("Age", 0, 100, 25)
            cp = st.sidebar.selectbox("Chest Pain Type", ("Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"))
            if cp == "Typical Angina":
                cp = 1
            elif cp == "Atypical Angina":
                cp = 2
            elif cp == "Non-anginal Pain":
                cp = 3
            elif cp == "Asymptomatic":
                cp = 4
            thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 0, 200, 100)
            slope = st.sidebar.selectbox("Slope", ("Upsloping", "Flat", "Downsloping"))
            if slope == "Upsloping":
                slope = 1
            elif slope == "Flat":
                slope = 2
            elif slope == "Downsloping":
                slope = 3
            ca = st.sidebar.slider("Number of Major Vessels", 0, 4, 0)
            oldpeak = st.sidebar.slider("ST Depression Induced by Exercise Relative to Rest", 0.0, 10.0, 0.0)
            exang = st.sidebar.selectbox("Exercise Induced Angina", ("Yes", "No"))
            if exang == "Yes":
                exang = 1
            elif exang == "No":
                exang = 0
            thal = st.sidebar.selectbox("Thal", ("Normal", "Fixed Defect", "Reversable Defect"))
            if thal == "Normal":
                thal = 1
            elif thal == "Fixed Defect":
                thal = 2
            elif thal == "Reversable Defect":
                thal = 3
            sex = st.sidebar.selectbox("Sex", ("Male", "Female"))
            if sex == "Male":
                sex = 1
            else:
                sex =0
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca': ca,
                    'thal': thal,
                    'sex': sex,
                    'age': age}
            features = pd.DataFrame(data, index=[0])
            return features

        input_df = user_input_manual()

    # Data df
    st.image('heart-disease.jpg', width=700)
    if st.sidebar.button("GO!"):
        df = input_df.copy()
        st.write(df)
        with open("generate_heart_disease.pkl", "rb") as f:
            model = pickle.load(f)

        prediction = model.predict(df)
        result = ['No Heart Disease' if prediction == 0 else 'Heart Disease']
        with st.spinner('Wait for it...'):
            time.sleep(3)
            st.success('This patient has {}'.format(result[0]))
            st.balloons()
# Home Page
if nav == "Home":
    st.title("Capstone Project DQLab")
    st.write('''
    **Machine Learning & AI Track**
    
    Hallo, saya [Budi Raharjo](https://www.linkedin.com/in/budi-raharjo-9b7a1b1b0/), saya adalah seorang mahasiswa di 
    salah satu perguruan tinggi di Indonesia. Saya mengambil jurusan Teknik Informatika. Saya mengikuti kelas Machine 
    Learning & AI di DQLab Academy. Ini adalah capstone project saya.
    ''')
    st.image("https://www.endocrine.org/-/media/endocrine/images/patient-engagement-webpage/condition-page-images/cardiovascular-disease/cardio_disease_t2d_pe_1796x943.jpg",
             width=700, caption="www.endocrine.org")

    st.write('''
    **Project Overview**
    
    Cardiovascular disease (CVDs) atau penyakit jantung merupakan penyebab kematian nomor satu secara global dengan 17,9 
    juta kasus kematian setiap tahunnya. Penyakit jantung disebabkan oleh hipertensi, obesitas, dan gaya hidup yang 
    tidak sehat. Deteksi dini penyakit jantung perlu dilakukan pada kelompok risiko tinggi agar dapat segera mendapatkan 
    penanganan dan pencegahan. Project ini bertujuan untuk memprediksi apakah seseorang memiliki penyakit jantung atau 
    tidak berdasarkan beberapa kriteria tertentu. Dataset yang digunakan adalah dataset penyakit jantung dari [UCI 
    Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
    ''')

    st.write('''
    **Project Objective**
    
    Tujuan dari project ini adalah untuk memprediksi apakah seseorang memiliki penyakit jantung atau tidak berdasarkan
    beberapa kriteria tertentu.
    ''')

elif nav == 'Dataset':
    st.title("Dataset")
    st.write('''
    **Dataset Overview**
    
    Dataset yang digunakan adalah dataset penyakit jantung dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/heart+Disease).
    Dataset ini memiliki 303 baris dan 14 kolom. Kolom target adalah kolom `target` yang menunjukkan apakah seseorang
    memiliki penyakit jantung atau tidak. Jika memiliki penyakit jantung, maka nilai kolom `target` adalah 1, jika tidak
    memiliki penyakit jantung, maka nilai kolom `target` adalah 0.
    ''')
    st.write('''
    **Dataset Description**
    
    Berikut adalah deskripsi dari dataset yang digunakan.
    
    1. `age` : usia dalam tahun (umur)
    2. `sex` : jenis kelamin (1 = laki-laki; 0 = perempuan)
    3. `cp` : tipe nyeri dada
        - 1: typical angina
        - 2: atypical angina
        - 3: non-anginal pain
        - 4: asymptomatic
    4. `trestbps` : tekanan darah istirahat (dalam mm Hg saat masuk ke rumah sakit)
    5. `chol` : serum kolestoral dalam mg/dl
    6. `fbs` : gula darah puasa > 120 mg/dl (1 = true; 0 = false)
    7. `restecg` : hasil elektrokardiografi istirahat
        - 0: normal
        - 1: memiliki ST-T wave abnormalitas (T wave inversions and/or ST elevation or depression of > 0.05 mV)
        - 2: menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri menurut kriteria Estes
    8. `thalach` : detak jantung maksimum yang dicapai
    9. `exang` : angina yang diinduksi oleh olahraga (1 = yes; 0 = no)
    10. `oldpeak` : ST depression yang disebabkan oleh olahraga relatif terhadap istirahat
    11. `slope` : kemiringan segmen ST latihan puncak
        - 1: naik
        - 2: datar
        - 3: turun
    12. `ca` : jumlah pembuluh darah utama (0-3) yang diwarnai dengan flourosopy
    13. `thal` : 3 = normal; 6 = cacat tetap; 7 = cacat yang dapat dibalik
    14. `target` : memiliki penyakit jantung atau tidak (1 = yes; 0 = no)
    ''')

    # show dataset
    st.write('''
    **Show Dataset**
    ''')
    st.dataframe(df.head())

    # show dataset shape
    st.write(f'''**Dataset Shape:** {df.shape}''')

    # show dataset description
    st.write('''
    **Dataset Description**
    ''')
    st.dataframe(df.describe())

    # show dataset count visualization
    st.write('''
    **Dataset Count Visualization**
    ''')
    views = st.selectbox("Select Visualization", ("", "Target", "Age"))
    if views == "Target":
        st.bar_chart(df.target.value_counts())
        st.write('''
        `Target` adalah kolom yang menunjukkan apakah seseorang memiliki penyakit jantung atau tidak. Jika memiliki penyakit
        jantung, maka nilai kolom `target` adalah 1, jika tidak memiliki penyakit jantung, maka nilai kolom `target` adalah 0.
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah orang yang memiliki penyakit jantung lebih banyak daripada
        yang tidak memiliki penyakit jantung sejumlah 526 orang dibandingkan 499 orang.
        ''')
    elif views == "Age":
        st.bar_chart(df['age'].value_counts())
        st.write('''
        Berdasarkan visualisasi di atas, dapat dilihat bahwa jumlah orang yang memiliki penyakit jantung paling banyak berada
        pada usia 58 tahun sebanyak 68 orang. Sedangkan jumlah orang yang tidak memiliki penyakit jantung paling banyak berada
        rentang 74-76 tahun sebanyak 9 orang.''')

elif nav == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    st.write('''
    **Data Cleaning**
    
    Pada tahap ini, dilakukan pengecekan terhadap data apakah terdapat data yang kosong atau tidak. Jika terdapat data yang
    kosong, maka data tersebut akan dihapus.
    ''')
    st.write('''
    Informasi yang akan kita gali adalah feature pada kesalahan penulisan:
    1. Feature `CA`: Memiliki 5 nilai dari rentang 0-4, maka dari itu nilai 4 diubah menjadi NaN (karena seharusnya tidak ada)
    2. Feature `thal`: Memiliki 4 nilai dari rentang 0-3, maka dari itu nulai 0 diubah menjadi NaN (karena seharusnya tidak ada)
    ''')
    views = st.radio("Show Data", ("CA", "Thal"))
    if views == "CA":
        st.write('''
        **Feature CA**
        
        Feature CA memiliki 5 nilai dari rentang 0-4, maka dari itu nilai 4 diubah menjadi NaN (karena seharusnya tidak ada)
        ''')
        st.dataframe(df.ca.value_counts().to_frame().transpose())
        st.write('''
        **Show Data After Cleaning**
        ''')
        st.dataframe(df.ca.replace(0, np.nan).value_counts().to_frame().transpose())
    elif views == "Thal":
        st.write('''
        **Feature Thal**
        
        Feature Thal memiliki 4 nilai dari rentang 0-3, maka dari itu nulai 0 diubah menjadi NaN (karena seharusnya tidak ada)
        ''')
        st.dataframe(df.thal.value_counts().to_frame().transpose())
        st.write('''
        **Show Data After Cleaning**
        ''')
        st.dataframe(df.thal.replace(0, np.nan).value_counts().to_frame().transpose())

elif nav == "Modelling":
    st.header("Modelling")
    var = st.select_slider("Select Model", ("Before Tuning", "After Tuning", "ROC-AUC", "Kesimpulan"))
    if var == "Before Tuning":
        accuracy_score = {
            'Logistic Regression': 0.85,
            'Decision Tree': 0.78,
            'Random Forest': 0.87,
            'MLP Classifier': 0.87,
        }
        st.write('''
        **Model Before Tuning**
        
        Berikut adalah hasil akurasi dari model sebelum dilakukan tuning.
        ''')
        st.dataframe(pd.DataFrame(accuracy_score.items(), columns=['Model', 'Accuracy Score']))
        st.write('''
        Berdasarkan hasil akurasi dari model sebelum dilakukan tuning, dapat dilihat bahwa model dengan akurasi tertinggi
        adalah Random Forest dan MLP Classifier dengan akurasi 0.87.
        ''')

    elif var == "After Tuning":
        accuracy_score = {
            'Logistic Regression': 0.87,
            'Decision Tree': 0.83,
            'Random Forest': 0.88,
            'MLP Classifier': 0.89,
        }
        st.write('''
        **Model After Tuning**
        
        Berikut adalah hasil akurasi dari model setelah dilakukan tuning.
        ''')
        st.dataframe(pd.DataFrame(accuracy_score.items(), columns=['Model', 'Accuracy Score']))
        st.write('''
        Berdasarkan hasil akurasi dari model setelah dilakukan tuning, dapat dilihat bahwa model dengan akurasi tertinggi
        adalah MLP Classifier dengan akurasi 0.89.
        ''')

    elif var == "Kesimpulan":
        st.write('''
        **Kesimpulan**
        
        Berdasarkan hasil akurasi dari model sebelum dan setelah dilakukan tuning, dapat disimpulkan bahwa model dengan
        akurasi tertinggi adalah MLP Classifier dengan akurasi 0.89. Jika kamu mendownload model ini, maka kamu akan mendapatkan
        di link berikut ini [Download Model](https://drive.google.com/file/d/1-8Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z3Z/view?usp=sharing).
        ''')

elif nav == 'Prediction':
    st.header("My Apps")
    heart()

elif nav == "About":
    st.title("About Me")
    st.image("https://avatars.githubusercontent.com/u/59464302?v=4", width=200)
    st.write('''
    **Budi Raharjo**
    
    Saya adalah seorang mahasiswa di salah satu perguruan tinggi di Indonesia. Saya mengambil jurusan Teknik Informatika.
    Saya mengikuti kelas Machine Learning & AI di DQLab Academy. Ini adalah capstone project saya.
    ''')
    st.write('''
    **Contact Me**
    
    - [LinkedIn](https://www.linkedin.com/in/budi-raharjo-9b7a1b1b0/)
    - [Github](https://www.github.com/budiraharjo)
    - [Instagram](https://www.instagram.com/budiraharjo)
    - [Facebook](https://www.facebook.com/budiraharjo)
    - [Twitter](https://www.twitter.com/budiraharjo)
    ''')

    select_item = st.selectbox("My Project Preview", ('', 'Iris Prediction', 'Housing Price Prediction'))
    if select_item == "Iris Prediction":
        st.header("Ini Contoh Iris")
    elif select_item == "Housing Price Prediction":
        st.header("Ini Contoh Housing Price Prediction")


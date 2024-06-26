import streamlit as st
st.set_page_config(layout="wide")
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import nltk
import swifter
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer

tab1, tab2 = st.tabs(["SVM", "SVM with ANOVA"])

def cleaning(Penjelasan):
  Penjelasan = re.sub(r'@[A-Za-a0-9]+',' ',Penjelasan)
  Penjelasan = re.sub(r'_x000D__x000D__x000D_',' ',Penjelasan)
  Penjelasan = re.sub(r'SCROLL TO CONTINUE WITH CONTENT_x000D_',' ',Penjelasan)
  Penjelasan = re.sub(r'#[A-Za-z0-9]+',' ',Penjelasan)
  Penjelasan = re.sub(r"http\S+",' ',Penjelasan)
  Penjelasan = re.sub(r'[0-9]+',' ',Penjelasan)
  Penjelasan = re.sub(r'\n',' ',Penjelasan)
  Penjelasan = re.sub(r"[-()\"#/@;:<>{}'+=~|.!?,_]", " ", Penjelasan)
  Penjelasan = Penjelasan.strip(' ')
  return Penjelasan

def casefoldingText(berita):
        berita =  berita.lower()
        return berita


# nltk.download('punkt')

def word_tokenize_wrapper(text):
    return word_tokenize(text)

nltk.download('stopwords')



nltk.download('stopwords')
from nltk.corpus import stopwords

# Mengambil daftar stopword dari NLTK
daftar_stopword_nltk = stopwords.words('indonesian')

def stopwordText(words):
  return [word for word in words if word not in daftar_stopword_nltk]

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


factory = StemmerFactory()
stemmer = factory.create_stemmer()

def stemmed_wrapper(term):
    return stemmer.stem(term)


def prediksi(text):
    tfidf_vektor = vectorizer.transform([text])
    if option == "RBF":
        # st.write("RBF CUY")
        joblib_models = joblib.load("model/model_fold9_rbf_new.pkl")
    elif option == "Linear":
        # st.write("Linear Cuy")
        joblib_models = joblib.load("model/model_fold3_linear_new.pkl")
    elif option == "Sigmoid":
        # st.write("Sigmoid Cuy")
        joblib_models = joblib.load("model/model_fold3_sigmoid_new.pkl")
    elif option == "Polynomial":
        # st.write("Sigmoid Cuy")
        joblib_models = joblib.load("model/model_fold1_poly_new.pkl")

    pred = joblib_models.predict(tfidf_vektor)
    if pred == 1:
        hate = "Alam"
    elif pred == 2:
        hate = "Buatan"
    elif pred == 3:
        hate = "Budaya"
    else:
        hate = "NON PARIWISATA"
    return hate


def prediksi_anova(text):
        tfidf_vektor_anova = vectorizer.transform([text])
        if option_anova == "RBF ":
            # st.write("RBF CUY")
            joblib_models_anova = joblib.load("model/anova/anova_svm_model_anova_1_rbf_fix.pkl")
        elif option_anova == "Linear ":
            # st.write("Linear Cuy")
            joblib_models_anova = joblib.load("model/anova/anova_svm_model_anova_3_linear_fix.pkl")
        elif option_anova == "Sigmoid ":
            # st.write("Sigmoid Cuy")
            joblib_models_anova = joblib.load("model/anova/anova_svm_model_fold_2_sigmoid_fix.pkl")
        elif option_anova == "Polynomial ":
            # st.write("Sigmoid Cuy")
            joblib_models_anova = joblib.load("model/anova/anova_svm_model_fold_1_poly_fix (1).pkl")

        pred_anova = joblib_models_anova.predict(tfidf_vektor_anova)
        if pred_anova == 1:
            hate = "Alam"
        elif pred_anova == 2:
            hate = "Buatan"
        elif pred_anova == 3:
            hate = "Budaya"
        else:
            hate = "NON PARIWISATA"
        return hate

st.markdown(
    """
    <style>
    /* Ganti warna dan ukuran tombol */
    .stButton>button:first-child {
        background-color: #fde380;
        padding: 12px 340px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        color: black;
    }a
    .stTextArea textarea{
        background-color: white;
        color: black;
    }
    .stRadio [role=radiogroup]{
        display: flex;
        flex-direction: row;
        align-items: center;
        justify-content: center;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with tab1:
    st.title(f"CLASSIFY OUR TOURISM NEWS")

    # joblib_models = joblib.load("model/model_fold9_rbf_new.pkl")
    data = pd.read_csv("data/data_prepos_fix.csv")

    vectorizer = TfidfVectorizer()
    data_df = vectorizer.fit_transform(data["Stemming"])

    left_column, right_column = st.columns(2)

    with left_column:
        option =  st.radio("Select Your Kernel", ["RBF", "Linear", "Polynomial", "Sigmoid"])
        masukkan_kalimat = st.text_area("Masukan Text Berita: ",  height=200, key="textarea")
        st.write(f"You write {len(masukkan_kalimat)} characters.")



        if st.button("Klasifikasi"):
            masukkan_kalimat = cleaning(masukkan_kalimat)
            masukkan_kalimat = masukkan_kalimat.lower()
            masukkan_kalimat = word_tokenize_wrapper(masukkan_kalimat)
            masukkan_kalimat = stopwordText(masukkan_kalimat)
            masukkan_kalimat = " ".join(masukkan_kalimat)
            masukkan_kalimat = stemmer.stem(masukkan_kalimat)
            st.subheader(f"Kategori : {prediksi(masukkan_kalimat)}")
            st.toast("Classification Complate", icon='🎉')


    with right_column:
        st.image('image/Artboard 8 copy.png')

with tab2:
    st.title(f"CLASSIFY OUR TOURISM NEWS")

    # joblib_models = joblib.load("model/model_fold9_rbf_new.pkl")
    data_anova = pd.read_csv("data/data_fitur_anova.csv")

    vectorizer = TfidfVectorizer()
    data_df_anova = vectorizer.fit_transform(data_anova["Clean"])

    left_column, right_column = st.columns(2)

    with left_column:
        option_anova =  st.radio("Select Your Kernel", ["RBF ", "Linear ", "Polynomial ", "Sigmoid "])
        masukkan_kalimat = st.text_area("Masukan Text Berita: ",  height=200, key="textarea ")
        # berita = masukkan_kalimat
        st.write(f"You write {len(masukkan_kalimat)} characters.")



        if st.button("Klasifikasi "):
            masukkan_kalimat = cleaning(masukkan_kalimat)
            masukkan_kalimat = masukkan_kalimat.lower()
            masukkan_kalimat = word_tokenize_wrapper(masukkan_kalimat)
            masukkan_kalimat = stopwordText(masukkan_kalimat)
            masukkan_kalimat = " ".join(masukkan_kalimat)
            masukkan_kalimat = stemmer.stem(masukkan_kalimat)
            st.subheader(f"Kategori : {prediksi_anova(masukkan_kalimat)}")
            st.toast("Classification Complate", icon='🎉')
        # st.write(berita)
    with right_column:
        st.image('image/Artboard 8 copy.png')
        # st.write(f'<p style="border-color: white; border-width: 2px;, border-style: solid;">{masukkan_kalimat}</p>', unsafe_allow_html=True)

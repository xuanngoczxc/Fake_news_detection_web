from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk
nltk.download('stopwords')

app = Flask(__name__)

# Tải mô hình SVM và TfidfVectorizer
svm_model = SVC(kernel='linear')
svm_model = joblib.load('fake_news_detection.pkl')  # đường dẫn đến tệp mô hình SVM
vectorization = TfidfVectorizer()
vectorization = joblib.load('vectorization.pkl')  # đường dẫn đến tệp vectorizer

# Tải hàm PorterStemmer và wordopt
ps = PorterStemmer()
def wordopt(text):
    text = re.sub('[^a-zA-Z]', ' ',text)
    text = text.lower()
    text = text.split()
    text = [ps.stem(word) for word in text if not word in stopwords.words('english')]
    text = ' '.join(text)
    return text

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Nhận văn bản tin tức từ biểu mẫu
    news = request.form['news_text']

    # Xử lý trước văn bản tin tức
    input_data = {"text": [news]}
    new_def_test = pd.DataFrame(input_data)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    
    # Vector hóa dữ liệu đầu vào
    vectorized_input_data = vectorization.transform(new_def_test["text"]).toarray()

    # Dự đoán bằng mô hình SVM
    prediction = svm_model.predict(vectorized_input_data)

    # Trả về kết quả dự đoán
    if prediction[0] == 1:
        result = "Tin thật"
    else:
        result = "Tin giả"
    # Hiển thị mẫu kết quả với dự đoán
    return render_template('result.html', prediction=result, news_text = news)

if __name__ == '__main__':
    app.run(debug=True)
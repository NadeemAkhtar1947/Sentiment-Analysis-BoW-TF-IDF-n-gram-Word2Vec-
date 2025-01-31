# 📌 Sentiment Analysis on IMDB Dataset 🎭

## 📝 Overview
This project performs **Sentiment Analysis** on the IMDB dataset using **Natural Language Processing (NLP)** techniques. The goal is to classify movie reviews as either **positive** or **negative** using various machine learning models.

## 📂 Dataset
- **📌 Source:** [IMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- **📊 Size:** 15,000 movie reviews (after filtering)
- **📑 Columns:**
  - `review`: The textual review of the movie.
  - `sentiment`: The target label (positive/negative).

## 🔄 Data Preprocessing
1. **🧹 Data Cleaning:**
   - Removed **HTML tags**, **special characters**, **URLs**, and **numbers**.
   - Converted text to **lowercase**.
2. **✂️ Tokenization & Stopword Removal:**
   - Tokenized text and removed **common stopwords**.
3. **🌱 Stemming:**
   - Used **Porter Stemmer** to reduce words to their root form.
4. **⚙️ Feature Engineering:**
   - Applied various text vectorization techniques:
     - **Bag of Words (BoW)** 📦
     - **TF-IDF (Term Frequency-Inverse Document Frequency)** 🔢
     - **n-grams (Unigrams, Bigrams)** 🔍
     - **Word2Vec embeddings** 🧠

## 🤖 Model Training & Evaluation
### **1️⃣ Gaussian Naïve Bayes**
- Used **Bag of Words (BoW)** for feature representation.
- Achieved an **accuracy of 64.91%**.

### **2️⃣ Random Forest Classifier** 🌲
- Trained using **BoW, TF-IDF, and Word2Vec** features.
- **🔬 Accuracy Comparison:**
  - **BoW** (All Features): **84.09%** 🎯
  - **BoW (Top 3000 Features)**: **82.86%** 📉
  - **BoW (n-gram = (1,2), Top 5000 Features)**: **83.92%** 📊
  - **TF-IDF Features:** **84.16%** 🏆
  - **Word2Vec Embeddings:** **81.69%** 🔍

## 📊 Results & Observations
- **Random Forest** performed the best with **TF-IDF features (84.16%)** 🏆.
- **n-gram (1,2) BoW model** also provided competitive results.
- **Word2Vec embeddings** performed slightly lower but captured **semantic meaning** better 🧠.

## 🚀 Installation & Usage
### **📌 Requirements**
Ensure you have the following Python libraries installed:
```bash
pip install numpy pandas seaborn matplotlib nltk scikit-learn gensim
```

### **▶️ Running the Project**
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd sentiment-analysis-imdb
   ```
2. Run the preprocessing script:
   ```python
   python preprocess.py
   ```
3. Train and evaluate the model:
   ```python
   python train.py
   ```

## 🚀 Future Improvements
- Experiment with **deep learning models** like **LSTMs, CNNs, and Transformer-based architectures (BERT, GPT-3)** 🧠.
- Implement **hyperparameter tuning** to improve model performance 🔍.
- Deploy the model as a **web app using Flask or Streamlit** 🌐.

## 🎯 Conclusion
This project demonstrates the effectiveness of **traditional machine learning models** for **sentiment analysis**. Different vectorization techniques and classifiers were evaluated, with **Random Forest using TF-IDF** yielding the best accuracy 🎯.

If this project helped in your learning, please consider supporting it! 🚀

**Happy Learning!** 😊📚


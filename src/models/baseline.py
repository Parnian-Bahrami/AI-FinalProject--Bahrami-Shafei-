import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

def run_baseline_model(data_path):
    # 1. بارگذاری داده‌های پیش‌پردازش شده (خروجی بخش ۳)
    if not os.path.exists(data_path):
        print("Data not found! Please run preprocessing first.")
        return
    
    df = pd.read_csv(data_path)
    # حذف سطرهای خالی احتمالی بعد از پیش‌پردازش
    df = df.dropna(subset=['cleaned_instruction'])

    # 2. تقسیم داده‌ها به آموزش و تست (الزام مستند پروژه)
    X = df['cleaned_instruction']
    y = df['intent_encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. تبدیل متن به عدد با TF-IDF
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    # 4. آموزش مدل Logistic Regression (Baseline)
    print("Training Baseline Model (Logistic Regression)...")
    model = LogisticRegression(max_iter=1000, multi_class='multinomial')
    model.fit(X_train_tfidf, y_train)

    # 5. ارزیابی اولیه
    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Baseline Model Accuracy: {acc:.4f}")
    print("\nClassification Report (Top 10 Classes):")
    print(classification_report(y_test, y_pred, digits=4))

    # 6. ذخیره مدل در پوشه models (طبق ساختار درختی)
    os.makedirs('../models', exist_ok=True)
    joblib.dump(model, '../models/baseline_model.pkl')
    joblib.dump(vectorizer, '../models/tfidf_vectorizer.pkl')
    print("Model saved to models/baseline_model.pkl")

if __name__ == "__main__":
    # مسیر داده‌های تمیز شده
    run_baseline_model('../data/processed/cleaned_data.csv')
    
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

# دانلود پیش‌نیازهای NLTK
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.label_encoder = LabelEncoder()

    def clean_text(self, text):
        # 1. حروف کوچک
        text = str(text).lower()
        # 2. حذف متغیرهای داخل آکولاد مثل {{Order Number}}
        text = re.sub(r'\{\{.*?\}\}', 'variable', text)
        # 3. حذف علائم نگارشی و اعداد
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # 4. توکن‌بندی
        tokens = nltk.word_tokenize(text)
        # 5. حذف Stopwords و Lemmatization
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word) 
            for word in tokens if word not in self.stop_words
        ]
        return " ".join(cleaned_tokens)

    def process_dataframe(self, df):
        print("--- Starting Preprocessing ---")
        # اعمال تمیزکاری روی ستون instruction
        df['cleaned_instruction'] = df['instruction'].apply(self.clean_text)
        
        # کدگذاری Labelها (Intents)
        df['intent_encoded'] = self.label_encoder.fit_transform(df['intent'])
        
        print(f"Total Unique Classes: {len(self.label_encoder.classes_)}")
        print("--- Preprocessing Finished ---")
        return df, self.label_encoder

# --- اجرای نمونه جهت تست ---
if __name__ == "__main__":
    # مسیر دیتاست را بر اساس ساختار درختی تنظیم کنید
    df = pd.read_csv('../data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv')
    
    preprocessor = TextPreprocessor()
    df_processed, encoder = preprocessor.process_dataframe(df)
    
    # ذخیره داده‌های پیش‌پردازش شده در پوشه مربوطه طبق ساختار درختی
    os.makedirs('../data/processed', exist_ok=True)
    df_processed.to_csv('../data/processed/cleaned_data.csv', index=False)
    print("Processed data saved to data/processed/cleaned_data.csv")
    
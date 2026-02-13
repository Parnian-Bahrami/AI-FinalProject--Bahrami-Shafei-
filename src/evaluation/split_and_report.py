import pandas as pd
from sklearn.model_selection import train_test_split
import os

def create_experiment_plan(data_path):
    # ۱. بارگذاری داده‌های تمیز شده از بخش ۳
    if not os.path.exists(data_path):
        print("❌ ابتدا باید مرحله پیش‌پردازش را اجرا کنید.")
        return
    
    df = pd.read_csv(data_path)
    
    # ۲. تقسیم‌بندی داده‌ها (۷۰٪ آموزش، ۱۵٪ تست، ۱۵٪ اعتبارسنجی)
    # استفاده از stratify برای حفظ توازن کلاس‌ها در تمام مجموعه‌ها
    train_df, temp_df = train_test_split(
        df, test_size=0.30, random_state=42, stratify=df['intent_encoded']
    )
    
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, random_state=42, stratify=temp_df['intent_encoded']
    )
    
    # ۳. ثبت نتایج آماری تقسیم‌بندی
    stats = {
        "Total Samples": len(df),
        "Train Set Size": len(train_df),
        "Validation Set Size": len(val_df),
        "Test Set Size": len(test_df),
        "Number of Classes": df['intent_encoded'].nunique()
    }
    
    # ۴. ذخیره گزارش در پوشه results/metrics طبق ساختار درختی
    os.makedirs('../../results/metrics', exist_ok=True)
    with open('../../results/metrics/experiment_summary.txt', 'w', encoding='utf-8') as f:
        f.write("=== Experiment Plan Summary ===\n")
        for key, value in stats.items():
            f.write(f"{key}: {value}\n")
            print(f"{key}: {value}")
            
    # ۵. ذخیره مجموعه‌ها برای فاز دوم (اختیاری اما حرفه‌ای)
    train_df.to_csv('../../data/processed/train.csv', index=False)
    val_df.to_csv('../../data/processed/val.csv', index=False)
    test_df.to_csv('../../data/processed/test.csv', index=False)
    
    print("\n✅ گزارش تقسیم‌بندی داده‌ها با موفقیت در پوشه results ذخیره شد.")

if __name__ == "__main__":
    create_experiment_plan('../../data/processed/cleaned_data.csv')
    
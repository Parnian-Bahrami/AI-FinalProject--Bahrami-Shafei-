import os

# تنظیمات اولیه پروژه
PROJECT_NAME = "AI-FinalProject-Chatbot"
folders = [
    "data/raw", "data/processed", "notebooks/EDA", 
    "src/preprocessing", "results/charts", "models"
]

# ایجاد پوشه‌ها
for folder in folders:
    os.makedirs(os.path.join(PROJECT_NAME, folder), exist_ok=True)

# ایجاد یک فایل تنظیمات برای تم نمودارها (برای بخش EDA)
config_content = """
# Global Config for Project
COLORS = {
    'primary': '#4facfe',
    'secondary': '#00f2fe',
    'background': '#1a1c2c',
    'text': '#e0e6ed'
}
DATA_PATH = 'data/raw/Bitext_Sample_Customer_Support_Training_Dataset_27K_responses-v11.csv'
"""

with open(os.path.join(PROJECT_NAME, "src/config.py"), "w") as f:
    f.write(config_content)

print(f"✅ Structure for '{PROJECT_NAME}' created with custom theme config.")

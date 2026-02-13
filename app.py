import gradio as gr
import torch
from transformers import BertTokenizer
import sys
import os

# اضافه کردن مسیرهای لازم
sys.path.append(os.path.abspath('./src'))
from models.architectures import BERTIntentClassifier
from inference.response_manager import ChatbotResponseManager

# ۱. پیکربندی و بارگذاری مدل
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_LEN = 64

# فرض می‌کنیم اسامی کلاس‌ها را از فاز قبل داریم (می‌توانی لیست را مستقیم اینجا بگذاری)
# این لیست باید دقیقاً همان ۲۷ کلاسی باشد که مدل با آن آموزش دیده است
intent_names = ['get_order', 'get_refund', 'cancel_order', 'check_shipping', 'edit_account', ...] # بقیه کلاس‌ها

model = BERTIntentClassifier(len(intent_names))
model.load_state_dict(torch.load('./models/best_model_state.bin', map_location=device))
model.to(device)
model.eval()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
response_manager = ChatbotResponseManager(intent_names)

def chatbot_interface(user_message):
    # الف) پیش‌پردازش ورودی
    encoding = tokenizer.encode_plus(
        user_message,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # ب) پیش‌بینی قصد
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, preds = torch.max(outputs, dim=1)
        intent_idx = preds.item()
    
    # ج) تولید پاسخ و استخراج موجودیت
    response = response_manager.generate_response(intent_idx, user_message)
    intent_label = intent_names[intent_idx]
    
    return f"Detected Intent: {intent_label}\n\nChatbot: {response}"

# ۲. ساخت رابط گرافیکی با Gradio
demo = gr.Interface(
    fn=chatbot_interface,
    inputs=gr.Textbox(lines=2, placeholder="Type your request here (e.g., 'I want to track order #12345')"),
    outputs="text",
    title="🤖 Customer Support AI Chatbot",
    description="This chatbot uses a Fine-tuned BERT model to detect intents and assist customers.",
    theme="soft",
    examples=[
        ["How can I get a refund for my last purchase?"],
        ["Where is my order #98765?"],
        ["I want to cancel my subscription."]
    ]
)

if __name__ == "__main__":
    # اجرای برنامه (اگر روی سیستم شخصی هستی share=True را بردار)
    demo.launch(share=True)

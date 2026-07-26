import os
import whisper
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import torch
from groq import Groq
from transformers import VitsModel, AutoTokenizer
from dotenv import load_dotenv

# load api key from .env
load_dotenv()  # .env 
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(" .env ဖိုင်ထဲမှာ GROQ_API_KEY မတွေ့ရပါ။ ကျေးဇူးပြု၍ စစ်ဆေးပါ။")

# groq client
client = Groq(api_key=GROQ_API_KEY)

# speech to text
def record_and_transcribe():
    print("ကျေးဇူးပြု၍ မေးခွန်းကို ပြောပါ...")
    sample_rate = 16000
    duration = 30
    print("စကားပြောနေပါ...")
    
    recording = sd.rec(int(sample_rate * duration), samplerate=sample_rate,
                       channels=1, dtype='float32')
    sd.wait()
    print("သွင်းပြီးပါပြီ။ စဉ်းစားနေပါ...")
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        temp_path = f.name
        audio_int16 = (recording * 32767).astype(np.int16)
        wav.write(temp_path, sample_rate, audio_int16)
        
        model = whisper.load_model("base")
        result = model.transcribe(temp_path, language="myanmar")
        os.unlink(temp_path)
        
        print(f"မေးခွန်း: {result['text']}")
        return result['text']

# LLM
def ask_groq(question):
    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # qwen-3-32b
            messages=[
                {"role": "system", "content": "သင်သည် Technological University ကျောင်းသားများ၏ မေးခွန်းကို မြန်မာလိုသာ ဖြေပေးရမည့် AI အကူဖြစ်သည်။ အဖြေကို တိုတိုရှင်းရှင်းနဲ့ အသုံးဝင်အောင်ဖြေပါ။"},
                {"role": "user", "content": f"အောက်ပါမေးခွန်းကို မြန်မာလို အသေးစိတ်ဖြေပေးပါ။\n\nမေးခွန်း: {question}\n\nအဖြေ (မြန်မာလို):"}
            ],
            temperature=0.7,
            max_tokens=500
        )
        answer = completion.choices[0].message.content
        print(f"အဖြေ: {answer}")
        return answer
    except Exception as e:
        print(f"Groq အမှား: {e}")
        return "တောင်းပန်ပါတယ်။ ကျွန်တော် အခုအချိန်မှာ အဖြေမပေးနိုင်သေးပါဘူး။"

# text to speech
def speak_text(text):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("အသံထွက်ရန် ပြင်ဆင်နေပါ...")
    try:
        model = VitsModel.from_pretrained("facebook/mms-tts-mya").to(device)
        tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-mya")
        inputs = tokenizer(text, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model(**inputs).waveform
        audio_numpy = output.squeeze().cpu().numpy()
        print("🔊 ဖြေနေပါ...")
        sd.play(audio_numpy, samplerate=model.config.sampling_rate)
        sd.wait()
    except Exception as e:
        print(f"TTS အလုပ်မလုပ်ပါ: {e}")
        print(f"အဖြေစာ: {text}")

# main loop
if __name__ == "__main__":
    print("မြန်မာ AI Voice Assistant (Groq + .env) စတင်ပါပြီ...")
    print("Ctrl+C နှိပ်ပြီး ရပ်နိုင်ပါတယ်။\n")
    
    while True:
        try:
            question = record_and_transcribe()
            if question and len(question.strip()) > 0:
                answer = ask_groq(question)
                speak_text(answer)
            else:
                print("ဘာမှမကြားရပါ။ ထပ်စမ်းပါ။")
        except KeyboardInterrupt:
            print("\nအားလုံးကောင်းပါစေ။")
            break
        except Exception as e:
            print(f"အမှားတစ်ခုဖြစ်သွားသည်: {e}")
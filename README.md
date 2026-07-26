# to activate python venv
python -m venv venv
venv\Scripts\activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate sounddevice whisper requests
pip install whisper sounddevice numpy scipy torch transformers requests
ollama pull gemma3   # or llama3.2, phi3, etc.
ollama serve
python voice_assistant.py

model_name="WYNN747/ai4burmese-padauk"


# groq api 
pip install groq whisper sounddevice numpy scipy transformers torch

# to quick
pip install torch --index-url https://download.pytorch.org/whl/cpu

pip install python-dotenv groq whisper sounddevice numpy scipy transformers torch

pip install openai-whisper sounddevice numpy scipy transformers torch groq python-dotenv
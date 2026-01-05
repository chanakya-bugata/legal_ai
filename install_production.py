"""
Production installation with model pre-download
"""
import subprocess
import sys

print("🚀 Installing production dependencies...")

# Install core packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

print("📥 Downloading required models...")

# Download spacy model
subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])

# Pre-download Legal-BERT
import transformers
transformers.AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
transformers.AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")

print("✅ Installation complete!")
print("Run: streamlit run streamlit_app/main.py")

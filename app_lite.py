from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import time
import requests
import base64
import os
from dotenv import load_dotenv
from PIL import Image
import pytesseract
import io
import json
from gtts import gTTS

# Load environment variables
load_dotenv()

app = Flask(__name__)
# Enable CORS for API routes so Flutter web app can access backend from another origin
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Serve service worker from root for PWA scope
@app.route('/sw.js')
def service_worker():
    return send_from_directory('static', 'sw.js', mimetype='application/javascript')

# Serve Assets folder
@app.route('/Assets/<path:filename>')
def serve_assets(filename):
    return send_from_directory('Assets', filename)

# Serve manifest with correct MIME type from root for installability
@app.route('/manifest.webmanifest')
def manifest():
    return send_from_directory('static', 'manifest.webmanifest', mimetype='application/manifest+json')

# Serve static icon files
@app.route('/static/LUMA.png')
def static_icon_192():
    return send_from_directory('static', 'LUMA.png')

# Serve favicon (updated to use LUMA logo)
@app.route('/favicon.ico')
def favicon_luma():
    return send_from_directory('static', 'LUMA.png', mimetype='image/png')


class VoiceAssistant:
    def __init__(self):
        # Gemini configuration
        self.gemini_api_key = os.getenv('GEMINI_API_KEY')
        self.gemini_model = os.getenv('GEMINI_MODEL_ID', 'gemini-2.0-flash')
        self.gemini_embed_model = os.getenv('GEMINI_EMBED_MODEL_ID', 'text-embedding-004')
        # Output brevity control
        try:
            self.answer_word_limit = int(os.getenv('ANSWER_WORD_LIMIT', '40'))
        except Exception:
            self.answer_word_limit = 40
        # RAG store (lightweight, JSON persisted)
        self.rag_store_path = os.getenv('RAG_STORE_PATH', 'rag_store.json')
        self.rag_items = []  # each: {"id": str, "text": str, "embedding": [float], "meta": {...}}
        self._load_rag_store()
        print(self.ai_provider_info())
    
    def answer_question(self, question):
        """Answer questions using Gemini AI."""
        try:
            original = question or ""
            q_lower = original.lower()
            # Gemini-only path
            if self.gemini_api_key:
                context_blocks = self.retrieve_context(original, top_k=5)
                prompt = self.build_rag_prompt(original, context_blocks)
                ai_answer = self.generate_ai_answer_gemini(prompt)
                if isinstance(ai_answer, str) and ai_answer.strip():
                    return ai_answer
            # Fallback keyword responses
            if "help" in q_lower or "what can you do" in q_lower:
                return "I can read text from your webcam and answer questions. Say 'read text' or ask me anything."
            elif "read" in q_lower and "text" in q_lower:
                return "Point your webcam at text and activate 'Read Text'."
            elif "hello" in q_lower or "hi" in q_lower:
                return "Hello! I'm Liya. How can I help you today?"
            elif "time" in q_lower:
                return f"The current time is {time.strftime('%I:%M %p')}"
            else:
                return "I'm ready to help with reading text and answering questions."
        except Exception as e:
            print(f"answer_question error: {e}")
            return "Sorry, I had trouble answering that. Please try again."

    def ai_provider_info(self) -> str:
        try:
            if self.gemini_api_key:
                return f"AI Provider: Gemini ({self.gemini_model})"
            return "AI Provider: Fallback (rules-based)"
        except Exception as e:
            print(f"ai_provider_info error: {e}")
            return "AI Provider: Unknown"

    # -----------------
    # RAG: Storage/Embed
    # -----------------
    def _load_rag_store(self):
        try:
            if os.path.exists(self.rag_store_path):
                with open(self.rag_store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.rag_items = data
        except Exception as e:
            print(f"RAG load error: {e}")

    def _save_rag_store(self):
        try:
            with open(self.rag_store_path, 'w', encoding='utf-8') as f:
                json.dump(self.rag_items, f, ensure_ascii=False)
        except Exception as e:
            print(f"RAG save error: {e}")

    def add_ocr_to_rag(self, text: str, meta: dict = None):
        try:
            clean = (text or '').strip()
            if not clean:
                return
            embedding = self.embed_text(clean)
            item = {
                "id": f"ocr-{int(time.time()*1000)}",
                "text": clean,
                "embedding": embedding,
                "meta": meta or {"source": "ocr", "ts": time.time()}
            }
            self.rag_items.append(item)
            # keep store from growing indefinitely: cap to last 500 items
            if len(self.rag_items) > 500:
                self.rag_items = self.rag_items[-500:]
            self._save_rag_store()
        except Exception as e:
            print(f"RAG add error: {e}")

    def embed_text(self, text: str):
        try:
            if self.gemini_api_key:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_embed_model}:embedContent?key={self.gemini_api_key}"
                payload = {"content": {"parts": [{"text": text}]}}
                resp = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=20)
                resp.raise_for_status()
                data = resp.json()
                vec = (((data or {}).get('embedding') or {}).get('values'))
                if isinstance(vec, list):
                    return vec
            # naive fallback embedding: hashed bag-of-words
            return self._naive_embed(text)
        except Exception as e:
            print(f"Embed error: {e}")
            return self._naive_embed(text)

    def _naive_embed(self, text: str):
        tokens = text.lower().split()
        # Fixed small-dim embedding via hashing
        dim = 128
        vec = [0.0]*dim
        for tok in tokens:
            h = abs(hash(tok)) % dim
            vec[h] += 1.0
        # L2 normalize
        norm = sum(v*v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def retrieve_context(self, query: str, top_k: int = 5):
        try:
            if not self.rag_items:
                return []
            q_emb = self.embed_text(query)
            scored = []
            for it in self.rag_items:
                emb = it.get('embedding') or []
                score = self._cosine_sim(q_emb, emb)
                scored.append((score, it))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [it for _, it in scored[:top_k] if _ > 0]
            return top
        except Exception as e:
            print(f"RAG retrieve error: {e}")
            return []

    def _cosine_sim(self, a, b):
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        dot = sum((a[i] or 0.0) * (b[i] or 0.0) for i in range(n))
        na = sum((a[i] or 0.0)**2 for i in range(n)) ** 0.5 or 1.0
        nb = sum((b[i] or 0.0)**2 for i in range(n)) ** 0.5 or 1.0
        return dot / (na * nb)

    def build_rag_prompt(self, question: str, contexts: list) -> str:
        # Prompt tailored for visually impaired users, emphasizing clarity and context use
        context_texts = [f"- {c.get('text','')[:1000]}" for c in (contexts or []) if c.get('text')]
        context_block = "\n".join(context_texts) if context_texts else "(no captured context)"
        instructions = (
            "You are Liya, a helpful voice assistant for visually impaired users. "
            "Answer clearly and concisely. Use the OCR context below only if it is relevant. "
            "If the context is not helpful, answer from your general knowledge instead of saying you cannot answer. "
            "When the question asks for examples or starts with 'what animals', 'which animals', or 'list animals', "
            "respond with ONLY a concise, comma-separated list of animal names and nothing else (no extra words, no punctuation except commas). "
            "IMPORTANT: Preserve any literal asterisks (*) from the content or your answer; do not remove or escape them. "
            f"Respond in at most {self.answer_word_limit} words unless a list of names is requested.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
        )
        return instructions

    def generate_ai_answer_hf(self, prompt: str) -> str:
        """Call Hugging Face Inference API to generate an answer."""
        try:
            if not self.hf_api_key:
                return ""
            url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
            headers = {
                'Authorization': f'Bearer {self.hf_api_key}',
                'Content-Type': 'application/json'
            }
            # Rough token target from word limit
            max_new = max(32, min(256, int(self.answer_word_limit * 2)))
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new,
                    "temperature": 0.4,
                    "return_full_text": False
                }
            }
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
            if resp.status_code == 503:
                # Model loading; return a friendly message
                return "The AI model is warming up. Please ask again in a moment."
            resp.raise_for_status()
            data = resp.json()
            # Response formats vary across models
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # text-generation format
                txt = data[0].get('generated_text') or data[0].get('summary_text')
                if isinstance(txt, str):
                    return txt.strip()
            if isinstance(data, dict):
                # Some models return dicts
                possible = data.get('generated_text') or data.get('summary_text')
                if isinstance(possible, str):
                    return possible.strip()
            # As last resort, stringify
            return (str(data)[:500] + '...') if data else ""
        except Exception as e:
            print(f"HF generate error: {e}")
            return ""

    def generate_ai_answer_gemini(self, prompt: str) -> str:
        """Call Google Gemini via Generative Language API to generate an answer."""
        try:
            if not self.gemini_api_key:
                return ""
            # API endpoint for text generation
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
            # Configure generation with a token cap aligned to word limit
            max_tokens = max(64, min(512, int(self.answer_word_limit * 3)))
            payload = {
                "contents": [{"parts": [{"text": str(prompt)}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.4,
                    "topP": 0.9,
                    "topK": 40
                }
            }
            headers = {"Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get('candidates') or []
            if candidates:
                first = candidates[0] or {}
                content = first.get('content') or {}
                parts = content.get('parts') or []
                texts = [p.get('text', '') for p in parts if isinstance(p, dict)]
                combined = " ".join([t for t in texts if t]).strip()
                if combined:
                    return combined
            return ""
        except Exception as e:
            print(f"Gemini generation error: {e}")
            return ""

    def ocr_image(self, pil_image: Image.Image):
        """Perform OCR on a provided PIL image using OCR.space API first, with Tesseract fallback."""
        try:
            # Try OCR.space API first if configured
            if self.ocr_api_key:
                text = self.ocr_with_api(pil_image)
                if text and text.strip():
                    return text.strip()
            
            # Fallback to local Tesseract OCR
            try:
                text = pytesseract.image_to_string(pil_image)
                return text.strip() if text and text.strip() else "No text detected in the image"
            except Exception as tesseract_error:
                print(f"Tesseract OCR error: {tesseract_error}")
                return "No text detected in the image"
                
        except Exception as e:
            print(f"OCR error: {e}")
            return "Error processing image for text recognition"

    def ocr_with_api(self, image):
        try:
            if not self.ocr_api_key:
                return None
            url = 'https://api.ocr.space/parse/image'
            buf = io.BytesIO()
            image.save(buf, format='PNG')
            buf.seek(0)
            img_base64 = base64.b64encode(buf.read()).decode('utf-8')
            data = {
                'apikey': self.ocr_api_key,
                'base64Image': f'data:image/png;base64,{img_base64}',
                'language': 'eng',
                'isOverlayRequired': False
            }
            response = requests.post(url, data=data, timeout=10)
            result = response.json()
            if result.get('IsErroredOnProcessing', False):
                return None
            text = ""
            for parsed_result in result.get('ParsedResults', []):
                text += parsed_result.get('ParsedText', '')
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"OCR API error: {e}")
            return None

# -----------------
# TTS API (gTTS-based)
# -----------------
@app.route('/api/speak', methods=['POST', 'GET'])
def api_speak():
    try:
        text = ''
        if request.method == 'POST':
            data = request.get_json(silent=True) or {}
            text = (data.get('text') or '').strip()
        else:
            text = (request.args.get('text') or '').strip()
        if not text:
            return jsonify({"error": "missing text"}), 400
        tts = gTTS(text=text, lang='en', slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return send_file(buf, mimetype='audio/mpeg', as_attachment=False, download_name='speech.mp3')
    except Exception as e:
        print(f"/api/speak error: {e}")
        return jsonify({"error": "tts_failed"}), 500

    # -----------------
    # RAG: Storage/Embed
    # -----------------
    def _load_rag_store(self):
        try:
            if os.path.exists(self.rag_store_path):
                with open(self.rag_store_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.rag_items = data
        except Exception as e:
            print(f"RAG load error: {e}")

    def _save_rag_store(self):
        try:
            with open(self.rag_store_path, 'w', encoding='utf-8') as f:
                json.dump(self.rag_items, f, ensure_ascii=False)
        except Exception as e:
            print(f"RAG save error: {e}")

    def add_ocr_to_rag(self, text: str, meta: dict = None):
        try:
            clean = (text or '').strip()
            if not clean:
                return
            embedding = self.embed_text(clean)
            item = {
                "id": f"ocr-{int(time.time()*1000)}",
                "text": clean,
                "embedding": embedding,
                "meta": meta or {"source": "ocr", "ts": time.time()}
            }
            self.rag_items.append(item)
            # keep store from growing indefinitely: cap to last 500 items
            if len(self.rag_items) > 500:
                self.rag_items = self.rag_items[-500:]
            self._save_rag_store()
        except Exception as e:
            print(f"RAG add error: {e}")

    def embed_text(self, text: str):
        try:
            if self.gemini_api_key:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_embed_model}:embedContent?key={self.gemini_api_key}"
                payload = {"content": {"parts": [{"text": text}]}}
                resp = requests.post(url, headers={"Content-Type": "application/json"}, data=json.dumps(payload), timeout=20)
                resp.raise_for_status()
                data = resp.json()
                vec = (((data or {}).get('embedding') or {}).get('values'))
                if isinstance(vec, list):
                    return vec
            # naive fallback embedding: hashed bag-of-words
            return self._naive_embed(text)
        except Exception as e:
            print(f"Embed error: {e}")
            return self._naive_embed(text)

    def _naive_embed(self, text: str):
        tokens = text.lower().split()
        # Fixed small-dim embedding via hashing
        dim = 128
        vec = [0.0]*dim
        for tok in tokens:
            h = abs(hash(tok)) % dim
            vec[h] += 1.0
        # L2 normalize
        norm = sum(v*v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]

    def retrieve_context(self, query: str, top_k: int = 5):
        try:
            if not self.rag_items:
                return []
            q_emb = self.embed_text(query)
            scored = []
            for it in self.rag_items:
                emb = it.get('embedding') or []
                score = self._cosine_sim(q_emb, emb)
                scored.append((score, it))
            scored.sort(key=lambda x: x[0], reverse=True)
            top = [it for _, it in scored[:top_k] if _ > 0]
            return top
        except Exception as e:
            print(f"RAG retrieve error: {e}")
            return []

    def _cosine_sim(self, a, b):
        if not a or not b:
            return 0.0
        n = min(len(a), len(b))
        dot = sum((a[i] or 0.0) * (b[i] or 0.0) for i in range(n))
        na = sum((a[i] or 0.0)**2 for i in range(n)) ** 0.5 or 1.0
        nb = sum((b[i] or 0.0)**2 for i in range(n)) ** 0.5 or 1.0
        return dot / (na * nb)

    def build_rag_prompt(self, question: str, contexts: list) -> str:
        # Prompt tailored for visually impaired users, emphasizing clarity and context use
        context_texts = [f"- {c.get('text','')[:1000]}" for c in (contexts or []) if c.get('text')]
        context_block = "\n".join(context_texts) if context_texts else "(no captured context)"
        instructions = (
            "You are Liya, a helpful voice assistant for visually impaired users. "
            "Answer clearly and concisely. Use the OCR context below only if it is relevant. "
            "If the context is not helpful, answer from your general knowledge instead of saying you cannot answer. "
            "When the question asks for examples or starts with 'what animals', 'which animals', or 'list animals', "
            "respond with ONLY a concise, comma-separated list of animal names and nothing else (no extra words, no punctuation except commas). "
            "IMPORTANT: Preserve any literal asterisks (*) from the content or your answer; do not remove or escape them. "
            f"Respond in at most {self.answer_word_limit} words unless a list of names is requested.\n\n"
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n"
        )
        return instructions

    def generate_ai_answer_hf(self, prompt: str) -> str:
        """Call Hugging Face Inference API to generate an answer."""
        try:
            if not self.hf_api_key:
                return ""
            url = f"https://api-inference.huggingface.co/models/{self.hf_model}"
            headers = {
                'Authorization': f'Bearer {self.hf_api_key}',
                'Content-Type': 'application/json'
            }
            # Rough token target from word limit
            max_new = max(32, min(256, int(self.answer_word_limit * 2)))
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": max_new,
                    "temperature": 0.4,
                    "return_full_text": False
                }
            }
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
            if resp.status_code == 503:
                # Model loading; return a friendly message
                return "The AI model is warming up. Please ask again in a moment."
            resp.raise_for_status()
            data = resp.json()
            # Response formats vary across models
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # text-generation format
                txt = data[0].get('generated_text') or data[0].get('summary_text')
                if isinstance(txt, str):
                    return txt.strip()
            if isinstance(data, dict):
                # Some models return dicts
                possible = data.get('generated_text') or data.get('summary_text')
                if isinstance(possible, str):
                    return possible.strip()
            # As last resort, stringify
            return (str(data)[:500] + '...') if data else ""
        except Exception as e:
            print(f"HF generate error: {e}")
            return ""

    def generate_ai_answer_gemini(self, prompt: str) -> str:
        """Call Google Gemini via Generative Language API to generate an answer."""
        try:
            if not self.gemini_api_key:
                return ""
            # API endpoint for text generation
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent?key={self.gemini_api_key}"
            # Configure generation with a token cap aligned to word limit
            max_tokens = max(64, min(512, int(self.answer_word_limit * 3)))
            payload = {
                "contents": [{"parts": [{"text": str(prompt)}]}],
                "generationConfig": {
                    "maxOutputTokens": max_tokens,
                    "temperature": 0.4,
                    "topP": 0.9,
                    "topK": 40
                }
            }
            headers = {"Content-Type": "application/json"}
            resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
            resp.raise_for_status()
            data = resp.json()
            # Parse candidates -> content.parts[].text
            candidates = data.get('candidates') or []
            if candidates:
                first = candidates[0] or {}
                content = first.get('content') or {}
                parts = content.get('parts') or []
                texts = [p.get('text', '') for p in parts if isinstance(p, dict)]
                combined = " ".join([t for t in texts if t]).strip()
                if combined:
                    return combined
            return ""
        except Exception as e:
            print(f"Gemini generate error: {e}")
            return ""
    
    def ocr_image(self, pil_image: Image.Image):
        """Perform OCR on a provided PIL image using OCR.space (if configured) with Tesseract fallback."""
        try:
            if self.ocr_api_key:
                text = self.ocr_with_api(pil_image)
                if text:
                    return text
            # Fallback to local OCR
            text = pytesseract.image_to_string(pil_image)
            return text.strip() if text and text.strip() else "No text detected in the image"
        except Exception as e:
            print(f"OCR error: {e}")
            return "Error processing image for text recognition"
    
    def ocr_with_api(self, image):
        """Perform OCR using OCR.space API"""
        try:
            # Convert PIL image to base64
            buffer = io.BytesIO()
            image.save(buffer, format='PNG')
            img_base64 = base64.b64encode(buffer.getvalue()).decode()
            
            url = 'https://api.ocr.space/parse/image'
            data = {
                'apikey': self.ocr_api_key,
                'base64Image': f'data:image/png;base64,{img_base64}',
                'language': 'eng',
                'isOverlayRequired': False
            }
            
            response = requests.post(url, data=data, timeout=10)
            result = response.json()
            
            if result.get('IsErroredOnProcessing', False):
                return None
            
            text = ""
            for parsed_result in result.get('ParsedResults', []):
                text += parsed_result.get('ParsedText', '')
            
            return text.strip() if text.strip() else None
            
        except Exception as e:
            print(f"OCR API error: {e}")
            return None

assistant = VoiceAssistant()

@app.route('/api/rag/clear', methods=['POST'])
def rag_clear():
    try:
        # Clear in-memory items and persist
        assistant.rag_items = []
        try:
            assistant._save_rag_store()
        except Exception as e:
            print(f"RAG save after clear error: {e}")
        return jsonify({'success': True, 'message': 'Memory cleared.'})
    except Exception as e:
        print(f"/api/rag/clear error: {e}")
        return jsonify({'success': False, 'message': 'Failed to clear memory.'}), 500

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

# Removed duplicate favicon route - now handled above with LUMA logo

@app.route('/api/ask', methods=['POST'])
def ask_question():
    data = request.get_json()
    question = data.get('question', '')
    
    if question:
        answer = assistant.answer_question(question)
        try:
            # If AI produced an empty string, surface a helpful error so the UI can speak it
            if not isinstance(answer, str) or not answer.strip():
                return jsonify({
                    'success': False,
                    'message': 'AI did not return a response. Please try again in a moment.',
                    'provider': assistant.ai_provider_info()
                })
        except Exception:
            pass
        return jsonify({'success': True, 'answer': answer, 'provider': assistant.ai_provider_info()})
    else:
        return jsonify({'success': False, 'message': 'No question provided'})



@app.route('/api/ocr', methods=['POST'])
def perform_ocr():
    """Accept a base64 data URL image from the client, run OCR, and return detected text."""
    try:
        data = request.get_json() or {}
        image_data_url = data.get('image_base64')
        if not image_data_url:
            return jsonify({'success': False, 'message': 'image_base64 is required'})

        # image_base64 can be in data URL format: data:image/png;base64,xxxx
        if ',' in image_data_url:
            image_base64 = image_data_url.split(',', 1)[1]
        else:
            image_base64 = image_data_url

        image_bytes = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        text = assistant.ocr_image(pil_image)
        if text and text.strip():
            try:
                assistant.add_ocr_to_rag(text, meta={
                    'source': 'ocr',
                    'ts': time.time(),
                    'length': len(text)
                })
            except Exception as e:
                print(f"RAG store from OCR error: {e}")
            return jsonify({'success': True, 'text': text})
        # No text extracted; provide a helpful message depending on available engines
        error_msg = 'No text detected in the image.'
        if not assistant.ocr_api_key:
            error_msg = 'OCR not available: Tesseract not installed and no OCR.space API key configured. Install Tesseract or set OCR_SPACE_API_KEY in .env.'
        return jsonify({'success': False, 'message': error_msg, 'text': ''})
    except Exception as e:
        print(f"/api/ocr error: {e}")
        return jsonify({'success': False, 'message': 'Error processing image'})

if __name__ == '__main__':
    print("Starting lightweight web version - browser handles mic and camera")
    app.run(debug=True, host='0.0.0.0', port=5001)

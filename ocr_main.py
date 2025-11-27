import pytesseract
import cv2
import os
import re
import json
from datetime import datetime
import numpy as np
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast
import requests
from rapidfuzz import fuzz


API_URL = "https://news-web-scraper-1.onrender.com/api/enews"
HEADERS = {"X-API-KEY": "capstone@2025"}


from keywords_and_weights import MINISTRY_KEYWORDS, PRIORITY_WEIGHTS


def send_to_api(payload):
    try:
        r = requests.post(API_URL, json=payload, headers=HEADERS, timeout=30)
        if r.status_code in [200, 201]:
            print(f"✅ Uploaded: {payload.get('image_name', 'Article')}")
            print("API Response:", r.text)
        else:
            print(f"⚠ API upload failed ({r.status_code}): {r.text}")
    except Exception as e:
        print(f"❌ Error sending to API: {e}")


def extract_keywords(text, lang="english"):
    try:
        from keybert import KeyBERT
        import string
        kw_model = KeyBERT(model='distilbert-base-multilingual-cased')
        cleaned = re.sub(r'\d+', '', text)
        cleaned = cleaned.translate(str.maketrans('', '', string.punctuation))
        keywords = kw_model.extract_keywords(cleaned, keyphrase_ngram_range=(1, 2), stop_words=None)
        filtered = [kw[0] for kw in keywords if not kw[0].isdigit() and len(kw[0]) > 2 and not re.match(r'^\d+$', kw[0])]
        return filtered[:10]
    except Exception as e:
        print("Keyword extraction error:", e)
        tokens = re.findall(r'\w+', text)
        freq = {}
        for token in tokens:
            if token.isdigit() or len(token) < 3:
                continue
            freq[token] = freq.get(token, 0) + 1
        return sorted(freq, key=freq.get, reverse=True)[:10]


# Optional: Rule-based strong negative override
VIOLENCE_KEYWORDS = [
    "blast", "explosion", "dead", "death", "killed", "injured", "riot", "shootout", "terror",
    "অগ্নিকাণ্ড", "নিহত", "বিস্ফোরণ", "মৃত",  # Bengali
    "धमाका", "मृत्यु", "हत्या", "दंगा",      # Hindi
    "ಸ್ಫೋಟ", "ಮರಣ", "ದಾಳಿ", "ಸಾವು"         # Kannada
]

def is_strongly_negative(text):
    tl = text.lower()
    return any(kw in tl for kw in VIOLENCE_KEYWORDS)


# Political-process neutralizer for elections/voting procedural news
POLITICS_NEUTRAL_WORDS = [
    "चुनाव", "मतदान", "उपराष्ट्रपति", "राष्ट्रपति", "सांसद",
    "उम्मीदवार", "लोकसभा", "राज्यसभा", "वोटिंग", "बहुमत",
    "election", "vote", "voting", "candidate", "parliament",
    "सदस्य", "उपचुनाव"
]

def is_political_process(text):
    tl = text.lower()
    return any(w.lower() in tl for w in POLITICS_NEUTRAL_WORDS)


class OCRProcessor:
    def __init__(self):
        self.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
        self.lang_map = {"hindi": "hin", "bengali": "ben", "kannada": "kan", "english": "eng"}
        self.mbart_codes = {"hindi": "hi_IN", "bengali": "bn_IN", "kannada": "kn_IN", "english": "en_XX"}

    def load_translation_models(self):
        tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        mbarmodel = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        return tokenizer, mbarmodel

    def load_sentiment_model(self):
        # Switched to lightweight, accurate, multilingual model
        try:
            return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
        except Exception as e:
            print("⚠️ Sentiment pipeline unavailable, fallback to rules.", e)
            return None

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        h, w = gray.shape
        if w < 1500:
            f = 1500 / w
            gray = cv2.resize(gray, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        gray = cv2.filter2D(gray, -1, kernel)
        return gray

    def ocr(self, img_path, lang_code):
        img = self.preprocess(img_path)
        configs = [
            f"--oem 3 --psm 6 -l {lang_code}",
            f"--oem 3 --psm 4 -l {lang_code}",
            f"--oem 3 --psm 3 -l {lang_code}"
        ]
        best_text = ""
        for cfg in configs:
            try:
                text = pytesseract.image_to_string(img, config=cfg)
                if len(text.strip()) > len(best_text.strip()):
                    best_text = text
            except:
                continue
        if len(best_text.strip()) < 20 and lang_code != "eng":
            for cfg in configs:
                try:
                    text = pytesseract.image_to_string(img, config=cfg.replace(f"-l {lang_code}", "-l eng"))
                    if len(text.strip()) > len(best_text.strip()):
                        best_text = text
                except:
                    continue
        return best_text

    def detect_language(self, text):
        h = len(re.findall(r"[\u0900-\u097F]", text))
        b = len(re.findall(r"[\u0980-\u09FF]", text))
        k = len(re.findall(r"[\u0C80-\u0CFF]", text))
        mx = max(h, b, k)
        if mx < 5:
            return "english"
        if h == mx:
            return "hindi"
        if b == mx:
            return "bengali"
        return "kannada"

    def translate_to_english(self, text, src_lang):
        if src_lang == "english":
            return text
        tokenizer, mbarmodel = self.load_translation_models()
        code = self.mbart_codes[src_lang]
        tokenizer.src_lang = code
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        ids = mbarmodel.generate(
            inputs.input_ids,
            forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
            num_beams=4,
            max_length=512,
            no_repeat_ngram_size=3
        )
        return tokenizer.decode(ids[0], skip_special_tokens=True)

    def classify_ministries(self, text):
        tl = text.lower()
        scores = {}
        hits = {}
        for m, levels in MINISTRY_KEYWORDS.items():
            sc = 0
            mk = []
            for prio, kws in levels.items():
                w = PRIORITY_WEIGHTS[prio]
                for kw in kws:
                    if fuzz.partial_ratio(kw.lower(), tl) > 80:
                        sc += w
                        mk.append(kw)
            scores[m] = sc
            hits[m] = mk
        thr = 5
        ministries = [m for m, s in scores.items() if s >= thr]
        if not ministries:
            ministries = ["General"]
            scores["General"] = 0
            hits["General"] = []
        ministries = sorted(ministries, key=lambda x: scores.get(x, 0), reverse=True)
        ministry = ministries[0] if ministries else "General"
        return ministry, scores, hits

    def extract_title_content(self, text):
        parts = re.split(r'[।\.!?|\n]+', text)
        segs = [p.strip() for p in parts if len(p.strip()) > 10]
        if not segs:
            return text.strip(), text.strip()
        title = segs[0].strip()
        content = " ".join([p.strip() for p in segs])
        return title, content

    def analyze_sentiment_nlp(self, raw_text, lang):
        sentiment_model = self.load_sentiment_model()
        # Rule-based override for strong violence/crisis detection
        if is_strongly_negative(raw_text):
            return "Negative", 0.20, "RULED_NEGATIVE"
        # Multilingual Bert model—no translation needed
        if sentiment_model:
            try:
                pred = sentiment_model(raw_text[:512])[0]
                label = pred['label']
                score = pred['score']
                # Map 5-star prediction to positive/neutral/negative
                if label in ["1 star", "2 stars"]:
                    sentiment_label = "Negative"
                elif label == "3 stars":
                    sentiment_label = "Neutral"
                else:
                    sentiment_label = "Positive"

                # Political-process neutralizer:
                # if model says Negative but article is routine politics/election, downgrade to Neutral
                if sentiment_label == "Negative" and is_political_process(raw_text) and score > 0.15:
                    sentiment_label = "Neutral"

                return sentiment_label, float(round(score, 2)), label
            except Exception as e:
                print(f"Sentiment error: {e}")
        return "Neutral", 0.0, None

    def process_image(self, img_path, edition_date=None, expected_lang=None):
        raw = self.ocr(img_path, self.lang_map.get(expected_lang, "eng"))
        if len(raw.strip()) < 15:
            raw = self.ocr(img_path, "eng")
        if len(raw.strip()) < 15:
            return None, {"error": "OCR failed", "image": img_path}
        lang = self.detect_language(raw)
        title_orig, content_orig = self.extract_title_content(raw)
        ministry, scores, keywords_hits = self.classify_ministries(content_orig)
        sentiment_label, sentiment_score, sentiment_raw = self.analyze_sentiment_nlp(content_orig, lang)
        keywords = extract_keywords(content_orig, lang)
        ts = datetime.now().isoformat() + "Z"
        output = {
            "image_name": os.path.basename(img_path),
            "extracted_text": content_orig,
            "language": lang.title(),
            "detected_category": [ministry],
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "keywords": keywords_hits[ministry][:10] if ministry in keywords_hits else [],
            "metadata": {
                "resolution": "Unknown",
                "processed_by": "Sarah Farooqui",
                "confidence_score": 0.88,
                "image_path": img_path
            }
        }
        if edition_date is not None:
            output["metadata"]["edition_date"] = edition_date
        return output, None


def main():
    processor = OCRProcessor()
    os.makedirs("outputs", exist_ok=True)
    tests = [
        ("data/hindi/hindi_sample.jpg", "2025-09-24", "hindi"),
        ("data/bengali/bengali_sample-1.jpg", "2025-09-24", "bengali"),
        ("data/kannada/kannada_sample-2.jpg", "2025-09-24", "kannada"),
    ]
    for img, ed, lang in tests:
        res, err = processor.process_image(img, ed, lang)
        fn = f"outputs/{lang}_updated.json"
        if res:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved {fn}")
            send_to_api(res)
        else:
            print(f"❌ Error: {err}")


if __name__ == "__main__":
    main()

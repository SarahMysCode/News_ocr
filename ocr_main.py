import pytesseract
import cv2
import os
import re
import json
import hashlib
from datetime import datetime
from rapidfuzz import fuzz
import numpy as np
from keywords_and_weights import MINISTRY_KEYWORDS, PRIORITY_WEIGHTS
from transformers import pipeline, MBartForConditionalGeneration, MBart50TokenizerFast

class OCRProcessor:
    def __init__(self):
        self.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd
        os.environ["TESSDATA_PREFIX"] = r"C:\Program Files\Tesseract-OCR\tessdata"
        self.lang_map = {"hindi":"hin", "bengali":"ben", "kannada":"kan", "english":"eng"}
        self.tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.mbarmodel = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
        self.mbart_codes = {"hindi":"hi_IN","bengali":"bn_IN","kannada":"kn_IN","english":"en_XX"}
        try:
            self.sentiment_model = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
        except Exception as e:
            print("⚠️ Sentiment pipeline unavailable, will fallback to rules.", e)
            self.sentiment_model = None

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 3)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        h, w = gray.shape
        if w < 1500:
            f = 1500 / w
            gray = cv2.resize(gray, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC)
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
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
        if mx < 5: return "english"
        if h == mx: return "hindi"
        if b == mx: return "bengali"
        return "kannada"

    def translate_to_english(self, text, src_lang):
        if src_lang == "english":
            return text
        code = self.mbart_codes[src_lang]
        self.tokenizer.src_lang = code
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        ids = self.mbarmodel.generate(inputs.input_ids, forced_bos_token_id=self.tokenizer.lang_code_to_id["en_XX"], num_beams=4, max_length=512, no_repeat_ngram_size=3)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)

    def classify_ministries(self, text):
        tl = text.lower(); scores = {}; hits = {}
        for m, levels in MINISTRY_KEYWORDS.items():
            sc=0; mk=[]
            for prio, kws in levels.items():
                w = PRIORITY_WEIGHTS[prio]
                for kw in kws:
                    if kw.lower() in tl:
                        sc += w; mk.append(kw)
            scores[m] = sc; hits[m] = mk
        thr = 5  # wider inclusion
        ministries = [m for m, s in scores.items() if s >= thr]
        if not ministries:
            ministries = ["General"]
            scores["General"] = 0
            hits["General"] = []
        ministries = sorted(ministries, key=lambda x: scores.get(x,0), reverse=True)
        # Return only the highest scoring ministry
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

    def analyze_sentiment_nlp(self, raw_text, orig_lang):
        text = raw_text
        if orig_lang != "english":
            try:
                text = self.translate_to_english(raw_text, orig_lang)
            except Exception as e:
                text = raw_text
        if self.sentiment_model:
            try:
                pred = self.sentiment_model(text[:512])[0]
                label = pred['label'].lower()
                score = pred['score']
                sentiment_label = "Positive" if "positive" in label else (
                    "Negative" if "negative" in label else "Neutral")
                return sentiment_label, float(round(score, 2))
            except Exception as e:
                pass
        return "Neutral", 0.0

    def process_image(self, img_path, edition_date=None, expected_lang=None):
        raw = self.ocr(img_path, self.lang_map.get(expected_lang, "eng"))
        if len(raw.strip()) < 15: raw = self.ocr(img_path, "eng")
        if len(raw.strip()) < 15: return None, {"error": "OCR failed", "image": img_path}

        lang = self.detect_language(raw)
        title_orig, content_orig = self.extract_title_content(raw)
        content_for_class = self.translate_to_english(content_orig, lang)
        ministry, scores, keywords = self.classify_ministries(content_for_class)
        sentiment_label, sentiment_score = self.analyze_sentiment_nlp(content_orig, lang)
        ts = datetime.now().isoformat() + "Z"

        # Assemble your new output to match the required JSON structure
        output = {
            "image_name": os.path.basename(img_path),
            "extracted_text": content_orig,
            "language": lang.title(),
            "detected_category": [ministry],
            "sentiment_score": sentiment_score,
            "sentiment_label": sentiment_label,
            "keywords": keywords[ministry][:10] if ministry in keywords else [],
            "metadata": {
                "resolution": "Unknown",  # You can set actual if you prefer
                "processed_by": "Your Name",
                "confidence_score": 0.88,  # You can set real confidence if calculated
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
        ("data/kannada/kannada_sample-1.jpg", "2025-09-24", "kannada"),
    ]
    for img, ed, lang in tests:
        res, err = processor.process_image(img, ed, lang)
        fn = f"outputs/{lang}_updated.json"
        if res:
            with open(fn, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False, indent=2)
            print(f"✅ Saved {fn}")
        else:
            print(f"❌ Error: {err}")

if __name__ == "__main__":
    main()

from pathlib import Path
import re
import io
import logging

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename

# Optional document readers. Install the packages listed below to enable them.
try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

try:
    from pdf2image import convert_from_bytes
except ImportError:
    convert_from_bytes = None


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MAX_UPLOAD_MB = 12
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024
MAX_TEXT_CHARS = 250_000
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "png", "jpg", "jpeg"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("legal_outcome_prediction")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


def load_artifact(filename):
    path = ARTIFACT_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"Required model artifact is missing: {path}. "
            "Place the trained files in an 'artifacts' folder beside app.py."
        )
    return joblib.load(path)


# Expected directory structure:
# project_folder/
#   app.py
#   artifacts/word_tfidf.joblib
#   artifacts/char_tfidf.joblib
#   artifacts/logistic_model.joblib
#   artifacts/retrieval_corpus.parquet
word = load_artifact("word_tfidf.joblib")
char = load_artifact("char_tfidf.joblib")
clf = load_artifact("logistic_model.joblib")
retrieval_path = ARTIFACT_DIR / "retrieval_corpus.parquet"
if not retrieval_path.exists():
    raise FileNotFoundError(
        f"Required retrieval corpus is missing: {retrieval_path}. "
        "Ensure retrieval_corpus.parquet is in the artifacts folder."
    )
# Stream the Parquet file instead of loading the full corpus into pandas.
# This reduces startup RAM usage on Render's 512 MiB instance.
import pyarrow.parquet as pq

retrieval_file = pq.ParquetFile(retrieval_path)
retrieval_columns = set(retrieval_file.schema.names)

if "model_text" not in retrieval_columns:
    raise ValueError("retrieval_corpus.parquet must contain a 'model_text' column.")
if "y" not in retrieval_columns:
    raise ValueError("retrieval_corpus.parquet must contain a 'y' outcome column.")

RETRIEVAL_BATCH_SIZE = 256
RETRIEVAL_COLUMNS = [
    column for column in ("model_text", "y", "district", "id")
    if column in retrieval_columns
]
logger.info(
    "Opened historical corpus (%s rows) for streamed retrieval",
    retrieval_file.metadata.num_rows,
)


def clean(text):
    """Normalize whitespace without translating or otherwise rewriting the text."""
    if text is None:
        return ""
    text = str(text).replace("\x00", " ")
    return re.sub(r"\s+", " ", text).strip()




# -----------------------------------------------------------------------------
# Input validation / fail-safe
# -----------------------------------------------------------------------------
# These are broad screening rules, not a legal understanding model. The system
# only supports Hindi bail-outcome prediction; it must refuse obvious noise and
# clearly unrelated inputs before calling the classifier.
LEGAL_TERMS = [
    # Hindi / common Hindi legal wording
    "जमानत", "जमानत याचिका", "अग्रिम जमानत", "नियमित जमानत", "अभियुक्त",
    "आरोपी", "आवेदक", "अभियोजन", "एफआईआर", "प्राथमिकी", "धारा", "न्यायालय",
    "अदालत", "माननीय", "दंड संहिता", "गिरफ्तारी", "हिरासत", "पुलिस",
    "मुकदमा", "आपराधिक", "अपराध", "जमानतदार", "रिमांड", "जमानत पर",
    # English terms can appear in mixed-language or OCR-extracted documents
    "bail", "applicant", "accused", "petitioner", "prosecution", "fir",
    "section", "criminal", "custody", "arrest", "anticipatory bail",
    "regular bail", "hon'ble", "honble court", "learned court", "police station",
]


def validate_case_text(text):
    """Return an error message or None. Never run the model on failed input."""
    text = clean(text)
    if not text:
        return "No text was provided. Paste a case description or upload a readable document."
    if len(text) < 150:
        return (
            "The text is too short for a meaningful screening (minimum 150 characters). "
            "Please provide the facts and arguments section of a bail-related case."
        )
    if len(text) > MAX_TEXT_CHARS:
        return f"The extracted text is too large. Please keep it below {MAX_TEXT_CHARS:,} characters."

    # Reject repeated-character or low-information input such as 'aaaaaaa...'.
    compact = re.sub(r"\s+", "", text)
    if len(compact) < 100:
        return "The text contains too little readable content. Please provide a fuller case description."
    if len(set(compact)) < 12:
        return "The text appears repetitive or unreadable. Please check the text or upload a clearer document."

    tokens = re.findall(r"[\w\u0900-\u097F]+", text, flags=re.UNICODE)
    if len(tokens) < 20:
        return "The text contains too few words. Please provide a fuller legal case description."

    lower_text = text.lower()
    legal_hits = sum(1 for term in LEGAL_TERMS if term.lower() in lower_text)
    devanagari_chars = len(re.findall(r"[\u0900-\u097F]", text))
    latin_chars = len(re.findall(r"[A-Za-z]", text))
    alphabetic_chars = devanagari_chars + latin_chars
    hindi_ratio = devanagari_chars / max(alphabetic_chars, 1)

    # Dataset/model is for Hindi bail cases. Accept Hindi-dominant legal content
    # or mixed-language text with explicit bail/legal indicators.
    if legal_hits == 0:
        return (
            "The text does not contain recognizable bail/legal terms. No prediction was made. "
            "This prototype is designed for Hindi bail-case facts and arguments, not general text or every type of court case."
        )
    if hindi_ratio < 0.08 and legal_hits < 2:
        return (
            "The text does not look sufficiently like the Hindi legal material this model was trained on. "
            "Please provide the Hindi facts and arguments from a bail-related case."
        )
    return None


# -----------------------------------------------------------------------------
# Text extraction
# -----------------------------------------------------------------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_pdf(data):
    if PdfReader is None:
        raise RuntimeError("PDF support is not installed. Install pypdf.")
    reader = PdfReader(io.BytesIO(data))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            pages.append("")
    text = clean("\n".join(pages))
    if len(text) >= 80:
        return text, "PDF text extraction"

    # Scanned PDF fallback: OCR needs pdf2image, Pillow, Tesseract, and Poppler.
    if convert_from_bytes is None or pytesseract is None:
        raise RuntimeError(
            "This PDF appears to be scanned and contains no selectable text. "
            "For OCR, install pdf2image and pytesseract, install Tesseract OCR and Poppler, "
            "then ensure Hindi language data (hin) is available."
        )
    try:
        # Convert and OCR one page at a time to reduce peak memory usage.
        ocr_pages = []
        page_count = min(len(reader.pages), 20)
        for page_number in range(1, page_count + 1):
            images = convert_from_bytes(
                data,
                dpi=180,
                first_page=page_number,
                last_page=page_number,
                fmt="jpeg",
                thread_count=1,
            )
            if not images:
                continue
            page_image = images[0]
            try:
                # Hindi plus English OCR; Tesseract needs hin.traineddata.
                ocr_pages.append(
                    pytesseract.image_to_string(page_image, lang="hin+eng")
                )
            finally:
                page_image.close()
                del page_image, images
        ocr_text = clean("\n".join(ocr_pages))
    except Exception as exc:
        logger.exception("PDF OCR failed")
        raise RuntimeError(
            "OCR could not read this scanned PDF. Check that Tesseract, Hindi OCR data, "
            "and Poppler are installed correctly."
        ) from exc
    if len(ocr_text) < 40:
        raise RuntimeError("No readable text was found in this PDF, even after OCR.")
    return ocr_text, "Scanned PDF OCR (first 20 pages maximum)"


def extract_document(file_storage):
    filename = secure_filename(file_storage.filename or "uploaded_document")
    if not filename or not allowed_file(filename):
        raise ValueError("Unsupported file type. Upload a PDF, DOCX, TXT, PNG, JPG, or JPEG file.")
    data = file_storage.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError(f"The file is too large. Maximum upload size is {MAX_UPLOAD_MB} MB.")

    extension = filename.rsplit(".", 1)[1].lower()
    if extension == "txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError:
            try:
                text = data.decode("utf-8", errors="replace")
            except Exception as exc:
                raise RuntimeError("The TXT file could not be decoded as text.") from exc
        method = "TXT text extraction"
    elif extension == "pdf":
        text, method = extract_pdf(data)
    elif extension == "docx":
        if Document is None:
            raise RuntimeError("DOCX support is not installed. Install python-docx.")
        try:
            document = Document(io.BytesIO(data))
            chunks = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
            for table in document.tables:
                for row in table.rows:
                    chunks.append(" | ".join(cell.text for cell in row.cells))
            text = "\n".join(chunks)
            method = "DOCX text extraction"
        except Exception as exc:
            raise RuntimeError("This DOCX file could not be read. Make sure it is a valid Word document.") from exc
    else:
        if Image is None or pytesseract is None:
            raise RuntimeError("Image OCR support is not installed. Install Pillow and pytesseract, plus Tesseract OCR with Hindi language data.")
        try:
            image = Image.open(io.BytesIO(data))
            text = pytesseract.image_to_string(image, lang="hin+eng")
            method = "Image OCR"
        except Exception as exc:
            logger.exception("Image OCR failed")
            raise RuntimeError("The image could not be processed. Check the file and OCR installation.") from exc

    text = clean(text)
    if not text:
        raise RuntimeError("No readable text could be extracted from this file.")
    if len(text) > MAX_TEXT_CHARS:
        raise RuntimeError(f"The extracted document is too long. Please use a document under {MAX_TEXT_CHARS:,} characters.")
    return text, method, filename


# -----------------------------------------------------------------------------
# Model inference
# -----------------------------------------------------------------------------
def vector(text):
    text = clean(text)
    word_features = word.transform([text])
    char_features = char.transform([text])
    return hstack([word_features, char_features]).tocsr()


def explain(text, topn=8):
    text = clean(text)
    v = word.transform([text])
    number_of_word_features = len(word.vocabulary_)
    coef = clf.coef_[0][:number_of_word_features]
    scores = v.multiply(coef).tocsr()
    inv = {index: term for term, index in word.vocabulary_.items()}
    pairs = [
        (inv.get(int(index), str(index)), float(score))
        for index, score in zip(scores.indices, scores.data)
    ]
    pairs.sort(key=lambda item: abs(item[1]), reverse=True)
    return pairs[:topn]


def predict_case(text):
    x = vector(text)
    probabilities = clf.predict_proba(x)[0]
    classes = list(clf.classes_)
    # Find probability associated with label 1 (Bail Granted), rather than
    # assuming class order in the saved classifier.
    grant_index = classes.index(1) if 1 in classes else None
    if grant_index is None:
        raise RuntimeError("The saved classifier does not contain the expected label 1.")
    probability = float(probabilities[grant_index])
    label = 1 if probability >= 0.5 else 0
    confidence = max(probability, 1.0 - probability)

    # Stream historical rows from Parquet in bounded batches. The full
    # DataFrame and full-corpus TF-IDF matrix are never created in memory.
    best_cases = []

    for record_batch in retrieval_file.iter_batches(
        batch_size=RETRIEVAL_BATCH_SIZE,
        columns=RETRIEVAL_COLUMNS,
    ):
        batch = record_batch.to_pandas()
        batch_text = batch["model_text"].fillna("").astype(str).map(clean)

        batch_vectors = hstack(
            [word.transform(batch_text), char.transform(batch_text)]
        ).tocsr()
        batch_scores = cosine_similarity(x, batch_vectors).ravel()

        if len(batch_scores):
            local_count = min(5, len(batch_scores))
            local_indices = np.argpartition(batch_scores, -local_count)[-local_count:]

            for local_index in local_indices:
                score = float(batch_scores[local_index])
                if score < 0.05:
                    continue

                row = batch.iloc[int(local_index)]
                case = {
                    "text": clean(row["model_text"]),
                    "y": int(row["y"]),
                    "similarity": score,
                }
                if "district" in batch.columns:
                    district = row["district"]
                    case["district"] = None if pd.isna(district) else str(district)
                if "id" in batch.columns:
                    case["id"] = row["id"]
                best_cases.append(case)

            best_cases.sort(key=lambda item: item["similarity"], reverse=True)
            del best_cases[5:]

        del batch, batch_text, batch_vectors, batch_scores, record_batch

    similar = pd.DataFrame(best_cases)

    # This is a conservative UI guard, not a calibrated uncertainty estimate.
    # Do not show a binary outcome when the two class probabilities are too close.
    uncertain = confidence < 0.60
    return {
        "label": label,
        "prob": round(probability * 100, 2),
        "conf": round(confidence * 100, 2),
        "uncertain": uncertain,
        "explain": explain(text),
        "similar": similar.to_dict("records"),
    }


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LegalLens | Hindi Bail Case Analysis</title>
<style>
:root{--bg:#f4f7fb;--panel:#fff;--ink:#172033;--muted:#68758a;--line:#e2e8f0;--navy:#172b4d;--blue:#315ee8;--blue2:#eaf0ff;--green:#087f5b;--red:#b42318;--amber:#9a6700;--shadow:0 12px 36px rgba(20,36,70,.07)}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Arial,sans-serif;line-height:1.55}
.topbar{background:var(--navy);color:#fff;padding:17px 5vw;display:flex;align-items:center;justify-content:space-between;gap:16px}.brand{font-size:18px;font-weight:800;letter-spacing:.2px}.brand span{color:#9fb7ff}.tag{font-size:12px;border:1px solid #50617e;padding:5px 10px;border-radius:999px;color:#e1e8f5}
.shell{max-width:1180px;margin:34px auto;padding:0 22px 50px}.hero{display:flex;justify-content:space-between;align-items:flex-start;gap:24px;margin-bottom:25px}.eyebrow{font-size:12px;font-weight:800;text-transform:uppercase;letter-spacing:1.4px;color:var(--blue)}h1{font-size:clamp(27px,4vw,38px);line-height:1.15;margin:8px 0 12px;letter-spacing:-.8px}.lead{max-width:690px;color:var(--muted);margin:0}.hero-note{min-width:205px;background:#eef3ff;border:1px solid #d9e3ff;padding:13px 15px;border-radius:14px;color:#344a7a;font-size:12px}
.layout{display:grid;grid-template-columns:minmax(0,1.15fr) minmax(280px,.85fr);gap:22px;align-items:start}.panel{background:var(--panel);border:1px solid var(--line);border-radius:18px;padding:23px;box-shadow:var(--shadow);margin-bottom:20px}.panel h2{font-size:18px;margin:0 0 5px;letter-spacing:-.2px}.sub{font-size:13px;color:var(--muted);margin:0 0 18px}.tabs{display:flex;gap:8px;background:#f1f4f9;padding:5px;border-radius:12px;margin-bottom:15px}.tab{flex:1;border:0;background:transparent;border-radius:9px;padding:10px 12px;font-weight:700;color:#647086;cursor:pointer}.tab.active{background:#fff;color:var(--navy);box-shadow:0 1px 5px #dfe5ef}.drop{border:1.5px dashed #b9c7df;background:#f8faff;border-radius:14px;padding:22px;text-align:center}.drop .glyph{font-size:28px}.drop strong{display:block;margin:5px 0}.drop p{font-size:12px;color:var(--muted);margin:0 0 12px}.file-input{max-width:100%;font-size:13px}.filename{font-size:12px;color:var(--green);margin-top:9px;overflow-wrap:anywhere}
textarea{width:100%;min-height:265px;resize:vertical;border:1px solid #d5deea;border-radius:12px;padding:14px;font:14px/1.7 inherit;outline:none;background:#fcfdff;color:var(--ink)}textarea:focus{border-color:#7d99f8;box-shadow:0 0 0 3px #eaf0ff}.actions{display:flex;flex-wrap:wrap;align-items:center;gap:10px;margin-top:15px}.btn{border:0;border-radius:11px;padding:12px 17px;font-weight:800;cursor:pointer;transition:.15s}.primary{background:var(--blue);color:#fff}.primary:hover{background:#244bd0;transform:translateY(-1px)}.secondary{background:#eef2f8;color:#35445e}.hint{font-size:12px;color:var(--muted);margin-top:12px}.notice{padding:13px 15px;border-radius:12px;font-size:13px;margin:0 0 18px}.notice.error{background:#fff0ee;border:1px solid #ffd2cc;color:#922018}.notice.ok{background:#ecfdf3;border:1px solid #b7ebcc;color:#17663f}.result-head{display:flex;justify-content:space-between;align-items:center;gap:10px}.pill{font-size:11px;font-weight:800;padding:5px 9px;border-radius:999px;background:#edf1f7;color:#475569}.outcome{font-size:26px;font-weight:900;letter-spacing:-.5px;margin:16px 0 5px}.outcome.granted{color:var(--green)}.outcome.denied{color:var(--red)}.metric-row{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin:17px 0}.metric{background:#f7f9fc;border:1px solid #e7edf5;border-radius:12px;padding:13px}.metric span{display:block;font-size:11px;color:var(--muted);margin-bottom:3px}.metric strong{font-size:20px}.bar{height:7px;border-radius:99px;background:#e7ebf3;overflow:hidden;margin-top:9px}.bar>i{display:block;height:100%;background:var(--blue);border-radius:99px}.case{padding:14px 0;border-top:1px solid var(--line)}.case:first-of-type{border-top:0}.case-title{display:flex;justify-content:space-between;gap:10px;font-size:13px;font-weight:800}.case p{font-size:12px;color:#536176;margin:8px 0 0;overflow-wrap:anywhere}.term-list{display:flex;flex-wrap:wrap;gap:8px;padding:0;list-style:none}.term-list li{font-size:12px;background:#f1f5ff;border:1px solid #dce6ff;border-radius:8px;padding:7px 9px}.term-list small{color:#596987}.empty{font-size:13px;color:var(--muted)}.footer{font-size:12px;color:#778398;text-align:center;margin-top:12px}.disclaimer{background:#fff8e7;border:1px solid #f5df9c;color:#745315;border-radius:13px;padding:13px 15px;font-size:12px;margin-top:18px}
@media(max-width:850px){.layout{grid-template-columns:1fr}.hero{display:block}.hero-note{margin-top:16px}.panel{padding:18px}.shell{margin-top:25px}}@media(max-width:480px){.topbar{padding:15px 20px}.tag{display:none}.shell{padding:0 13px 35px}.metric-row{grid-template-columns:1fr 1fr}.outcome{font-size:23px}}
</style>
</head>
<body>
<header class="topbar"><div class="brand">Legal<span>Lens</span> <span style="color:#dbe5ff;font-weight:500">/ Research Prototype</span></div><div class="tag">HINDI BAIL CASE ANALYSIS</div></header>
<main class="shell">
  <section class="hero"><div><div class="eyebrow">Legal text intelligence</div><h1>Analyze a bail case</h1><p class="lead">Submit the facts and arguments of a Hindi bail-related case to inspect a model-estimated historical outcome pattern and retrieve similar cases.</p></div><div class="hero-note"><strong>Designed for Hindi bail cases</strong><br>Not a general-purpose predictor for every court matter.</div></section>
  {% if error %}<div class="notice error" role="alert"><strong>No prediction was made.</strong><br>{{ error }}</div>{% endif %}
  {% if info %}<div class="notice ok">{{ info }}</div>{% endif %}
  <div class="layout">
    <section class="panel">
      <h2>Case input</h2><p class="sub">Paste text directly or extract text from a document.</p>
      <form method="post" enctype="multipart/form-data" id="caseForm">
        <div class="tabs"><button type="button" class="tab active" id="textTab" onclick="showTab('text')">Enter text</button><button type="button" class="tab" id="fileTab" onclick="showTab('file')">Upload document</button></div>
        <div id="textPane"><label for="caseText" class="sub" style="display:block;margin-bottom:8px">Hindi facts and arguments</label><textarea id="caseText" name="case" placeholder="यहाँ जमानत आवेदन से संबंधित तथ्य और तर्क लिखें या पेस्ट करें...">{{ text }}</textarea><div class="hint">For best results, use the facts-and-arguments section. Avoid including the final judgment or outcome.</div></div>
        <div id="filePane" style="display:none"><div class="drop"><div class="glyph">▤</div><strong>Upload a case document</strong><p>PDF, DOCX, TXT, PNG, JPG or JPEG · Maximum {{ max_upload_mb }} MB</p><input class="file-input" type="file" id="document" name="document" accept=".pdf,.docx,.txt,.png,.jpg,.jpeg"><div class="filename" id="fileName"></div></div><div class="hint">Selectable PDFs and Word files are read directly. Scanned PDFs and images require Tesseract OCR with Hindi language data. After extraction, the text is validated before prediction.</div></div>
        <div class="actions"><button class="btn primary" type="submit" name="action" value="predict" id="submitBtn">Validate and analyze</button><button class="btn secondary" type="button" onclick="clearInput()">Clear</button></div>
      </form>
      {% if extracted %}<div style="margin-top:20px"><h2 style="font-size:15px">Extracted text preview</h2><p class="sub">Source: {{ source_method }}{% if source_filename %} · {{ source_filename }}{% endif %}</p><textarea readonly style="min-height:145px">{{ extracted }}</textarea></div>{% endif %}
    </section>
    <aside>
      {% if result %}
      <section class="panel">
        <div class="result-head"><h2>Model result</h2><span class="pill">SCREENED INPUT</span></div>
        {% if result.uncertain %}
          <div class="notice error" style="margin-top:16px;margin-bottom:10px"><strong>Uncertain result — do not rely on a binary outcome.</strong><br>The model's class probabilities are too close under this prototype's conservative display rule. Review the input and consult a qualified legal professional.</div>
        {% else %}
          <div class="outcome {{ 'granted' if result.label == 1 else 'denied' }}">{{ 'BAIL GRANTED' if result.label == 1 else 'BAIL DENIED' }}</div>
          <p class="sub" style="margin-bottom:0">Estimated class from historical text patterns; not a legal conclusion.</p>
        {% endif %}
        <div class="metric-row"><div class="metric"><span>Estimated probability of grant</span><strong>{{ result.prob }}%</strong><div class="bar"><i style="width:{{ result.prob }}%"></i></div></div><div class="metric"><span>Top-class probability*</span><strong>{{ result.conf }}%</strong><div class="bar"><i style="width:{{ result.conf }}%"></i></div></div></div>
        <p class="hint">*This is the classifier's probability, not a calibrated measure of legal certainty.</p>
      </section>
      <section class="panel"><h2>Influential terms</h2><p class="sub">Terms with larger absolute feature contributions influenced this model output. They are not legal reasoning.</p>{% if result.explain %}<ul class="term-list">{% for term, score in result.explain %}<li>{{ term }} <small>({{ '%.3f'|format(score) }})</small></li>{% endfor %}</ul>{% else %}<p class="empty">No influential word features were available for this input.</p>{% endif %}</section>
      <section class="panel"><h2>Similar historical cases</h2><p class="sub">Similarity is lexical and does not prove that cases are legally equivalent.</p>{% if result.similar %}{% for r in result.similar %}<div class="case"><div class="case-title"><span>{{ r.district if r.district else 'District not available' }}</span><span class="pill">{{ 'Granted' if r.y == 1 else 'Denied' }}{% if r.similarity is defined %} · {{ '%.0f'|format(r.similarity * 100) }}% match{% endif %}</span></div><p>{{ (r.text or '')[:650] }}{% if (r.text or '')|length > 650 %}…{% endif %}</p></div>{% endfor %}{% else %}<p class="empty">No sufficiently similar historical cases were found. Do not infer that the prediction is reliable from the absence of matches.</p>{% endif %}</section>
      {% else %}
      <section class="panel"><h2>What this tool checks</h2><p class="sub">The application runs validation before the classifier.</p><div class="case"><div class="case-title"><span>01 · Readability</span><span class="pill">Required</span></div><p>Rejects empty, too-short, repetitive or unreadable text.</p></div><div class="case"><div class="case-title"><span>02 · Legal relevance</span><span class="pill">Required</span></div><p>Looks for bail/legal indicators and text resembling the Hindi legal domain.</p></div><div class="case"><div class="case-title"><span>03 · Prediction screening</span><span class="pill">Cautious</span></div><p>Very close class probabilities are flagged as uncertain. This does not guarantee that other predictions are correct.</p></div></section>
      <section class="panel"><h2>Supported documents</h2><p class="sub">Text extraction is performed locally by the Flask app.</p><ul style="padding-left:19px;font-size:13px;color:#536176"><li>PDF with selectable text</li><li>DOCX and TXT files</li><li>Scanned PDF or image via OCR when dependencies are installed</li></ul></section>
      {% endif %}
    </aside>
  </div>
  <div class="disclaimer"><strong>Research prototype only.</strong> This system estimates patterns from historical Hindi bail data. It does not verify current law, evaluate all case facts, or replace a lawyer or judge. Do not use its output as the basis for a legal decision.</div>
  <div class="footer">LegalLens · Hindi bail-outcome prediction research prototype</div>
</main>
<script>
function showTab(which){const text=which==='text';document.getElementById('textPane').style.display=text?'block':'none';document.getElementById('filePane').style.display=text?'none':'block';document.getElementById('textTab').classList.toggle('active',text);document.getElementById('fileTab').classList.toggle('active',!text);document.getElementById('caseText').required=text;document.getElementById('document').required=!text;}
document.getElementById('document').addEventListener('change',function(){document.getElementById('fileName').textContent=this.files.length?this.files[0].name:'';});
function clearInput(){document.getElementById('caseText').value='';document.getElementById('document').value='';document.getElementById('fileName').textContent='';showTab('text');}
// If a file was selected, ensure the form uses the upload branch. The server
// accepts either pasted text or a selected document.
document.getElementById('caseForm').addEventListener('submit',function(){const file=document.getElementById('document').files.length>0;if(file){document.getElementById('caseText').required=false;}else{document.getElementById('caseText').required=false;}});
</script>
</body>
</html>"""


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def home():
    text = ""
    extracted = ""
    source_method = ""
    source_filename = ""
    result = None
    error = None
    info = None

    if request.method == "POST":
        pasted_text = clean(request.form.get("case", ""))
        uploaded = request.files.get("document")

        # If a document is supplied, it is the active input. Otherwise use text.
        if uploaded and uploaded.filename:
            try:
                extracted, source_method, source_filename = extract_document(uploaded)
                text = extracted
                validation_error = validate_case_text(text)
                if validation_error:
                    error = validation_error
                else:
                    result = predict_case(text)
                    info = "Document text was extracted and passed the initial input checks."
            except (ValueError, RuntimeError) as exc:
                error = str(exc)
            except Exception:
                logger.exception("Unexpected document processing error")
                error = "The document could not be processed because of an unexpected error. Check the file format and application logs."
        else:
            text = pasted_text
            validation_error = validate_case_text(text)
            if validation_error:
                error = validation_error
            else:
                try:
                    result = predict_case(text)
                except Exception:
                    logger.exception("Prediction failed")
                    error = "The model could not process this input. No prediction was returned. Check that all trained artifacts are compatible."

    return render_template_string(
        HTML,
        text=text,
        extracted=extracted,
        source_method=source_method,
        source_filename=source_filename,
        result=result,
        error=error,
        info=info,
        max_upload_mb=MAX_UPLOAD_MB,
    )


@app.errorhandler(413)
def too_large(_error):
    return render_template_string(
        HTML,
        text="",
        extracted="",
        source_method="",
        source_filename="",
        result=None,
        error=f"The upload exceeds the {MAX_UPLOAD_MB} MB limit. Please choose a smaller file.",
        info=None,
        max_upload_mb=MAX_UPLOAD_MB,
    ), 413


if __name__ == "__main__":
    # Keep debug disabled unless you are actively debugging on your own machine.
    app.run(host="127.0.0.1", port=5000, debug=False)

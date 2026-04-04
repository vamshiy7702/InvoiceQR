import streamlit as st
import os
import base64
import json
import tempfile
import cv2
import numpy as np
import jwt
import Levenshtein
import pandas as pd
import fitz  # pymupdf — no poppler needed
from groq import Groq
from qreader import QReader
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="InvoiceIQ — Smart Invoice Extractor",
    page_icon="🧾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────────────
# STYLING
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;900&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; }

html, body, [class*="css"], .stApp {
    font-family: 'Outfit', sans-serif;
    background: #f7f5f0;
    color: #1a1a1a;
}

.topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.4rem 0 1rem;
    border-bottom: 2px solid #1a1a1a;
    margin-bottom: 2.5rem;
}
.topbar-logo { font-size: 1.5rem; font-weight: 900; letter-spacing: -0.5px; color: #1a1a1a; }
.topbar-logo span { color: #d4522a; }
.topbar-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
    border: 1px solid #ccc;
    padding: 4px 10px;
    border-radius: 20px;
}

div[data-testid="stFileUploader"] {
    background: #fff;
    border: 2px dashed #ccc;
    border-radius: 16px;
    padding: 1.5rem;
}
div[data-testid="stFileUploader"]:hover { border-color: #d4522a; }

.stButton > button {
    background: #d4522a !important;
    color: #fff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

.stDownloadButton > button {
    background: transparent !important;
    color: #d4522a !important;
    border: 1.5px solid #d4522a !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    border-radius: 8px !important;
    width: 100% !important;
}

.stat-row { display: flex; gap: 1rem; margin-bottom: 1.8rem; }
.stat-card {
    flex: 1; background: #fff; border: 1.5px solid #e8e4dc;
    border-radius: 12px; padding: 1rem 1.2rem;
}
.stat-num { font-size: 2rem; font-weight: 900; color: #d4522a; line-height: 1; }
.stat-label {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    text-transform: uppercase; letter-spacing: 0.1em; color: #999; margin-top: 0.2rem;
}

.group-card {
    background: #fff; border: 1.5px solid #e8e4dc;
    border-radius: 14px; padding: 1.4rem 1.6rem; margin-bottom: 1.2rem;
}
.group-title {
    font-size: 0.68rem; font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase; letter-spacing: 0.14em;
    color: #d4522a; margin-bottom: 1rem; font-weight: 500;
}
.field-row {
    display: flex; justify-content: space-between; align-items: flex-start;
    padding: 0.5rem 0; border-bottom: 1px solid #f0ece4; gap: 1rem;
}
.field-row:last-child { border-bottom: none; }
.field-key {
    font-family: 'JetBrains Mono', monospace; font-size: 0.75rem;
    color: #999; min-width: 180px; flex-shrink: 0;
}
.field-val { font-size: 0.88rem; font-weight: 600; color: #1a1a1a; text-align: right; word-break: break-all; }
.field-val.empty { color: #ddd; font-weight: 400; font-style: italic; font-size: 0.82rem; }

.info-card {
    background: #1a1a1a; border-radius: 14px;
    padding: 1.6rem; color: #f7f5f0; margin-top: 1.5rem;
}
.info-card h4 {
    font-size: 0.7rem; font-family: 'JetBrains Mono', monospace;
    text-transform: uppercase; letter-spacing: 0.14em;
    color: #d4522a; margin-bottom: 1rem;
}
.info-item {
    display: flex; gap: 0.7rem; align-items: flex-start;
    margin-bottom: 0.7rem; font-size: 0.84rem; color: #ccc; line-height: 1.5;
}
.info-dot { color: #d4522a; font-size: 1rem; margin-top: 1px; flex-shrink: 0; }

.empty-state {
    text-align: center; padding: 4rem 2rem; background: #fff;
    border: 1.5px solid #e8e4dc; border-radius: 14px;
}
.empty-icon { font-size: 3.5rem; margin-bottom: 1rem; }
.empty-text { font-family: 'JetBrains Mono', monospace; font-size: 0.78rem; color: #ccc; }

.success-bar {
    background: #edf7ed; border: 1.5px solid #6abf69; border-radius: 10px;
    padding: 0.75rem 1.2rem; font-size: 0.88rem; color: #2e7d32;
    font-weight: 600; margin-bottom: 1.5rem;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; max-width: 1200px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def pdf_to_png(pdf_path: str, output_png: str, dpi: int = 200) -> str:
    """
    Render the first page of a PDF to PNG using PyMuPDF (fitz).
    ✅ No Poppler required — pure Python, works on all platforms.
    """
    doc = fitz.open(pdf_path)
    page = doc[0]
    # Scale matrix: PDF default is 72 dpi, multiply to get target dpi
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    pix.save(output_png)
    doc.close()
    return output_png


def decode_qr(png_path: str) -> dict:
    """
    Detect and decode QR code from image.
    Uses QReader (deep-learning based) — more robust than pyzbar on real invoices.
    """
    qreader = QReader()
    image = cv2.cvtColor(cv2.imread(png_path), cv2.COLOR_BGR2RGB)
    decoded_texts = qreader.detect_and_decode(image=image)

    if not decoded_texts or decoded_texts[0] is None:
        raise ValueError(
            "No QR code detected in the invoice. "
            "Ensure the PDF contains an embedded GST QR code."
        )

    payload = jwt.decode(decoded_texts[0], options={"verify_signature": False})
    return eval(payload["data"])


def extract_with_groq(png_path: str) -> dict:
    """Send invoice image to Groq Llama 4 Vision and return extracted fields as dict."""
    with open(png_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    prompt = """
You are an expert invoice data extraction system.

TASK:
Extract invoice details from the provided image and return ONLY valid JSON.
Do not add explanations. Do not add extra text.

STRICT EXTRACTION RULES:

1. IRN Rules:
   - IRN must be exactly 64 characters.
   - If IRN contains "-", remove all "-" characters.
   - If IRN appears in multiple lines, concatenate them.
   - If IRN length is more than 64 after cleanup, truncate to first 64 characters.
   - If IRN length is less than 64, return None.

2. GSTIN Rules:
   - hiib_gstin = buyer GSTIN (Hyundai India Insurance Broking Pvt Ltd — HIIB)
   - dealer_gstin = seller/service provider GSTIN
   - Both must be different. GSTIN = 15 characters.

3. Date Format: DD-MM-YYYY for invoice_date and ack_date.

4. Null Handling: Return None for any missing field. Do NOT guess or hallucinate.

5. Multi-line strings broken with "-" must be concatenated without the hyphen.

6. Numeric fields: digits only, no ₹ or commas.

RETURN FORMAT (STRICT JSON ONLY):
{
    "irn": "",
    "ack_no": "",
    "ack_date": "",
    "invoice_no": "",
    "invoice_date": "",
    "taxable_value": "",
    "cgst_amount": "",
    "sgst_utgst_amount": "",
    "igst_amount": "",
    "total_invoice_value": "",
    "dealer_code": "",
    "hiib_misp_code": "",
    "account_holders_name": "",
    "bank_name": "",
    "account_no": "",
    "branch": "",
    "bank_ifsc": "",
    "micr_code": "",
    "hiib_gstin": "",
    "dealer_gstin": "",
    "hiib_pincode": "",
    "dealer_pincode": "",
    "hiib_state_code": "",
    "dealer_state_code": "",
    "msme_code": "",
    "dealer_pan": "",
    "sac": "",
    "consigner_details": "",
    "consigner_address": "",
    "consigner_pincode": "",
    "buyer_name": "",
    "buyer_address": "",
    "buyer_pincode": "",
    "consigner_place_of_supply": "",
    "buyer_place_of_supply": "",
    "description_of_service": "",
    "oem": "",
    "quantity": "",
    "period_of_service": ""
}
Return only JSON.
"""
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    completion = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        }],
        temperature=1,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content)


def cross_verify(res: dict, qr: dict) -> dict:
    """
    Use Levenshtein distance to auto-correct IRN and GSTINs
    against the ground-truth values decoded from the QR code.
    """
    checks = {
        "irn":          ("Irn",          4),
        "hiib_gstin":   ("BuyerGstin",   2),
        "dealer_gstin": ("SellerGstin",  2),
    }
    for field, (qr_key, threshold) in checks.items():
        ocr_val = str(res.get(field) or "")
        qr_val  = str(qr.get(qr_key, ""))
        if qr_val and Levenshtein.distance(ocr_val, qr_val) <= threshold:
            res[field] = qr_val
    return res


def generate_details(pdf_bytes: bytes, filename: str) -> dict:
    """Full end-to-end pipeline: raw PDF bytes → verified extracted dict."""
    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = os.path.join(tmp, filename)
        png_path = os.path.join(tmp, "page1.png")

        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)

        pdf_to_png(pdf_path, png_path)       # Step 1: PDF → image (PyMuPDF)
        qr_data = decode_qr(png_path)        # Step 2: QR decode
        result  = extract_with_groq(png_path) # Step 3: AI extraction
        result  = cross_verify(result, qr_data) # Step 4: QR cross-verify

    return result


# ─────────────────────────────────────────────────────────────────────────────
# FIELD GROUPS
# ─────────────────────────────────────────────────────────────────────────────
GROUPS = {
    "Identity & Reference": {
        "icon": "🔑",
        "fields": ["irn", "ack_no", "ack_date", "invoice_no", "invoice_date"],
    },
    "Financial Breakdown": {
        "icon": "💰",
        "fields": ["taxable_value", "cgst_amount", "sgst_utgst_amount", "igst_amount", "total_invoice_value"],
    },
    "Party & GSTIN Details": {
        "icon": "🏢",
        "fields": ["hiib_gstin", "dealer_gstin", "hiib_pincode", "dealer_pincode",
                   "hiib_state_code", "dealer_state_code", "dealer_code",
                   "hiib_misp_code", "dealer_pan", "msme_code", "sac"],
    },
    "Bank Details": {
        "icon": "🏦",
        "fields": ["account_holders_name", "bank_name", "account_no", "branch", "bank_ifsc", "micr_code"],
    },
    "Consigner & Buyer": {
        "icon": "📦",
        "fields": ["consigner_details", "consigner_address", "consigner_pincode",
                   "consigner_place_of_supply", "buyer_name", "buyer_address",
                   "buyer_pincode", "buyer_place_of_supply"],
    },
    "Service Information": {
        "icon": "📋",
        "fields": ["description_of_service", "oem", "quantity", "period_of_service"],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# UI HELPER
# ─────────────────────────────────────────────────────────────────────────────
def render_group(title: str, icon: str, fields: list, data: dict):
    rows = ""
    for key in fields:
        val = data.get(key)
        display = str(val) if val and str(val).lower() not in ("none", "", "null") else None
        if display:
            rows += f'<div class="field-row"><span class="field-key">{key}</span><span class="field-val">{display}</span></div>'
        else:
            rows += f'<div class="field-row"><span class="field-key">{key}</span><span class="field-val empty">not found</span></div>'
    st.markdown(f'<div class="group-card"><div class="group-title">{icon} &nbsp;{title}</div>{rows}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">Invoice<span>IQ</span></div>
    <div class="topbar-tag">GST · QR-Verified · AI-Powered</div>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns([1, 1.7], gap="large")

# ── LEFT ─────────────────────────────────────────────────────────────────────
with col_left:
    uploaded = st.file_uploader(
        "Upload Invoice PDF",
        type=["pdf"],
        help="Upload a GST invoice PDF with an embedded QR code",
    )

    if uploaded:
        st.markdown(f"""
        <div style="margin-top:0.8rem; padding:0.8rem 1rem; background:#fff3f0;
                    border:1px solid #f5c6b8; border-radius:10px;">
            <b style="color:#d4522a;">📄 {uploaded.name}</b><br>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.7rem; color:#aaa;">
                {uploaded.size / 1024:.1f} KB
            </span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    extract_clicked = st.button("⚡ Extract Invoice Data", disabled=not uploaded)

    st.markdown("""
    <div class="info-card">
        <h4>How it works</h4>
        <div class="info-item"><span class="info-dot">→</span>
            <span>Upload your invoice PDF</span></div>
        <div class="info-item"><span class="info-dot">→</span>
            <span>System scans and reads invoice content</span></div>
        <div class="info-item"><span class="info-dot">→</span>
            <span>QR code is verified for authenticity</span></div>
        <div class="info-item"><span class="info-dot">→</span>
            <span>Key details are extracted automatically</span></div>
        <div class="info-item"><span class="info-dot">→</span>
            <span>Results are validated and structured</span></div>
        <div class="info-item"><span class="info-dot">→</span>
            <span>Download as JSON or CSV</span></div>
    </div>
    """, unsafe_allow_html=True)

# ── RIGHT ────────────────────────────────────────────────────────────────────
with col_right:

    if extract_clicked and uploaded:
        with st.spinner("Analysing invoice — this usually takes ~20 seconds..."):
            try:
                data = generate_details(uploaded.read(), uploaded.name)
                st.session_state["result"] = data
                st.session_state["fname"]  = uploaded.name
            except Exception as e:
                st.error(f"❌ Extraction failed: {e}")
                st.session_state.pop("result", None)

    if "result" in st.session_state:
        data  = st.session_state["result"]
        fname = st.session_state.get("fname", "invoice")

        filled = sum(1 for v in data.values() if v and str(v).lower() not in ("none", "", "null"))
        total  = len(data)

        st.markdown(f"""
        <div class="success-bar">
            ✅ &nbsp; Extraction complete — <b>{filled} of {total} fields</b> found
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-row">
            <div class="stat-card"><div class="stat-num">{filled}</div><div class="stat-label">Fields Extracted</div></div>
            <div class="stat-card"><div class="stat-num">{total - filled}</div><div class="stat-label">Not Found</div></div>
            <div class="stat-card"><div class="stat-num">{round(filled/total*100)}%</div><div class="stat-label">Fill Rate</div></div>
        </div>
        """, unsafe_allow_html=True)

        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇ Download JSON",
                data=json.dumps(data, indent=2, ensure_ascii=False),
                file_name=f"{fname[:-4]}_extracted.json",
                mime="application/json",
            )
        with dl2:
            df = pd.DataFrame(list(data.items()), columns=["Field", "Value"])
            st.download_button(
                "⬇ Download CSV",
                data=df.to_csv(index=False),
                file_name=f"{fname[:-4]}_extracted.csv",
                mime="text/csv",
            )

        st.markdown("<br>", unsafe_allow_html=True)
        for group_title, meta in GROUPS.items():
            render_group(group_title, meta["icon"], meta["fields"], data)

    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">🧾</div>
            <div style="font-size:1rem; font-weight:600; color:#aaa; margin-bottom:0.5rem;">No results yet</div>
            <div class="empty-text">Upload a PDF and click Extract to begin</div>
        </div>
        """, unsafe_allow_html=True)





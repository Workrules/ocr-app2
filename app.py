# dummy line
import streamlit as st
import pypdfium2 as pdfium
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os
import io
from PIL import Image, ImageFile
import cv2
import numpy as np
import json
from openai import OpenAI
import re
import difflib
from itertools import zip_longest

# ==== 環境変数 ====
AZURE_ENDPOINT = os.getenv("AZURE_DOCINT_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_DOCINT_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OCR_GPT_MODEL", "gpt-5")  # 必要なら環境変数で切替

# ==== 事前チェック ====
if not AZURE_ENDPOINT or not AZURE_KEY:
    st.error("環境変数 AZURE_DOCINT_ENDPOINT と AZURE_DOCINT_KEY を設定してください。")
    st.stop()

# ==== クライアント初期化 ====
client = DocumentAnalysisClient(endpoint=AZURE_ENDPOINT, credential=AzureKeyCredential(AZURE_KEY))
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ==== ファイルパス ====
DICT_FILE = "ocr_char_corrections.json"
UNTRAINED_FILE = "untrained_confusions.json"

JP_CHAR_RE = re.compile(r"^[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]$")

# ==== ユーティリティ ====
def remove_red_stamp(img_pil: Image.Image) -> Image.Image:
    img = np.array(img_pil)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    img[mask > 0] = [255, 255, 255]
    return Image.fromarray(img)

def load_json(path: str) -> dict:
    return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else {}

def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def learn_charwise_with_missing(original: str, corrected: str) -> dict:
    learned = {}
    sm = difflib.SequenceMatcher(None, original, corrected)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ["replace", "insert"]:
            o_seg = original[i1:i2]
            c_seg = corrected[j1:j2]
            for o_char, c_char in zip_longest(o_seg, c_seg, fillvalue=""):
                if c_char and (not o_char or o_char != c_char):
                    wrong = o_char if o_char else "□"
                    if JP_CHAR_RE.match(c_char) or c_char == "□":
                        learned[wrong] = {"right": c_char, "count": 1}
    return learned

def update_dictionary_and_untrained(learned: dict):
    dictionary = load_json(DICT_FILE)
    for w, meta in learned.items():
        if w in dictionary:
            if dictionary[w]["right"] == meta["right"]:
                dictionary[w]["count"] += meta["count"]
            else:
                if meta["count"] > dictionary[w]["count"]:
                    dictionary[w] = meta
        else:
            dictionary[w] = meta
    save_json(dictionary, DICT_FILE)

    untrained = load_json(UNTRAINED_FILE)
    for w, meta in learned.items():
        untrained[w] = meta["right"]
    save_json(untrained, UNTRAINED_FILE)

def gpt_fix_text(text: str, dictionary: dict) -> str:
    prompt = f"""
次のOCR結果を自然な日本語に直してください。
- 日本語に存在しない文字は「□」にしてください。
- 辞書候補を参考にしてください: {json.dumps(dictionary, ensure_ascii=False)}
- 意味を勝手に補完せず、最小限の修正だけ行ってください。

OCR結果:
{text}
""".strip()
    try:
        # GPT-5系：Responses API。temperatureは未サポートのため指定しない。
        resp = openai_client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            text={"verbosity": "low"},
            reasoning={"effort": "minimal"}
        )
        # 出力テキスト抽出（将来のスキーマ変化に強め）
        out_parts = []
        output = getattr(resp, "output", None)
        if output is None and hasattr(resp, "output_text"):
            return (resp.output_text or "").strip() or text
        for item in output or []:
            content = getattr(item, "content", []) or []
            for part in content:
                t = getattr(part, "text", None)
                if t:
                    out_parts.append(t)
        out = "".join(out_parts).strip()
        return out or text
    except Exception as e:
        st.warning(f"GPT補正をスキップしました（エラー）：{e}")
        return text

def render_pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 200) -> list[Image.Image]:
    imgs = []
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    scale = dpi / 72.0
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=scale).to_pil()
        imgs.append(pil.convert("RGB"))
    return imgs

def is_pdf(b: bytes) -> bool:
    return len(b) >= 5 and b[:5] == b"%PDF-"

# ==== UI ====
st.title("📄 Document Intelligence OCR - GPT＋印影除去＋欠落補正（複数ページ対応）")

dictionary = load_json(DICT_FILE)
st.sidebar.subheader("📖 現在の辞書")
st.sidebar.json(dictionary)
st.sidebar.markdown("### 🔧 診断")
st.sidebar.write({
    "AZURE_DOCINT_ENDPOINT_set": bool(AZURE_ENDPOINT),
    "AZURE_DOCINT_KEY_set": bool(AZURE_KEY),
    "OPENAI_API_KEY_set": bool(OPENAI_API_KEY),
    "MODEL": MODEL_NAME,
})

uploaded_file = st.file_uploader("画像またはPDFをアップロードしてください", type=["jpg", "jpeg", "png", "pdf"])
if not uploaded_file:
    st.info("📂 ここにファイルをアップロードしてください")
    st.stop()

file_bytes = uploaded_file.read()
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ==== ファイル読込（堅牢化） ====
try:
    if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf") or is_pdf(file_bytes):
        pages = render_pdf_bytes_to_images(file_bytes, dpi=200)
    else:
        try:
            pages = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
        except Exception:
            arr = np.frombuffer(file_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("画像の読み込みに失敗しました（JPG/PNG/PDFのみ対応）。")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pages = [Image.fromarray(img)]
except Exception as e:
    st.exception(e)
    st.stop()

if not pages:
    st.error("ページを生成できませんでした。ファイル形式をご確認ください。")
    st.stop()

# ==== ページ範囲選択 ====
total_pages = len(pages)
if total_pages > 1:
    start, end = st.slider(
        "処理するページ範囲を選択（1始まり）",
        min_value=1, max_value=total_pages, value=(1, min(total_pages, 5))
    )
    proc_range = range(start - 1, end)
else:
    st.info("このファイルは1ページです。")
    proc_range = range(0, 1)

# ==== メイン処理 ====
all_corrected: list[str] = []

for page_index in proc_range:
    page_img = pages[page_index]
    page_num = page_index + 1
    st.write(f"## ページ {page_num}")

    clean_img = remove_red_stamp(page_img)

    buf = io.BytesIO()
    clean_img.save(buf, format="PNG")
    buf.seek(0)

    with st.spinner("OCRを実行中..."):
        try:
            poller = client.begin_analyze_document("prebuilt-read", document=buf)
            result = poller.result()
        except Exception as e:
            st.exception(e)
            st.stop()

    doc_page = result.pages[0] if getattr(result, "pages", None) else None
    if not doc_page:
        st.warning("OCR結果にページが見つかりませんでした。")
        continue

    default_text = "\n".join([line.content for line in doc_page.lines])

    gpt_checked_text = gpt_fix_text(default_text, dictionary)

    tab1, tab2, tab3, tab4 = st.tabs(["📄 元ファイル", "🖨️ OCRテキスト", "🤖 GPT補正", "✍️ 手作業修正"])
    with tab1:
        st.image(clean_img, caption=f"元ファイル (ページ {page_num})", use_container_width=True)
    with tab2:
        st.text_area(f"OCRテキスト（ページ {page_num}）", default_text, height=320, key=f"ocr_{page_num}")
    with tab3:
        st.text_area(f"GPT補正（ページ {page_num}）", gpt_checked_text, height=320, key=f"gpt_{page_num}")
    with tab4:
        corrected_text = st.text_area(
            f"手作業修正（ページ {page_num}）", gpt_checked_text, height=320, key=f"edit_{page_num}"
        )
        if st.button(f"修正を保存 (ページ {page_num})", key=f"save_{page_num}"):
            learned = learn_charwise_with_missing(default_text, corrected_text)
            if learned:
                update_dictionary_and_untrained(learned)
                st.success(f"辞書と学習候補に {len(learned)} 件を追加しました！")
            else:
                st.info("修正が検出されませんでした。")
            st.rerun()

    all_corrected.append(f"【ページ {page_num}】\n{(corrected_text or gpt_checked_text).strip()}")

# ==== 一括ダウンロード ====
if all_corrected:
    joined = "\n\n".join(all_corrected)
    st.download_button(
        "📥 補正テキストを一括ダウンロード",
        data=joined.encode("utf-8"),
        file_name="ocr_corrected_all.txt",
        mime="text/plain"
    )

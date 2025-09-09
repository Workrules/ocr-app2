# dummy line
import streamlit as st
import pypdfium2 as pdfium
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os
import io
from PIL import Image
import cv2
import numpy as np
import json
import openai
import re
import difflib
from itertools import zip_longest

# ==== Azure Document Intelligence 認証情報 ====
endpoint = os.getenv("AZURE_DOCINT_ENDPOINT")
key = os.getenv("AZURE_DOCINT_KEY")
if not endpoint or not key:
    st.error("環境変数 AZURE_DOCINT_ENDPOINT と AZURE_DOCINT_KEY を設定してください。")
    st.stop()
client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# ==== OpenAI APIキー ====
openai.api_key = os.getenv("OPENAI_API_KEY")

# ==== 辞書ファイル ====
DICT_FILE = "ocr_char_corrections.json"
UNTRAINED_FILE = "untrained_confusions.json"

JP_CHAR_RE = re.compile(r"^[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]$")

# ==== 印影除去 ====
def remove_red_stamp(img_pil):
    img = np.array(img_pil)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    img[mask > 0] = [255, 255, 255]  # 白塗り
    return Image.fromarray(img)

# ==== JSON管理 ====
def load_json(path: str) -> dict:
    return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else {}

def save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ==== 学習用の誤読抽出 ====
def learn_charwise_with_missing(original: str, corrected: str):
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
    # OCR補正用辞書を更新
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

    # 学習候補リスト（untrained）を更新
    untrained = load_json(UNTRAINED_FILE)
    for w, meta in learned.items():
        untrained[w] = meta["right"]
    save_json(untrained, UNTRAINED_FILE)

# ==== GPT補正 ====
def gpt_fix_text(text: str, dictionary: dict) -> str:
    prompt = f"""
次のOCR結果を自然な日本語に直してください。
- 日本語に存在しない文字は「□」にしてください。
- 辞書候補を参考にしてください: {json.dumps(dictionary, ensure_ascii=False)}
- 意味を勝手に補完せず、最小限の修正だけ行ってください。

OCR結果:
{text}
"""
    try:
        response = openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return text

# ==== PDFレンダリング ====
def render_pdf_bytes_to_images(pdf_bytes: bytes, dpi: int = 200):
    imgs = []
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    scale = dpi / 72.0
    for i in range(len(pdf)):
        page = pdf[i]
        pil = page.render(scale=scale).to_pil()
        imgs.append(pil.convert("RGB"))
    return imgs

# ==== Streamlit UI ====
st.title("📄 Document Intelligence OCR - GPT＋印影除去＋欠落補正")

dictionary = load_json(DICT_FILE)
st.sidebar.subheader("📖 現在の辞書")
st.sidebar.json(dictionary)

uploaded_file = st.file_uploader("画像またはPDFをアップロードしてください", type=["jpg", "jpeg", "png", "pdf"])

if not uploaded_file:
    st.info("📂 ここにファイルをアップロードしてください")
    st.stop()

file_bytes = uploaded_file.read()

# PDF/画像の分岐
try:
    if uploaded_file.type == "application/pdf":
        pages = render_pdf_bytes_to_images(file_bytes, dpi=200)
    else:
        pages = [Image.open(io.BytesIO(file_bytes)).convert("RGB")]
except Exception as e:
    st.error(f"ファイルの読み込みに失敗しました: {e}")
    st.stop()

if not pages:
    st.error("ページを生成できませんでした。ファイル形式をご確認ください。")
    st.stop()

# ==== ページ範囲選択（長尺PDF向けの高速化） ====
total_pages = len(pages)
start, end = st.slider(
    "処理するページ範囲を選択（1始まり）",
    min_value=1, max_value=total_pages, value=(1, min(total_pages, 5))
)
proc_range = range(start - 1, end)  # 0始まりのインデックス

# ==== ページごとの処理 ====
all_corrected = []

for page_index in proc_range:
    page_img = pages[page_index]
    page_num = page_index + 1
    st.write(f"## ページ {page_num}")

    # 印影除去
    clean_img = remove_red_stamp(page_img)

    # Azureに送る前にPNG圧縮
    buf = io.BytesIO()
    clean_img.save(buf, format="PNG")
    buf.seek(0)

    # OCR
    with st.spinner("OCRを実行中..."):
        poller = client.begin_analyze_document("prebuilt-read", document=buf)
        result = poller.result()

    # 各ページごとにOCRをかけているため、結果は先頭ページを参照するのが堅牢
    doc_page = result.pages[0] if getattr(result, "pages", None) else None
    if not doc_page:
        st.warning("OCR結果にページが見つかりませんでした。")
        continue

    default_text = "\n".join([line.content for line in doc_page.lines])

    # GPT補正
    gpt_checked_text = gpt_fix_text(default_text, dictionary)

    # ==== タブ ====
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📄 元ファイル", "🖨️ OCRテキスト", "🤖 GPT補正", "✍️ 手作業修正"]
    )
    with tab1:
        st.image(clean_img, caption=f"元ファイル (ページ {page_num})", use_container_width=True)
    with tab2:
        st.text_area(f"OCRテキスト（ページ {page_num}）", default_text, height=320)
    with tab3:
        st.text_area(f"GPT補正（ページ {page_num}）", gpt_checked_text, height=320)
    with tab4:
        corrected_text = st.text_area(
            f"手作業修正（ページ {page_num}）", gpt_checked_text, height=320, key=f"edit_{page_num}"
        )
        if st.button(f"修正を保存 (ページ {page_num})"):
            learned = learn_charwise_with_missing(default_text, corrected_text)
            if learned:
                update_dictionary_and_untrained(learned)
                st.success(f"辞書と学習候補に {len(learned)} 件を追加しました！")
            else:
                st.info("修正が検出されませんでした。")
            st.rerun()

    # 一括ダウンロード用の集約
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

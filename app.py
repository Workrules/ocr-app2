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
from typing import List, Tuple

# ==== 環境変数 ====
AZURE_ENDPOINT = os.getenv("AZURE_DOCINT_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_DOCINT_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OCR_GPT_MODEL", "gpt-5")  # gpt-5 / gpt-5-mini など

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
        # GPT-5系：Responses API。temperatureは未サポート（指定しない）
        resp = openai_client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            text={"verbosity": "low"},
            reasoning={"effort": "minimal"}
        )
        # 出力テキスト抽出
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()
        out_parts = []
        for item in getattr(resp, "output", []) or []:
            for part in getattr(item, "content", []) or []:
                t = getattr(part, "text", None)
                if t:
                    out_parts.append(t)
        out = "".join(out_parts).strip()
        return out or text
    except Exception as e:
        st.warning(f"GPT補正をスキップしました（エラー）：{e}")
        return text

def render_pdf_selected_pages(pdf_bytes: bytes, indices_0based: List[int], dpi: int = 200) -> Tuple[List[Image.Image], List[int]]:
    """選択ページ（0始まり）だけレンダリングして返す。page_numbersは1始まりで返す。"""
    imgs: List[Image.Image] = []
    nums: List[int] = []
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    scale = dpi / 72.0
    for idx in indices_0based:
        page = pdf[idx]
        pil = page.render(scale=scale).to_pil().convert("RGB")
        imgs.append(pil)
        nums.append(idx + 1)
    return imgs, nums

def is_pdf(b: bytes) -> bool:
    return len(b) >= 5 and b[:5] == b"%PDF-"

def parse_page_spec(spec: str, max_pages: int) -> List[int]:
    """
    '1,3,5-7' のような指定を0始まりのインデックス配列に変換。
    範囲外は自動でクリップ。重複は排除。昇順ソート。
    """
    s = (spec or "").strip()
    if not s:
        return []
    out = set()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                start = max(1, min(int(a), int(b)))
                end = min(max_pages, max(int(a), int(b)))
                for n in range(start, end + 1):
                    out.add(n - 1)
            except ValueError:
                continue
        else:
            try:
                n = int(p)
                if 1 <= n <= max_pages:
                    out.add(n - 1)
            except ValueError:
                continue
    return sorted(out)

# ==== UI ====
st.title("📄 Document Intelligence OCR - GPT＋印影除去＋欠落補正（ページ先指定）")

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

# ==== PDF/画像の分岐（ここでは まだOCR開始しない） ====
is_input_pdf = uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf") or is_pdf(file_bytes)

# ==== PDF の場合：まずページ指定フォームを出す（Submit後にのみOCR開始） ====
if is_input_pdf:
    # 軽量にページ数だけ取得（レンダリングはまだしない）
    try:
        pdf_for_count = pdfium.PdfDocument(io.BytesIO(file_bytes))
        total_pages = len(pdf_for_count)
    except Exception as e:
        st.exception(e)
        st.stop()

    with st.form("pdf_select_form"):
        st.subheader("▶ OCRするページを先に選択")
        select_mode = st.radio(
            "選択方法",
            options=["全ページ", "範囲指定", "ページ番号指定（例: 1,3,5-7）"],
            index=1 if total_pages > 1 else 0,
            horizontal=True
        )
        dpi = st.slider("レンダリングDPI（高いほど精細・重い）", min_value=72, max_value=300, value=200, step=4)

        if select_mode == "範囲指定" and total_pages > 1:
            start, end = st.slider(
                "処理するページ範囲（1始まり）",
                min_value=1, max_value=total_pages,
                value=(1, min(total_pages, 5))
            )
            chosen_indices = list(range(start - 1, end))
        elif select_mode == "ページ番号指定（例: 1,3,5-7）":
            spec = st.text_input("ページ番号（カンマ区切り、範囲はハイフン）", value="1-3" if total_pages >= 3 else "1")
            chosen_indices = parse_page_spec(spec, total_pages)
            if not chosen_indices:
                st.info("有効なページ番号を入力してください。例: 1,3,5-7")
        else:
            # 全ページ
            chosen_indices = list(range(total_pages))

        submitted = st.form_submit_button("このページだけOCRを実行")
    # ---- フォーム外：未Submitならここで終了（OCRは走らない） ----
    if not submitted or not chosen_indices:
        st.stop()

    # 必要なページだけレンダリング
    try:
        pages, page_numbers = render_pdf_selected_pages(file_bytes, chosen_indices, dpi=dpi)
    except Exception as e:
        st.exception(e)
        st.stop()

else:
    # 画像ファイル：ページ指定は不要
    try:
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            arr = np.frombuffer(file_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("画像の読み込みに失敗しました（JPG/PNG/PDFのみ対応）。")
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pages = [img]
        page_numbers = [1]
    except Exception as e:
        st.exception(e)
        st.stop()

# ==== ここからOCR開始（PDFも画像も合流） ====
all_corrected: List[str] = []

for page_img, page_num in zip(pages, page_numbers):
    st.write(f"## ページ {page_num}")

    # 印影除去
    clean_img = remove_red_stamp(page_img)

    # Azureに送る前にPNG化
    buf = io.BytesIO()
    clean_img.save(buf, format="PNG")
    buf.seek(0)

    # OCR
    with st.spinner("OCRを実行中..."):
        try:
            poller = client.begin_analyze_document("prebuilt-read", document=buf)
            result = poller.result()
        except Exception as e:
            st.exception(e)
            st.stop()

    # 各ページごとにOCRしているため結果は先頭を参照
    doc_page = result.pages[0] if getattr(result, "pages", None) else None
    if not doc_page:
        st.warning("OCR結果にページが見つかりませんでした。")
        continue

    default_text = "\n".join([line.content for line in doc_page.lines])

    # GPT補正
    dictionary = load_json(DICT_FILE)  # 処理中に辞書が更新される可能性を考慮して毎回読み出し
    gpt_checked_text = gpt_fix_text(default_text, dictionary)

    # ==== タブ ====
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

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
from typing import List, Tuple, Dict, Any
import gc

# === New: Word出力用 ===
from docx import Document
from docx.shared import Cm, Pt

# ==== 環境変数 ====
AZURE_ENDPOINT = os.getenv("AZURE_DOCINT_ENDPOINT")
AZURE_KEY = os.getenv("AZURE_DOCINT_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OCR_GPT_MODEL", "gpt-5")  # gpt-5 / gpt-5-mini など
BATCH_SIZE = max(1, int(os.getenv("OCR_BATCH_PAGES", "10")))  # 10ページずつ処理

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
        # GPT-5系：Responses API（temperature未サポートなので指定しない）
        resp = openai_client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            text={"verbosity": "low"},
            reasoning={"effort": "minimal"}
        )
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

def chunked(seq: List[int], n: int) -> List[List[int]]:
    """seqをn個ずつに分割して順に返す"""
    return [seq[i:i+n] for i in range(0, len(seq), n)]

# === New: 行ポリゴンから左端X/Yを取得 ===
def line_xy(line_obj: Any) -> Tuple[float, float]:
    """
    Azure Document IntelligenceのLineのbounding_polygonから
    左端x、上端y（小さいほど上）を推定する
    """
    poly = getattr(line_obj, "polygon", None) or getattr(line_obj, "bounding_polygon", None)
    if not poly:
        # v3系: line.polygon は [Point], v2024: bounding_regions/…実装差があるためガード
        return (0.0, 0.0)
    xs, ys = [], []
    for p in poly:
        # pはPoint(x,y)か、辞書{ "x":..., "y":... } の想定
        x = getattr(p, "x", None)
        y = getattr(p, "y", None)
        if x is None and isinstance(p, dict):
            x = p.get("x", 0.0)
            y = p.get("y", 0.0)
        xs.append(float(x))
        ys.append(float(y))
    return (min(xs or [0.0]), min(ys or [0.0]))

# === New: Word（docx）をざっくり元レイアウトに寄せて生成 ===
def build_docx_from_layout(pages_layout: List[Dict[str, Any]]) -> bytes:
    """
    pages_layout: 各ページの辞書リスト
      {
        "page_width": float,
        "page_height": float,
        "unit": str,  # "pixel", "inch", "millimeter" など（相対値で使うので文字列は参考）
        "lines": [ { "text": str, "x": float, "y": float }, ... ]  # y昇順x昇順でなくてもOK
      }
    """
    doc = Document()
    # A4縦＋余白（2cm）
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.0)
    section.right_margin = Cm(2.0)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(2.0)
    content_width_cm = (section.page_width - section.left_margin - section.right_margin).cm

    # 既定フォント（環境により変化）
    style = doc.styles["Normal"]
    style.font.name = "Yu Gothic"
    style.font.size = Pt(11)

    for idx, page in enumerate(pages_layout, start=1):
        pw = float(page.get("page_width") or 1.0)
        ph = float(page.get("page_height") or 1.0)
        lines = page.get("lines", [])

        # y→xで安定ソート
        lines_sorted = sorted(lines, key=lambda r: (r["y"], r["x"]))

        # 行間のしきい値（縦方向クラスタリング）
        y_thresh = ph * 0.018  # ページ高さの約1.8%（調整可）
        prev_y = None

        for item in lines_sorted:
            txt = item["text"]
            x = float(item["x"])
            y = float(item["y"])

            # 新しい段落
            para = doc.add_paragraph()
            run = para.add_run(txt)

            # 左インデント：左余白＋(x/pw) * content_width
            # 深すぎる食い込みにならないように最大0.9倍にクリップ
            indent_cm = max(0.0, min(0.9 * content_width_cm, (x / max(pw, 1e-6)) * content_width_cm))
            para.paragraph_format.left_indent = Cm(indent_cm)

            # 行間を少し詰める（見た目用）
            para.paragraph_format.space_after = Pt(2)

            # 縦方向の大きな飛びは段落前スペースで表現
            if prev_y is not None and (y - prev_y) > y_thresh:
                para.paragraph_format.space_before = Pt(8)

            prev_y = y

        # ページ区切り（最終ページ以外）
        if idx < len(pages_layout):
            doc.add_page_break()

    bio = io.BytesIO()
    doc.save(bio)
    bio.seek(0)
    return bio.read()

# ==== UI ====
st.title("📄 Document Intelligence OCR - GPT＋印影除去＋欠落補正（ページ先指定・10ページバッチ・Word出力）")

dictionary = load_json(DICT_FILE)
st.sidebar.subheader("📖 現在の辞書")
st.sidebar.json(dictionary)
st.sidebar.markdown("### 🔧 診断")
st.sidebar.write({
    "AZURE_DOCINT_ENDPOINT_set": bool(AZURE_ENDPOINT),
    "AZURE_DOCINT_KEY_set": bool(AZURE_KEY),
    "OPENAI_API_KEY_set": bool(OPENAI_API_KEY),
    "MODEL": MODEL_NAME,
    "BATCH_SIZE": BATCH_SIZE,
})

uploaded_file = st.file_uploader("画像またはPDFをアップロードしてください", type=["jpg", "jpeg", "png", "pdf"])
if not uploaded_file:
    st.info("📂 ここにファイルをアップロードしてください")
    st.stop()

file_bytes = uploaded_file.read()
ImageFile.LOAD_TRUNCATED_IMAGES = True
is_input_pdf = uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf") or is_pdf(file_bytes)

# ==== PDF：OCR前にページ指定フォーム ====
if is_input_pdf:
    # 軽量にページ数だけ取得
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
            chosen_indices = list(range(total_pages))

        submitted = st.form_submit_button("このページだけOCRを実行")

    if not submitted or not chosen_indices:
        st.stop()

    # ==== ここから10ページずつのバッチ処理 ====
    total_to_process = len(chosen_indices)
    progress = st.progress(0.0)
    status = st.empty()
    all_corrected_texts: List[str] = []     # ★ページ見出しなしの連結用
    pages_layout: List[Dict[str, Any]] = [] # ★Wordレイアウト用
    done = 0

    for batch_no, batch_indices in enumerate(chunked(chosen_indices, BATCH_SIZE), start=1):
        status.info(f"🔄 バッチ {batch_no} / {((total_to_process - 1) // BATCH_SIZE) + 1} を処理中（ページ: {', '.join(str(i+1) for i in batch_indices)}）")

        # 必要なページだけ都度レンダリングしてメモリ節約
        try:
            pages, page_numbers = render_pdf_selected_pages(file_bytes, batch_indices, dpi=dpi)
        except Exception as e:
            st.exception(e)
            st.stop()

        # バッチ内ページを処理
        for page_img, page_num in zip(pages, page_numbers):
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
                done += 1
                progress.progress(done / total_to_process)
                continue

            # Azureの行を取得（座標付き）
            azure_lines = getattr(doc_page, "lines", []) or []
            default_text = "\n".join([line.content for line in azure_lines])

            dictionary = load_json(DICT_FILE)
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
                    f"手作業修正（ページ {page_num})」", gpt_checked_text, height=320, key=f"edit_{page_num}"
                )
                if st.button(f"修正を保存 (ページ {page_num})", key=f"save_{page_num}"):
                    learned = learn_charwise_with_missing(default_text, corrected_text)
                    if learned:
                        update_dictionary_and_untrained(learned)
                        st.success(f"辞書と学習候補に {len(learned)} 件を追加しました！")
                    else:
                        st.info("修正が検出されませんでした。")
                    st.rerun()

            # === (1) TXT用：ページ見出しなしで連結 ===
            final_text_page = (corrected_text or gpt_checked_text).strip()
            all_corrected_texts.append(final_text_page)

            # === (2) Word用レイアウト収集 ===
            #   - Azureの行座標を使って左インデントや段落の間隔を再現
            #   - 行テキストは、GPT補正結果の行を優先（行数が合わなければ可能な範囲で上から対応）
            gpt_lines = [ln for ln in (corrected_text or gpt_checked_text).splitlines()]
            lines_for_layout = []
            for i, ln in enumerate(azure_lines):
                x, y = line_xy(ln)
                text_for_line = gpt_lines[i] if i < len(gpt_lines) else ln.content
                lines_for_layout.append({"text": text_for_line, "x": x, "y": y})

            page_layout_info = {
                "page_width": getattr(doc_page, "width", None) or 1.0,
                "page_height": getattr(doc_page, "height", None) or 1.0,
                "unit": getattr(doc_page, "unit", None) or "pixel",
                "lines": lines_for_layout
            }
            pages_layout.append(page_layout_info)

            # 進捗更新
            done += 1
            progress.progress(done / total_to_process)

        # バッチ終了ごとにクリーンアップ
        del pages
        del page_numbers
        gc.collect()

    status.success("✅ すべてのページの処理が完了しました。")

    # ==== ダウンロード（ページ見出しなしTXT / Word） ====
    if all_corrected_texts:
        joined_txt = "\n\n".join(all_corrected_texts)
        st.download_button(
            "📥 補正テキストをダウンロード（TXT, ページ見出しなし）",
            data=joined_txt.encode("utf-8"),
            file_name="ocr_corrected.txt",
            mime="text/plain"
        )

    if pages_layout:
        try:
            docx_bytes = build_docx_from_layout(pages_layout)
            st.download_button(
                "📥 Word（.docx）でダウンロード（レイアウト近似）",
                data=docx_bytes,
                file_name="ocr_layout.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        except Exception as e:
            st.warning(f"Word出力に失敗しました：{e}")

else:
    # ==== 画像ファイル：そのまま1ページ処理 ====
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

    all_corrected_texts: List[str] = []
    pages_layout: List[Dict[str, Any]] = []

    for page_img, page_num in zip(pages, page_numbers):
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

        azure_lines = getattr(doc_page, "lines", []) or []
        default_text = "\n".join([line.content for line in azure_lines])

        dictionary = load_json(DICT_FILE)
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

        # TXT（ページ見出しなし）
        final_text_page = (corrected_text or gpt_checked_text).strip()
        all_corrected_texts.append(final_text_page)

        # Wordレイアウト情報
        gpt_lines = [ln for ln in (corrected_text or gpt_checked_text).splitlines()]
        lines_for_layout = []
        for i, ln in enumerate(azure_lines):
            x, y = line_xy(ln)
            text_for_line = gpt_lines[i] if i < len(gpt_lines) else ln.content
            lines_for_layout.append({"text": text_for_line, "x": x, "y": y})

        page_layout_info = {
            "page_width": getattr(doc_page, "width", None) or 1.0,
            "page_height": getattr(doc_page, "height", None) or 1.0,
            "unit": getattr(doc_page, "unit", None) or "pixel",
            "lines": lines_for_layout
        }
        pages_layout.append(page_layout_info)

    # ダウンロード（TXT / Word）
    if all_corrected_texts:
        joined_txt = "\n\n".join(all_corrected_texts)
        st.download_button(
            "📥 補正テキストをダウンロード（TXT, ページ見出しなし）",
            data=joined_txt.encode("utf-8"),
            file_name="ocr_corrected.txt",
            mime="text/plain"
        )

    if pages_layout:
        try:
            docx_bytes = build_docx_from_layout(pages_layout)
            st.download_button(
                "📥 Word（.docx）でダウンロード（レイアウト近似）",
                data=docx_bytes,
                file_name="ocr_layout.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        except Exception as e:
            st.warning(f"Word出力に失敗しました：{e}")

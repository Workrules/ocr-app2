# dummy line
import streamlit as st
import os
import io
import json
import time
import gc
import re
from itertools import zip_longest
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
from PIL import Image, ImageFile
import pypdfium2 as pdfium

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from openai import OpenAI

# ===== Word出力（シンプルなレイアウト近似） =====
from docx import Document
from docx.shared import Cm, Pt

# ===================== 基本設定 =====================
# Azure Document Intelligence / OpenAI
AZURE_DOCINT_ENDPOINT = os.getenv("AZURE_DOCINT_ENDPOINT") or st.secrets.get("AZURE_DOCINT_ENDPOINT")
AZURE_DOCINT_KEY = os.getenv("AZURE_DOCINT_KEY") or st.secrets.get("AZURE_DOCINT_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OCR_GPT_MODEL") or st.secrets.get("OCR_GPT_MODEL", "gpt-5")
BATCH_SIZE_DEFAULT = max(1, int(os.getenv("OCR_BATCH_PAGES") or st.secrets.get("OCR_BATCH_PAGES", "10")))

if not AZURE_DOCINT_ENDPOINT or not AZURE_DOCINT_KEY:
    st.error("環境変数またはSecretsで AZURE_DOCINT_ENDPOINT / AZURE_DOCINT_KEY を設定してください。")
    st.stop()

# クライアント
client = DocumentAnalysisClient(endpoint=AZURE_DOCINT_ENDPOINT, credential=AzureKeyCredential(AZURE_DOCINT_KEY))
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# 文字判定
JP_CHAR_RE = re.compile(r"^[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]$")

# ===================== ストレージ選択（local / azureblob） =====================
STORAGE_BACKEND = os.getenv("OCR_DICT_BACKEND") or st.secrets.get("OCR_DICT_BACKEND", "local")

if STORAGE_BACKEND == "azureblob":
    # Azure Blob Storage を使用（本番Web推奨）
    try:
        from azure.storage.blob import BlobServiceClient, ContentSettings
        from azure.core.exceptions import ResourceNotFoundError, ResourceExistsError
    except Exception as e:
        st.error(f"azure-storage-blob が必要です。requirements.txt に追加してください。詳細: {e}")
        st.stop()

    AZURE_BLOB_CONN_STR = os.getenv("AZURE_BLOB_CONN_STR") or st.secrets.get("AZURE_BLOB_CONN_STR")
    OCR_DICT_CONTAINER = os.getenv("OCR_DICT_CONTAINER") or st.secrets.get("OCR_DICT_CONTAINER", "ocr-shared-dict")

    if not AZURE_BLOB_CONN_STR:
        st.error("Secrets/環境変数 AZURE_BLOB_CONN_STR を設定してください（Blob接続文字列）。")
        st.stop()

    _blob_service = BlobServiceClient.from_connection_string(AZURE_BLOB_CONN_STR)
    _container = _blob_service.get_container_client(OCR_DICT_CONTAINER)
    try:
        _container.create_container()
    except ResourceExistsError:
        pass  # 既存ならOK

    # Blob名（固定）
    DICT_FILE = "ocr_char_corrections.json"
    UNTRAINED_FILE = "untrained_confusions.json"
    TRAINED_FILE = "trained_confusions.json"

    # Blob I/O
    def _load_json_blob(blob_name: str) -> dict:
        try:
            data = _container.download_blob(blob_name).readall()
            return json.loads(data.decode("utf-8"))
        except ResourceNotFoundError:
            return {}
        except Exception:
            return {}

    def _save_json_blob(obj: dict, blob_name: str):
        b = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        _container.upload_blob(
            name=blob_name,
            data=b,
            overwrite=True,
            content_settings=ContentSettings(content_type="application/json; charset=utf-8"),
        )

    def load_json_any(key: str) -> dict:
        return _load_json_blob(key)

    def save_json_any(obj: dict, key: str):
        _save_json_blob(obj, key)

elif STORAGE_BACKEND == "local":
    # ローカルJSON（開発・検証向け）
    DICT_DIR = os.getenv("OCR_DICT_DIR") or st.secrets.get("OCR_DICT_DIR", ".")
    DICT_FILE = os.path.join(DICT_DIR, "ocr_char_corrections.json")
    UNTRAINED_FILE = os.path.join(DICT_DIR, "untrained_confusions.json")
    TRAINED_FILE = os.path.join(DICT_DIR, "trained_confusions.json")

    def _load_json_local(path: str, retries: int = 3, delay: float = 0.1) -> dict:
        for _ in range(retries):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except FileNotFoundError:
                return {}
            except (json.JSONDecodeError, PermissionError):
                time.sleep(delay)
        return {}

    def _save_json_local(obj: dict, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = f"{path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp_path, path)

    def load_json_any(key: str) -> dict:
        return _load_json_local(key)

    def save_json_any(obj: dict, key: str):
        _save_json_local(obj, key)

else:
    st.error(f"未知の OCR_DICT_BACKEND: {STORAGE_BACKEND}（local / azureblob のみ対応）")
    st.stop()

# ===================== ユーティリティ =====================
def remove_red_stamp(img_pil: Image.Image) -> Image.Image:
    """赤い印影を白に飛ばす簡易フィルタ"""
    img = np.array(img_pil)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50]); upper_red2 = np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1),
                          cv2.inRange(hsv, lower_red2, upper_red2))
    img[mask > 0] = [255, 255, 255]
    return Image.fromarray(img)

def learn_charwise_with_missing(original: str, corrected: str) -> dict:
    """文字単位の差分から「誤→正」を学習（欠落は '□' として扱う）"""
    learned = {}
    import difflib
    sm = difflib.SequenceMatcher(None, original, corrected)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ["replace", "insert"]:
            o_seg, c_seg = original[i1:i2], corrected[j1:j2]
            for o_char, c_char in zip_longest(o_seg, c_seg, fillvalue=""):
                if c_char and (not o_char or o_char != c_char):
                    wrong = o_char if o_char else "□"
                    if JP_CHAR_RE.match(c_char) or c_char == "□":
                        learned[wrong] = {"right": c_char, "count": 1}
    return learned

def update_dictionary_and_untrained(learned: dict):
    # メイン辞書
    dictionary = load_json_any(DICT_FILE)
    for w, meta in learned.items():
        if w in dictionary:
            if dictionary[w]["right"] == meta["right"]:
                dictionary[w]["count"] += meta["count"]
            else:
                if meta["count"] > dictionary[w]["count"]:
                    dictionary[w] = meta
        else:
            dictionary[w] = meta
    save_json_any(dictionary, DICT_FILE)
    # 未学習候補
    untrained = load_json_any(UNTRAINED_FILE)
    for w, meta in learned.items():
        untrained[w] = meta["right"]
    save_json_any(untrained, UNTRAINED_FILE)

def gpt_fix_text(text: str, dictionary: dict) -> str:
    """GPTで最小限に整形（temperature等は未指定：gpt-5既定値で実行）"""
    prompt = f"""
次のOCR結果を自然な日本語に直してください。
- 日本語に存在しない文字は「□」にしてください。
- 辞書候補を参考にしてください: {json.dumps(dictionary, ensure_ascii=False)}
- 意味を勝手に補完せず、最小限の修正だけ行ってください。

OCR結果:
{text}
""".strip()
    try:
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

# ========== PDFレンダリング・ページ指定 ==========
def is_pdf(b: bytes) -> bool:
    return len(b) >= 5 and b[:5] == b"%PDF-"

def render_pdf_selected_pages(pdf_bytes: bytes, indices_0based: List[int], dpi: int = 200) -> Tuple[List[Image.Image], List[int]]:
    """選択ページ（0始まり）だけレンダリングして返す。戻りのpage_numbersは1始まり。"""
    imgs: List[Image.Image] = []
    nums: List[int] = []
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    scale = dpi / 72.0
    for idx in indices_0based:
        page = pdf[idx]
        pil = page.render(scale=scale).to_pil().convert("RGB")
        imgs.append(pil); nums.append(idx + 1)
    return imgs, nums

def parse_page_spec(spec: str, max_pages: int) -> List[int]:
    """'1,3,5-7' → 0始まりの昇順インデックス配列（範囲外クリップ・重複排除）"""
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
    return [seq[i:i+n] for i in range(0, len(seq), n)]

# ========== Azure行の座標（左x/上y） ==========
def line_xy(line_obj: Any) -> Tuple[float, float]:
    poly = getattr(line_obj, "polygon", None) or getattr(line_obj, "bounding_polygon", None)
    if not poly:
        return (0.0, 0.0)
    xs, ys = [], []
    for p in poly:
        x = getattr(p, "x", None); y = getattr(p, "y", None)
        if x is None and isinstance(p, dict):
            x = p.get("x", 0.0); y = p.get("y", 0.0)
        xs.append(float(x)); ys.append(float(y))
    return (min(xs or [0.0]), min(ys or [0.0]))

# ========== Word（docx）生成（左インデント近似のみ） ==========
EMU_PER_CM = 360000.0
def to_cm(val) -> float:
    try:
        return float(getattr(val, "cm"))
    except Exception:
        try:
            return float(val) / EMU_PER_CM
        except Exception:
            return float(val)

def build_docx_from_layout(pages_layout: List[Dict[str, Any]]) -> bytes:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0); section.page_height = Cm(29.7)
    section.left_margin = Cm(2.0); section.right_margin = Cm(2.0)
    section.top_margin = Cm(2.0); section.bottom_margin = Cm(2.0)

    page_w_cm = to_cm(section.page_width)
    left_cm = to_cm(section.left_margin); right_cm = to_cm(section.right_margin)
    content_width_cm = max(0.1, page_w_cm - left_cm - right_cm)

    style = doc.styles["Normal"]; style.font.name = "Yu Gothic"; style.font.size = Pt(11)

    for idx, page in enumerate(pages_layout, start=1):
        pw = float(page.get("page_width") or 1.0)
        ph = float(page.get("page_height") or 1.0)
        lines = page.get("lines", [])
        lines_sorted = sorted(lines, key=lambda r: (r["y"], r["x"]))
        y_thresh = ph * 0.018
        prev_y = None

        for item in lines_sorted:
            txt = item["text"]; x = float(item["x"]); y = float(item["y"])
            para = doc.add_paragraph(); para.add_run(txt)
            indent_cm = max(0.0, min(0.9 * content_width_cm, (x / max(pw, 1e-6)) * content_width_cm))
            para.paragraph_format.left_indent = Cm(indent_cm)
            para.paragraph_format.space_after = Pt(2)
            if prev_y is not None and (y - prev_y) > y_thresh:
                para.paragraph_format.space_before = Pt(8)
            prev_y = y

        if idx < len(pages_layout):
            doc.add_page_break()

    bio = io.BytesIO(); doc.save(bio); bio.seek(0)
    return bio.read()

# ===================== UI =====================
st.title("📄 Document Intelligence OCR - Web（Azure Blob辞書）/ ページ指定 / 10ページバッチ / GPT補正 / Word出力 / 画面保持")

# 診断
st.sidebar.markdown("### 🔧 環境")
st.sidebar.write({
    "AZURE_DOCINT_ENDPOINT_set": bool(AZURE_DOCINT_ENDPOINT),
    "AZURE_DOCINT_KEY_set": bool(AZURE_DOCINT_KEY),
    "OPENAI_API_KEY_set": bool(OPENAI_API_KEY),
    "MODEL": MODEL_NAME,
    "BATCH_SIZE(default)": BATCH_SIZE_DEFAULT,
})

st.sidebar.markdown("### 📂 辞書の参照先")
if STORAGE_BACKEND == "azureblob":
    st.sidebar.write({
        "OCR_DICT_BACKEND": "azureblob",
        "container": os.getenv("OCR_DICT_CONTAINER") or st.secrets.get("OCR_DICT_CONTAINER", "ocr-shared-dict"),
        "DICT_BLOB": DICT_FILE,
        "UNTRAINED_BLOB": UNTRAINED_FILE,
        "TRAINED_BLOB": TRAINED_FILE,
    })
else:
    st.sidebar.write({
        "OCR_DICT_BACKEND": "local",
        "OCR_DICT_DIR": os.path.abspath(os.getenv("OCR_DICT_DIR") or st.secrets.get("OCR_DICT_DIR", ".")),
        "DICT_FILE": os.path.abspath(DICT_FILE),
        "UNTRAINED_FILE": os.path.abspath(UNTRAINED_FILE),
        "TRAINED_FILE": os.path.abspath(TRAINED_FILE),
    })

# デバッグUI
st.sidebar.markdown("### 🛠 デバッグ")
skip_gpt = st.sidebar.checkbox("GPT補正をスキップ", value=False)
ocr_timeout = st.sidebar.slider("OCRタイムアウト（秒）", 10, 180, 60, step=5)
batch_size_override = st.sidebar.number_input("バッチサイズ上書き", 1, 20, value=BATCH_SIZE_DEFAULT)

# 現在の辞書プレビュー
dictionary_preview = load_json_any(DICT_FILE)
st.sidebar.subheader("📖 現在の辞書（プレビュー）")
st.sidebar.json(dictionary_preview)

# ファイル入力
uploaded_file = st.file_uploader("画像またはPDFをアップロードしてください", type=["jpg", "jpeg", "png", "pdf"])
if not uploaded_file:
    st.info("📂 ここにファイルをアップロードしてください")
    st.stop()

file_bytes = uploaded_file.read()
ImageFile.LOAD_TRUNCATED_IMAGES = True
is_input_pdf = uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf") or is_pdf(file_bytes)

# ===================== PDF：OCR前にページ指定 =====================
if is_input_pdf:
    try:
        pdf_for_count = pdfium.PdfDocument(io.BytesIO(file_bytes))
        total_pages = len(pdf_for_count)
    except Exception as e:
        st.exception(e); st.stop()

    with st.form("pdf_select_form"):
        st.subheader("▶ OCRするページを先に選択")
        select_mode = st.radio(
            "選択方法",
            options=["全ページ", "範囲指定", "ページ番号指定（例: 1,3,5-7）"],
            index=1 if total_pages > 1 else 0,
            horizontal=True
        )
        dpi = st.slider("レンダリングDPI（高いほど精細・重い）", 72, 300, 200, step=4)

        if select_mode == "範囲指定" and total_pages > 1:
            start, end = st.slider("処理するページ範囲（1始まり）", 1, total_pages, (1, min(total_pages, 5)))
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

    EFFECTIVE_BATCH = int(batch_size_override) if batch_size_override else BATCH_SIZE_DEFAULT

    total_to_process = len(chosen_indices)
    progress = st.progress(0.0)
    status = st.empty()
    all_corrected_texts: List[str] = []      # TXT用（ページ見出しなし）
    pages_layout: List[Dict[str, Any]] = []  # Word用
    done = 0

    for batch_no, batch_indices in enumerate(chunked(chosen_indices, EFFECTIVE_BATCH), start=1):
        status.info(f"🔄 バッチ {batch_no} / {((total_to_process - 1) // EFFECTIVE_BATCH) + 1} を処理中（ページ: {', '.join(str(i+1) for i in batch_indices)}）")
        try:
            pages, page_numbers = render_pdf_selected_pages(file_bytes, batch_indices, dpi=dpi)
        except Exception as e:
            st.exception(e); st.stop()

        for page_img, page_num in zip(pages, page_numbers):
            st.write(f"## ページ {page_num}")
            clean_img = remove_red_stamp(page_img)

            buf = io.BytesIO(); clean_img.save(buf, format="PNG"); buf.seek(0)

            with st.spinner("OCRを実行中..."):
                t0 = time.perf_counter()
                try:
                    poller = client.begin_analyze_document("prebuilt-read", document=buf)
                    result = poller.result(timeout=float(ocr_timeout))
                except Exception as e:
                    st.error(f"OCRが{ocr_timeout}秒以内に完了しませんでした / 失敗しました：{e}")
                    st.caption(f"OCR実行時間: {time.perf_counter() - t0:.1f}s")
                    continue
                st.caption(f"OCR実行時間: {time.perf_counter() - t0:.1f}s")

            doc_page = result.pages[0] if getattr(result, "pages", None) else None
            if not doc_page:
                st.warning("OCR結果にページが見つかりませんでした。")
                done += 1; progress.progress(done / total_to_process)
                continue

            azure_lines = getattr(doc_page, "lines", []) or []
            default_text = "\n".join([line.content for line in azure_lines])

            # 共有辞書を毎回最新で読みつつ、必要に応じてGPT補正
            dictionary = load_json_any(DICT_FILE)
            gpt_checked_text = default_text if skip_gpt else gpt_fix_text(default_text, dictionary)

            # --- セッションキー ---
            ocr_key = f"ocr_{page_num}"
            gpt_key = f"gpt_{page_num}"
            edit_key = f"edit_{page_num}"

            # 初期化（初回だけ）
            if ocr_key not in st.session_state:
                st.session_state[ocr_key] = default_text
            if gpt_key not in st.session_state:
                st.session_state[gpt_key] = gpt_checked_text
            if edit_key not in st.session_state:
                st.session_state[edit_key] = gpt_checked_text

            # ビュータブ
            tab1, tab2, tab3, tab4 = st.tabs(["📄 元ファイル", "🖨️ OCRテキスト", "🤖 GPT補正", "✍️ 手作業修正"])
            with tab1:
                st.image(clean_img, caption=f"元ファイル (ページ {page_num})", use_container_width=True)
            with tab2:
                st.text_area(
                    f"OCRテキスト（ページ {page_num}）",
                    value=st.session_state.get(ocr_key, default_text),
                    height=320,
                    key=ocr_key
                )
            with tab3:
                st.text_area(
                    f"GPT補正（ページ {page_num}）",
                    value=st.session_state.get(gpt_key, gpt_checked_text),
                    height=320,
                    key=gpt_key
                )
            with tab4:
                st.text_area(
                    f"手作業修正（ページ {page_num}）",
                    value=st.session_state.get(edit_key, gpt_checked_text),
                    height=320,
                    key=edit_key
                )
                if st.button(f"修正を保存 (ページ {page_num})", key=f"save_{page_num}"):
                    corrected_text_current = st.session_state.get(edit_key, gpt_checked_text)
                    learned = learn_charwise_with_missing(default_text, corrected_text_current)
                    if learned:
                        update_dictionary_and_untrained(learned)
                        st.success(f"辞書と学習候補に {len(learned)} 件を追加しました！")
                    else:
                        st.info("修正が検出されませんでした。")
                    # 画面保持のため rerun しない

            # TXT（ページ見出しなしで連結：セッションを優先）
            final_text_page = (st.session_state.get(edit_key) or st.session_state.get(gpt_key) or gpt_checked_text or default_text).strip()
            all_corrected_texts.append(final_text_page)

            # Word用レイアウト：セッションの編集結果を優先
            gpt_lines = [ln for ln in final_text_page.splitlines()]
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

            done += 1; progress.progress(done / total_to_process)

        del pages, page_numbers
        gc.collect()

    status.success("✅ すべてのページの処理が完了しました。")

    # ダウンロード
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
                "📥 Word（.docx：レイアウト近似）",
                data=docx_bytes,
                file_name="ocr_layout.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        except Exception as e:
            st.warning(f"Word出力に失敗しました：{e}")

# ===================== 画像：1ページ処理 =====================
else:
    try:
        try:
            img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        except Exception:
            arr = np.frombuffer(file_bytes, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError("画像の読み込みに失敗しました（JPG/PNG/PDFのみ対応）。")
            img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        pages = [img]; page_numbers = [1]
    except Exception as e:
        st.exception(e); st.stop()

    all_corrected_texts: List[str] = []
    pages_layout: List[Dict[str, Any]] = []

    for page_img, page_num in zip(pages, page_numbers):
        st.write(f"## ページ {page_num}")
        clean_img = remove_red_stamp(page_img)
        buf = io.BytesIO(); clean_img.save(buf, format="PNG"); buf.seek(0)

        with st.spinner("OCRを実行中..."):
            t0 = time.perf_counter()
            try:
                poller = client.begin_analyze_document("prebuilt-read", document=buf)
                result = poller.result(timeout=float(ocr_timeout))
            except Exception as e:
                st.error(f"OCRが{ocr_timeout}秒以内に完了しませんでした / 失敗しました：{e}")
                st.caption(f"OCR実行時間: {time.perf_counter() - t0:.1f}s")
                continue
            st.caption(f"OCR実行時間: {time.perf_counter() - t0:.1f}s")

        doc_page = result.pages[0] if getattr(result, "pages", None) else None
        if not doc_page:
            st.warning("OCR結果にページが見つかりませんでした。")
            continue

        azure_lines = getattr(doc_page, "lines", []) or []
        default_text = "\n".join([line.content for line in azure_lines])

        dictionary = load_json_any(DICT_FILE)
        gpt_checked_text = default_text if skip_gpt else gpt_fix_text(default_text, dictionary)

        # --- セッションキー ---
        ocr_key = f"ocr_{page_num}"
        gpt_key = f"gpt_{page_num}"
        edit_key = f"edit_{page_num}"

        if ocr_key not in st.session_state:
            st.session_state[ocr_key] = default_text
        if gpt_key not in st.session_state:
            st.session_state[gpt_key] = gpt_checked_text
        if edit_key not in st.session_state:
            st.session_state[edit_key] = gpt_checked_text

        tab1, tab2, tab3, tab4 = st.tabs(["📄 元ファイル", "🖨️ OCRテキスト", "🤖 GPT補正", "✍️ 手作業修正"])
        with tab1:
            st.image(clean_img, caption=f"元ファイル (ページ {page_num})", use_container_width=True)
        with tab2:
            st.text_area(
                f"OCRテキスト（ページ {page_num}）",
                value=st.session_state.get(ocr_key, default_text),
                height=320,
                key=ocr_key
            )
        with tab3:
            st.text_area(
                f"GPT補正（ページ {page_num}）",
                value=st.session_state.get(gpt_key, gpt_checked_text),
                height=320,
                key=gpt_key
            )
        with tab4:
            st.text_area(
                f"手作業修正（ページ {page_num}）",
                value=st.session_state.get(edit_key, gpt_checked_text),
                height=320,
                key=edit_key
            )
            if st.button(f"修正を保存 (ページ {page_num})", key=f"save_{page_num}"):
                corrected_text_current = st.session_state.get(edit_key, gpt_checked_text)
                learned = learn_charwise_with_missing(default_text, corrected_text_current)
                if learned:
                    update_dictionary_and_untrained(learned)
                    st.success(f"辞書と学習候補に {len(learned)} 件を追加しました！")
                else:
                    st.info("修正が検出されませんでした。")
                # rerunしない

        # TXT
        final_text_page = (st.session_state.get(edit_key) or st.session_state.get(gpt_key) or gpt_checked_text or default_text).strip()
        all_corrected_texts.append(final_text_page)

        # Word
        gpt_lines = [ln for ln in final_text_page.splitlines()]
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
                "📥 Word（.docx：レイアウト近似）",
                data=docx_bytes,
                file_name="ocr_layout.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )
        except Exception as e:
            st.warning(f"Word出力に失敗しました：{e}")

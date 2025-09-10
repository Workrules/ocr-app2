# dummy line
import streamlit as st
import os
import io
import json
import time
import gc
import re
import hashlib
import unicodedata
from itertools import zip_longest
from typing import List, Tuple, Dict, Any

import numpy as np
import cv2
from PIL import Image, ImageFile
import pypdfium2 as pdfium

from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential

from openai import OpenAI

from docx import Document
from docx.shared import Cm, Pt

# ===================== 基本設定 =====================
AZURE_DOCINT_ENDPOINT = os.getenv("AZURE_DOCINT_ENDPOINT") or st.secrets.get("AZURE_DOCINT_ENDPOINT")
AZURE_DOCINT_KEY = os.getenv("AZURE_DOCINT_KEY") or st.secrets.get("AZURE_DOCINT_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY")
MODEL_NAME = os.getenv("OCR_GPT_MODEL") or st.secrets.get("OCR_GPT_MODEL", "gpt-5")
BATCH_SIZE_DEFAULT = max(1, int(os.getenv("OCR_BATCH_PAGES") or st.secrets.get("OCR_BATCH_PAGES", "10")))

if not AZURE_DOCINT_ENDPOINT or not AZURE_DOCINT_KEY:
    st.error("環境変数またはSecretsで AZURE_DOCINT_ENDPOINT / AZURE_DOCINT_KEY を設定してください。")
    st.stop()

client = DocumentAnalysisClient(endpoint=AZURE_DOCINT_ENDPOINT, credential=AzureKeyCredential(AZURE_DOCINT_KEY))
openai_client = OpenAI(api_key=OPENAI_API_KEY)

JP_CHAR_RE = re.compile(r"^[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]$")

# ===================== ストレージ選択（local / azureblob） =====================
STORAGE_BACKEND = os.getenv("OCR_DICT_BACKEND") or st.secrets.get("OCR_DICT_BACKEND", "local")

if STORAGE_BACKEND == "azureblob":
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
        pass

    DICT_FILE = "ocr_char_corrections.json"
    UNTRAINED_FILE = "untrained_confusions.json"
    TRAINED_FILE = "trained_confusions.json"

    def _load_json_blob(blob_name: str) -> dict:
        try:
            data = _container.download_blob(blob_name).readall()
            return json.loads(data.decode("utf-8"))
        except Exception:
            return {}

    def _save_json_blob(obj: dict, blob_name: str):
        b = json.dumps(obj, ensure_ascii=False, indent=2).encode("utf-8")
        _container.upload_blob(
            name=blob_name, data=b, overwrite=True,
            content_settings=ContentSettings(content_type="application/json; charset=utf-8"),
        )

    def load_json_any(key: str) -> dict:
        return _load_json_blob(key)
    def save_json_any(obj: dict, key: str):
        _save_json_blob(obj, key)

elif STORAGE_BACKEND == "local":
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
    img = np.array(img_pil)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    lower_red1 = np.array([0, 70, 50]); upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50]); upper_red2 = np.array([180, 255, 255])
    mask = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    img[mask > 0] = [255, 255, 255]
    return Image.fromarray(img)

def learn_charwise_with_missing(original: str, corrected: str) -> dict:
    learned: Dict[str, Dict[str, Any]] = {}
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
    untrained = load_json_any(UNTRAINED_FILE)
    for w, meta in learned.items():
        untrained[w] = meta["right"]
    save_json_any(untrained, UNTRAINED_FILE)

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
        resp = openai_client.responses.create(
            model=MODEL_NAME,
            input=prompt,
            text={"verbosity": "low"},
            reasoning={"effort": "minimal"},
        )
        if hasattr(resp, "output_text") and resp.output_text:
            return resp.output_text.strip()
        out_parts = []
        for item in getattr(resp, "output", []) or []:
            for part in getattr(item, "content", []) or []:
                t = getattr(part, "text", None)
                if t: out_parts.append(t)
        out = "".join(out_parts).strip()
        return out or text
    except Exception as e:
        st.warning(f"GPT補正をスキップしました（エラー）：{e}")
        return text

# ========== PDFレンダリング ==========
def is_pdf(b: bytes) -> bool:
    return len(b) >= 5 and b[:5] == b"%PDF-"

def render_pdf_selected_pages(pdf_bytes: bytes, indices_0based: List[int], dpi: int = 200) -> Tuple[List[Image.Image], List[int]]:
    imgs: List[Image.Image] = []
    nums: List[int] = []
    pdf = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    scale = dpi / 72.0
    for idx in indices_0based:
        page = pdf[idx]
        pil = page.render(scale=scale).to_pil().convert("RGB")
        imgs.append(pil); nums.append(idx + 1)
    return imgs, nums

# ====== ★ ページ指定の正規化・解析（全角対応） ======
def parse_page_spec(spec: str, max_pages: int) -> List[int]:
    """
    '1,3,5-7' などを 0始まりのインデックスへ。
    全角数字/カンマ/ハイフン/長音も許可（例：'１，３，５－７' '1ー3' '1—3' '1–3' '1―3'）
    """
    s = (spec or "").strip()
    if not s:
        return []
    # 全角→半角に正規化
    s = unicodedata.normalize("NFKC", s)
    # 日本語カンマなどを半角カンマに、各種ダッシュを半角ハイフンに
    s = s.replace("，", ",").replace("、", ",")
    for dash in ["－", "ー", "―", "—", "–"]:
        s = s.replace(dash, "-")
    # 分割
    out = set()
    parts = [p.strip() for p in s.split(",") if p.strip()]
    for p in parts:
        if "-" in p:
            a, b = p.split("-", 1)
            try:
                ia = int(a); ib = int(b)
                lo, hi = min(ia, ib), max(ia, ib)
                lo = max(1, lo); hi = min(max_pages, hi)
                for n in range(lo, hi + 1):
                    out.add(n - 1)
            except ValueError:
                # 無効トークンは無視
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

# ========== Azure行の座標 ==========
def line_xy(line_obj: Any) -> Tuple[float, float]:
    poly = getattr(line_obj, "polygon", None) or getattr(line_obj, "bounding_polygon", None)
    if not poly: return (0.0, 0.0)
    xs, ys = [], []
    for p in poly:
        x = getattr(p, "x", None); y = getattr(p, "y", None)
        if x is None and isinstance(p, dict):
            x = p.get("x", 0.0); y = p.get("y", 0.0)
        xs.append(float(x)); ys.append(float(y))
    return (min(xs or [0.0]), min(ys or [0.0]))

# ========== Word生成 ==========
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
st.title("📄 Document Intelligence OCR（Azure）— ページ指定/バッチ/GPT/Word/状態保持/ログ強化")

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
        "DICT_BLOB": DICT_FILE, "UNTRAINED_BLOB": UNTRAINED_FILE, "TRAINED_BLOB": TRAINED_FILE,
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
ocr_timeout = st.sidebar.slider("OCRタイムアウト（秒）", 10, 120, 45, step=5)
batch_size_override = st.sidebar.number_input("バッチサイズ上書き", 1, 20, value=BATCH_SIZE_DEFAULT)
use_cache = st.sidebar.checkbox("OCRキャッシュ（実験）", value=False)
debug_log = st.sidebar.checkbox("🔍 詳細ログ", value=True)

# 辞書プレビュー（手動再読込）
dict_preview_box = st.sidebar.container()
with dict_preview_box:
    st.subheader("📖 現在の辞書（プレビュー）")
    st.json(load_json_any(DICT_FILE))
if st.sidebar.button("🔄 辞書プレビューを再読込", type="secondary"):
    with dict_preview_box:
        st.subheader("📖 現在の辞書（プレビュー）")
        st.json(load_json_any(DICT_FILE))

# ===================== アップロード保持 =====================
if "file_bytes" not in st.session_state:
    st.session_state["file_bytes"] = None
    st.session_state["file_name"] = None
    st.session_state["file_mime"] = None
    st.session_state["is_pdf"] = False
    st.session_state["dpi"] = 200
    st.session_state["page_indices"] = []
    st.session_state["ran"] = False

uploaded = st.file_uploader("画像またはPDFをアップロードしてください", type=["jpg", "jpeg", "png", "pdf"], key="uploader")
if uploaded is not None:
    st.session_state["file_bytes"] = uploaded.getvalue()
    st.session_state["file_name"] = uploaded.name
    st.session_state["file_mime"] = uploaded.type
    st.session_state["is_pdf"] = (uploaded.type == "application/pdf") or uploaded.name.lower().endswith(".pdf")
    # 既存状態の初期化
    for k in list(st.session_state.keys()):
        if k.startswith("ocr_") or k.startswith("gpt_") or k.startswith("edit_"):
            del st.session_state[k]
    st.session_state["page_indices"] = []
    st.session_state["ran"] = False

file_bytes = st.session_state["file_bytes"]
if not file_bytes:
    st.info("📂 ここにファイルをアップロードしてください")
    st.stop()

ImageFile.LOAD_TRUNCATED_IMAGES = True
is_input_pdf = st.session_state["is_pdf"]

# ====== サイドバー：実行ステータス（常時表示で“動いてない”を可視化） ======
st.sidebar.markdown("### 📊 実行ステータス")
st.sidebar.write({
    "ran": bool(st.session_state.get("ran")),
    "has_file": bool(file_bytes),
    "is_pdf": bool(is_input_pdf),
    "saved_indices": st.session_state.get("page_indices", []),
    "dpi": st.session_state.get("dpi", 200),
})

# ===================== OCRコア（手動ポーリング） =====================
def _ocr_polling(png_bytes: bytes, timeout_sec: float) -> dict:
    poller = client.begin_analyze_document("prebuilt-read", document=io.BytesIO(png_bytes))
    t0 = time.perf_counter()
    status = st.empty()
    while True:
        if poller.done():
            break
        elapsed = time.perf_counter() - t0
        if elapsed > float(timeout_sec):
            status.empty()
            raise TimeoutError(f"Azure OCR timeout after {elapsed:.1f}s")
        status.info(f"Azure OCR 実行中… {elapsed:.1f}s / {timeout_sec:.0f}s")
        time.sleep(0.3)
    status.empty()

    result = poller.result()
    doc_page = result.pages[0] if getattr(result, "pages", None) else None
    if not doc_page:
        return {"pw": 1.0, "ph": 1.0, "lines": [], "raw": getattr(result, "content", "")}

    lines = []
    for ln in getattr(doc_page, "lines", []) or []:
        x, y = line_xy(ln)
        lines.append({"content": ln.content, "x": float(x), "y": float(y)})

    return {
        "pw": float(getattr(doc_page, "width", 1.0) or 1.0),
        "ph": float(getattr(doc_page, "height", 1.0) or 1.0),
        "lines": lines,
        "raw": getattr(result, "content", "") or "\n".join([l["content"] for l in lines]),
    }

@st.cache_data(show_spinner=False)
def _ocr_cached(digest: str, png_bytes: bytes, timeout_sec: float) -> dict:
    return _ocr_polling(png_bytes, timeout_sec)

def _ocr_dispatch(png_bytes: bytes, timeout_sec: float, use_cache: bool) -> dict:
    if use_cache:
        digest = hashlib.md5(png_bytes).hexdigest()
        return _ocr_cached(digest, png_bytes, timeout_sec)
    else:
        return _ocr_polling(png_bytes, timeout_sec)

# ===================== ページ選択（常時UI＋実行ボタン） =====================
if is_input_pdf:
    # PDFページ数
    try:
        pdf_for_count = pdfium.PdfDocument(io.BytesIO(file_bytes))
        total_pages = len(pdf_for_count)
        st.info(f"📘 このPDFは {total_pages} ページあります。")
    except Exception as e:
        st.exception(e); st.stop()

    st.subheader("▶ OCRするページを選択")
    col1, col2 = st.columns([2,1])
    with col1:
        select_mode = st.radio(
            "選択方法",
            options=["全ページ", "範囲指定", "ページ番号指定（例: 1,3,5-7 / １，３，５－７）"],
            index=1 if total_pages > 1 else 0,
            horizontal=True
        )
    with col2:
        dpi = st.slider("DPI", 72, 300, value=st.session_state.get("dpi", 200), step=4)
        st.session_state["dpi"] = dpi

    if select_mode == "範囲指定" and total_pages > 1:
        start, end = st.slider("処理するページ範囲（1始まり）", 1, total_pages, (1, min(total_pages, 5)))
        chosen_indices = list(range(start - 1, end))
    elif select_mode == "ページ番号指定（例: 1,3,5-7 / １，３，５－７）":
        spec_default = "1-3" if total_pages >= 3 else "1"
        spec = st.text_input("ページ番号（カンマ区切り、範囲はハイフン。全角OK）", value=spec_default)
        chosen_indices = parse_page_spec(spec, total_pages)
    else:
        chosen_indices = list(range(total_pages))

    st.caption(f"選択中: {', '.join(str(i+1) for i in chosen_indices) if chosen_indices else '(なし)'} / DPI={dpi}")

    run_clicked = st.button("▶ この設定でOCRを実行", type="primary")
   # 1) クリックされたか、過去に実行済み（ran=True）なら走る
# 2) さらに、page_indices が空なら今回の chosen_indices を採用
if (run_clicked or st.session_state.get("ran")) and chosen_indices:
    if not st.session_state.get("page_indices"):
        st.session_state["page_indices"] = chosen_indices
    st.session_state["ran"] = True
else:
    # 選択が空 or まだ実行ボタンが押されていない
    if not chosen_indices:
        st.error("選択されたページが空です。『全ページ』に切り替えるか、範囲/ページ番号を入れてください。")
    else:
        st.warning("実行待ちです。『▶ この設定でOCRを実行』を押してください。")
    st.stop()

    dpi = st.session_state.get("dpi", 200)
    EFFECTIVE_BATCH = int(batch_size_override) if batch_size_override else BATCH_SIZE_DEFAULT
    total_to_process = len(chosen_indices)
    progress = st.progress(0.0)
    status_area = st.empty()
    all_corrected_texts: List[str] = []
    pages_layout: List[Dict[str, Any]] = []
    done = 0

    st.write("### ▶ 実行開始")
    st.write(f"🧪 ページ: {', '.join(str(i+1) for i in chosen_indices)} / DPI={dpi} / バッチ={EFFECTIVE_BATCH}")

    for batch_no, batch_indices in enumerate(chunked(chosen_indices, EFFECTIVE_BATCH), start=1):
        status_area.info(f"🔄 バッチ {batch_no} / {((total_to_process - 1) // EFFECTIVE_BATCH) + 1} （ページ: {', '.join(str(i+1) for i in batch_indices)}）")
        try:
            st.write(f"🖼 レンダリング開始（{len(batch_indices)}ページ）...")
            pages, page_numbers = render_pdf_selected_pages(file_bytes, batch_indices, dpi=dpi)
            st.write(f"✅ レンダリング完了: {len(pages)}ページ")
        except Exception as e:
            st.exception(e); st.stop()

        for page_img, page_num in zip(pages, page_numbers):
            st.write(f"## ページ {page_num}")
            clean_img = remove_red_stamp(page_img)
            st.image(clean_img, caption=f"元ファイル (ページ {page_num})", use_container_width=True)

            buf = io.BytesIO(); clean_img.save(buf, format="PNG"); png_bytes = buf.getvalue()

            with st.spinner("OCRを実行中..."):
                t0 = time.perf_counter()
                try:
                    cached = _ocr_dispatch(png_bytes, ocr_timeout, use_cache)
                except Exception as e:
                    st.error(f"OCRに失敗：{e}")
                    st.caption(f"OCR実行時間: {time.perf_counter() - t0:.1f}s")
                    continue
                elapsed = time.perf_counter() - t0
                st.caption(f"OCR実行時間: {elapsed:.1f}s")

            azure_lines = cached.get("lines") or []
            default_text = "\n".join([ln["content"] for ln in azure_lines]) if azure_lines else (cached.get("raw") or "")
            if not default_text.strip():
                st.warning("OCRは成功しましたがテキストが空でした。DPIや画像品質を見直してください。")

            dictionary = load_json_any(DICT_FILE)
            gpt_checked_text = default_text if skip_gpt else gpt_fix_text(default_text, dictionary)

            ocr_key = f"ocr_{page_num}"
            gpt_key = f"gpt_{page_num}"
            edit_key = f"edit_{page_num}"
            if ocr_key not in st.session_state: st.session_state[ocr_key] = default_text
            if gpt_key not in st.session_state: st.session_state[gpt_key] = gpt_checked_text
            if edit_key not in st.session_state: st.session_state[edit_key] = gpt_checked_text

            tab2, tab3, tab4 = st.tabs(["🖨️ OCRテキスト", "🤖 GPT補正", "✍️ 手作業修正"])
            with tab2:
                st.text_area(f"OCRテキスト（ページ {page_num}）", height=320, key=ocr_key)
            with tab3:
                st.text_area(f"GPT補正（ページ {page_num}）", height=320, key=gpt_key)
            with tab4:
                st.text_area(f"手作業修正（ページ {page_num}）", height=320, key=edit_key)
                if st.button(f"修正を保存 (ページ {page_num})", key=f"save_{page_num}"):
                    corrected_text_current = st.session_state.get(edit_key, gpt_checked_text)
                    learned = learn_charwise_with_missing(st.session_state.get(ocr_key, default_text), corrected_text_current)
                    if learned:
                        update_dictionary_and_untrained(learned)
                        st.success(f"辞書と学習候補に {len(learned)} 件を追加しました！")
                    else:
                        st.info("修正が検出されませんでした。")

            final_text_page = (
                st.session_state.get(edit_key)
                or st.session_state.get(gpt_key)
                or gpt_checked_text
                or default_text
            ).strip()
            all_corrected_texts.append(final_text_page)

            gpt_lines = final_text_page.splitlines()
            lines_for_layout = []
            for i, ln in enumerate(azure_lines):
                x, y = ln["x"], ln["y"]
                text_for_line = gpt_lines[i] if i < len(gpt_lines) else ln["content"]
                lines_for_layout.append({"text": text_for_line, "x": x, "y": y})
            pages_layout.append({
                "page_width": cached["pw"], "page_height": cached["ph"], "unit": "pixel",
                "lines": lines_for_layout
            })

            done += 1; progress.progress(done / total_to_process)

        del pages, page_numbers
        gc.collect()

    status_area.success("✅ すべてのページの処理が完了しました。")

    if all_corrected_texts:
        joined_txt = "\n\n".join(all_corrected_texts)
        st.download_button(
            "📥 補正テキストをダウンロード（TXT, ページ見出しなし）",
            data=joined_txt.encode("utf-8"), file_name="ocr_corrected.txt", mime="text/plain"
        )
    if pages_layout:
        try:
            st.download_button(
                "📥 Word（.docx：レイアウト近似）",
                data=build_docx_from_layout(pages_layout),
                file_name="ocr_layout.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
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

    st.caption("単一画像としてOCRします。")
    run_img = st.button("▶ この画像でOCRを実行", type="primary")
    if not run_img and not st.session_state.get("ran"):
        st.stop()
    st.session_state["ran"] = True

    all_corrected_texts: List[str] = []
    pages_layout: List[Dict[str, Any]] = []

    for page_img, page_num in zip(pages, page_numbers):
        st.write(f"## ページ {page_num}")
        clean_img = remove_red_stamp(page_img)
        st.image(clean_img, caption=f"元ファイル (ページ {page_num})", use_container_width=True)

        buf = io.BytesIO(); clean_img.save(buf, format="PNG"); png_bytes = buf.getvalue()

        with st.spinner("OCRを実行中..."):
            t0 = time.perf_counter()
            try:
                cached = _ocr_dispatch(png_bytes, ocr_timeout, use_cache)
            except Exception as e:
                st.error(f"OCRに失敗：{e}")
                st.caption(f"OCR実行時間: {time.perf_counter() - t0:.1f}s")
                continue
            elapsed = time.perf_counter() - t0
            st.caption(f"OCR実行時間: {elapsed:.1f}s")

        azure_lines = cached.get("lines") or []
        default_text = "\n".join([ln["content"] for ln in azure_lines]) if azure_lines else (cached.get("raw") or "")
        if not default_text.strip():
            st.warning("OCRは成功しましたがテキストが空でした。DPIや画像品質を見直してください。")

        dictionary = load_json_any(DICT_FILE)
        gpt_checked_text = default_text if skip_gpt else gpt_fix_text(default_text, dictionary)

        ocr_key = f"ocr_{page_num}"; gpt_key = f"gpt_{page_num}"; edit_key = f"edit_{page_num}"
        if ocr_key not in st.session_state: st.session_state[ocr_key] = default_text
        if gpt_key not in st.session_state: st.session_state[gpt_key] = gpt_checked_text
        if edit_key not in st.session_state: st.session_state[edit_key] = gpt_checked_text

        tab2, tab3, tab4 = st.tabs(["🖨️ OCRテキスト", "🤖 GPT補正", "✍️ 手作業修正"])
        with tab2: st.text_area(f"OCRテキスト（ページ {page_num}）", height=320, key=ocr_key)
        with tab3: st.text_area(f"GPT補正（ページ {page_num}）", height=320, key=gpt_key)
        with tab4:
            st.text_area(f"手作業修正（ページ {page_num}）", height=320, key=edit_key)
            if st.button(f"修正を保存 (ページ {page_num})", key=f"save_{page_num}"):
                corrected_text_current = st.session_state.get(edit_key, gpt_checked_text)
                learned = learn_charwise_with_missing(st.session_state.get(ocr_key, default_text), corrected_text_current)
                if learned:
                    update_dictionary_and_untrained(learned)
                    st.success(f"辞書と学習候補に {len(learned)} 件を追加しました！")
                else:
                    st.info("修正が検出されませんでした。")

        final_text_page = (st.session_state.get(edit_key) or st.session_state.get(gpt_key) or gpt_checked_text or default_text).strip()
        all_corrected_texts.append(final_text_page)

        gpt_lines = final_text_page.splitlines()
        lines_for_layout = []
        for i, ln in enumerate(azure_lines):
            x, y = ln["x"], ln["y"]
            text_for_line = gpt_lines[i] if i < len(gpt_lines) else ln["content"]
            lines_for_layout.append({"text": text_for_line, "x": x, "y": y})
        pages_layout.append({"page_width": cached["pw"], "page_height": cached["ph"], "unit": "pixel", "lines": lines_for_layout})

    if all_corrected_texts:
        joined_txt = "\n\n".join(all_corrected_texts)
        st.download_button("📥 補正テキストをダウンロード（TXT, ページ見出しなし）",
                           data=joined_txt.encode("utf-8"), file_name="ocr_corrected.txt", mime="text/plain")
    if pages_layout:
        try:
            st.download_button("📥 Word（.docx：レイアウト近似）",
                               data=build_docx_from_layout(pages_layout),
                               file_name="ocr_layout.docx",
                               mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        except Exception as e:
            st.warning(f"Word出力に失敗しました：{e}")

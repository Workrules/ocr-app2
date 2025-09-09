# dummy line
import os
import json
import struct
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

# ========= 設定 =========
RECORD_SIZE = 8096
ETL9B_DIR = "./ETL9B"
ETL_OUT_DIR = "./etl9b_confusion"
CONF_FILE = "untrained_confusions.json"   # 誤読リスト（誤字→正字）
TRAINED_FILE = "trained_confusions.json"  # 学習済リスト
MODEL_FILE = "char_classifier_resnet18.pt"

# ========= ユーティリティ =========
def load_json(path):
    return json.load(open(path, "r", encoding="utf-8")) if os.path.exists(path) else {}

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ========= 誤字・正字リスト管理 =========
def get_untrained_chars():
    conf = load_json(CONF_FILE)      # 誤字→正字のマップ
    trained = load_json(TRAINED_FILE)

    # まだ学習していない誤字
    untrained = {ch: conf[ch] for ch in conf if ch not in trained}

    # 学習対象 = 誤字と正字の両方
    target_chars = set()
    for wrong, right in untrained.items():
        target_chars.add(wrong)
        target_chars.add(right)

    return target_chars, conf, trained

# ========= ETL9B 読み取り =========
def jis_to_char(jis):
    ku = (jis >> 8) & 0xFF
    ten = jis & 0xFF
    if ku < 0x21 or ten < 0x21:
        return None
    jis_bytes = bytes([ku | 0x80, ten | 0x80])
    try:
        return jis_bytes.decode("euc_jp")
    except:
        return None

def read_record_ETL9B(f):
    s = f.read(RECORD_SIZE)
    if not s or len(s) < RECORD_SIZE:
        return None
    jis_code = struct.unpack(">H", s[4:6])[0]  # JISコードはoffset=4
    img = np.frombuffer(s[-4032:], dtype=np.uint8).reshape(63, 64)
    return jis_code, img

def split_record_image(img, rows=7, cols=7):
    """63x64の画像を7x7=49枚に分割"""
    h, w = img.shape
    sub_h, sub_w = h // rows, w // cols
    sub_imgs = []
    for r in range(rows):
        for c in range(cols):
            sub_imgs.append(img[r*sub_h:(r+1)*sub_h, c*sub_w:(c+1)*sub_w])
    return sub_imgs

def extract_etl9b_for_chars(target_chars, out_dir=ETL_OUT_DIR) -> int:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    count = 0
    for fn in sorted(os.listdir(ETL9B_DIR)):
        if not fn.startswith("ETL9B_"):
            continue
        with open(os.path.join(ETL9B_DIR, fn), "rb") as f:
            while True:
                rec = read_record_ETL9B(f)
                if rec is None:
                    break
                jis, img = rec
                ch = jis_to_char(jis)
                if ch and ch in target_chars:
                    out_ch = Path(out_dir) / ch
                    out_ch.mkdir(parents=True, exist_ok=True)
                    sub_imgs = split_record_image(img)
                    for i, sub in enumerate(sub_imgs):
                        im = Image.fromarray(255 - sub)  # 白黒反転
                        fname = f"{fn}_{f.tell()}_{i}.png"
                        im.save(out_ch / fname)
                        count += 1
    return count

# ========= PyTorch Dataset =========
class CharImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.transform = transform
        for cls in self.classes:
            for img_file in os.listdir(os.path.join(root_dir, cls)):
                self.samples.append(os.path.join(root_dir, cls, img_file))
                self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

# ========= モデル学習 =========
def train_model(data_dir, num_epochs=10):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])
    dataset = CharImageDataset(data_dir, transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, len(dataset.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_acc = 0.0
    for epoch in range(num_epochs):
        correct = 0
        total = 0
        for imgs, labels in loader:
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        acc = correct / total
        print(f"[Epoch {epoch+1}/{num_epochs}] val_acc={acc:.3f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), MODEL_FILE)
    print(f"✅ 最良モデル保存: {MODEL_FILE} (val_acc={best_acc:.3f})")

# ========= メイン処理 =========
def main():
    target_chars, conf, trained = get_untrained_chars()
    print("🎯 今回学習対象:", sorted(target_chars))

    extracted = extract_etl9b_for_chars(target_chars)
    print(f"📂 ETL9Bから抽出: {extracted} 枚")

    if extracted > 0:
        train_model(ETL_OUT_DIR)
        # 学習済リストを更新
        for wrong, right in conf.items():
            trained[wrong] = right
        save_json(trained, TRAINED_FILE)
        print("✅ 学習済みリストを更新しました。")
    else:
        print("❗対象クラスの画像が集まりませんでした。")

if __name__ == "__main__":
    main()

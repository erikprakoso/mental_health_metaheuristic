# ================================
# ENHANCED BERT-BiLSTM + METAHEURISTIC OPTIMIZATION
# Base model F1: 0.9328
# Goal: Use GWO/MVO/SCA to find best hyperparameters
# ================================

import sys
import subprocess
import os


# ══════════════════════════════════════════════════════════════════════════════
# FIX: PyTorch + torchvision + P100 (SM 6.0) CUDA Compatibility
# ══════════════════════════════════════════════════════════════════════════════
def _get_installed_version(pkg):
    """Return (major, minor) of an installed package via pip show (no import)."""
    try:
        out = subprocess.check_output(
            [sys.executable, "-m", "pip", "show", pkg], stderr=subprocess.DEVNULL
        ).decode()
        for line in out.splitlines():
            if line.startswith("Version:"):
                ver_str = line.split(":", 1)[1].strip().split("+")[0]
                parts = ver_str.split(".")
                return tuple(int(x) for x in parts[:2])
    except Exception:
        pass
    return None


# ==============================================================================
# FIX: PyTorch + torchvision + P100 (SM 6.0) Compatibility (No Restart Version)
# ==============================================================================
def _fix_torch_for_p100():
    try:
        # 1. Cek SM capability tanpa import torch (lewat nvidia-smi)
        gpu_info = (
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=compute_cap",
                    "--format=csv,noheader,nounits",
                ]
            )
            .decode()
            .strip()
        )
        is_sm60 = "6.0" in gpu_info
    except:
        is_sm60 = False

    torch_ver = _get_installed_version("torch") or (0, 0)
    tv_ver = _get_installed_version("torchvision") or (0, 0)

    # Kita hanya butuh fix jika di P100 (SM 6.0) dan versi torch bukan 2.4.x
    need_fix = is_sm60 and (torch_ver != (2, 4))

    if need_fix:
        print(
            f"[COMPAT] Detected Tesla P100 (SM 6.0) with incompatible torch {torch_ver}"
        )
        print("[COMPAT] Installing torch 2.4.1 + torchvision 0.19.1...")
        subprocess.check_call(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--quiet",
                "torch==2.4.1",
                "torchvision==0.19.1",
                "torchaudio==2.4.1",
                "--index-url",
                "https://download.pytorch.org/whl/cu121",
            ]
        )
        print("[COMPAT] Installation complete. Proceeding without restart.")
    else:
        print(
            f"[COMPAT] Environment OK (GPU SM {gpu_info if 'gpu_info' in locals() else 'N/A'}, torch {torch_ver})"
        )


# PENTING: Panggil fungsi ini di paling atas, SEBELUM ada import torch apapun di level global
_fix_torch_for_p100()

# ── Install pyMetaheuristic if needed ──────────────────────────────────────────
try:
    from pyMetaheuristic.algorithm import (
        grey_wolf_optimizer,
        muti_verse_optimizer,
        sine_cosine_algorithm,
    )
except ImportError:
    print("Installing pyMetaheuristic...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pyMetaheuristic"])
    from pyMetaheuristic.algorithm import (
        grey_wolf_optimizer,
        muti_verse_optimizer,
        sine_cosine_algorithm,
    )


import numpy as np
import pandas as pd
import re
import emoji
from textblob import TextBlob
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import AutoTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
import random
import warnings

warnings.filterwarnings("ignore")

# ── Device setup ───────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"CUDA: {torch.version.cuda} | Compute: {torch.cuda.get_device_capability(0)}"
    )
    # Workaround for 'no kernel image' error (P100 + PyTorch mismatch)
    torch.backends.cudnn.enabled = False


# ── Reproducibility ────────────────────────────────────────────────────────────
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


set_seed(42)


# ══════════════════════════════════════════════════════════════════════════════
# PREPROCESSING  (same as original)
# ══════════════════════════════════════════════════════════════════════════════
class TextAugmenter:
    def __init__(self):
        self.synonym_dict = {
            "sad": ["unhappy", "sorrowful", "depressed", "down"],
            "happy": ["joyful", "cheerful", "glad", "pleased"],
            "tired": ["exhausted", "weary", "fatigued", "drained"],
            "lonely": ["alone", "isolated", "solitary"],
            "angry": ["mad", "furious", "upset", "frustrated"],
            "scared": ["afraid", "frightened", "terrified", "fearful"],
            "worried": ["anxious", "concerned", "troubled", "uneasy"],
            "bad": ["terrible", "awful", "horrible", "poor"],
            "good": ["great", "excellent", "wonderful", "nice"],
            "hate": ["despise", "dislike", "detest", "loathe"],
        }

    def synonym_replacement(self, text, n=2):
        words = text.split()
        if len(words) < 2:
            return text
        new_words = words.copy()
        for idx in random.sample(range(len(words)), min(n, len(words))):
            if words[idx] in self.synonym_dict:
                new_words[idx] = random.choice(self.synonym_dict[words[idx]])
        return " ".join(new_words)

    def random_deletion(self, text, p=0.1):
        words = text.split()
        if len(words) == 1:
            return text
        new_words = [w for w in words if random.random() > p]
        return " ".join(new_words) if new_words else random.choice(words)

    def random_swap(self, text, n=2):
        words = text.split()
        if len(words) < 2:
            return text
        new_words = words.copy()
        for _ in range(n):
            i1, i2 = random.sample(range(len(new_words)), 2)
            new_words[i1], new_words[i2] = new_words[i2], new_words[i1]
        return " ".join(new_words)

    def augment(self, text):
        aug_type = random.choice(["synonym", "delete", "swap", "none"])
        if aug_type == "synonym":
            return self.synonym_replacement(text)
        elif aug_type == "delete":
            return self.random_deletion(text)
        elif aug_type == "swap":
            return self.random_swap(text)
        return text


class EnhancedTextPreprocessor:
    def __init__(self):
        self.slang_dict = {
            "lol": "laugh out loud",
            "lmao": "laughing my ass off",
            "brb": "be right back",
            "omg": "oh my god",
            "wtf": "what the fuck",
            "imo": "in my opinion",
            "smh": "shaking my head",
            "idk": "i don't know",
            "tbh": "to be honest",
            "rn": "right now",
            "u": "you",
            "ur": "your",
            "r": "are",
            "gonna": "going to",
            "wanna": "want to",
            "gotta": "got to",
            "thx": "thanks",
            "plz": "please",
            "ppl": "people",
        }

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)
        text = re.sub(r"@\w+|#|<.*?>", "", text)
        text = text.lower()
        return re.sub(r"\s+", " ", text).strip()

    def handle_emoji(self, text):
        text = emoji.demojize(text, delimiters=(" ", " "))
        return re.sub(r"_", " ", text)

    def replace_slang(self, text):
        return " ".join([self.slang_dict.get(w, w) for w in text.split()])

    def preprocess_pipeline(self, text):
        text = self.handle_emoji(text)
        text = self.clean_text(text)
        text = self.replace_slang(text)
        return text


def extract_emotion_score(text):
    try:
        return float(TextBlob(text).sentiment.polarity)
    except:
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class DepressionDataset(Dataset):
    def __init__(self, texts, emotion_scores, labels, tokenizer, max_len=128):
        self.texts = texts
        self.emotion_scores = emotion_scores
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Use tokenizer as a callable (works with all HuggingFace tokenizer versions)
        encoding = self.tokenizer(
            str(self.texts[idx]),
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "emotion_score": torch.tensor(self.emotion_scores[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.float),
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODEL  (configurable hidden_size, dropout, n_layers for metaheuristic)
# ══════════════════════════════════════════════════════════════════════════════
class EnhancedBertBiLSTM(nn.Module):
    def __init__(self, hidden_size=256, dropout=0.3, n_layers=2, bert_model=None):
        super().__init__()
        # Re-use a shared, frozen BERT to avoid loading 400 MB per evaluation
        self.bert = (
            bert_model
            if bert_model is not None
            else BertModel.from_pretrained("bert-base-uncased")
        )

        self.bilstm = nn.LSTM(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if n_layers > 1 else 0,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            batch_first=True,
            dropout=0.1,
        )
        self.emotion_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
        )
        combined_dim = hidden_size * 2 + 64
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(combined_dim)
        self.fc1 = nn.Linear(combined_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, emotion_score):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        seq = bert_out.last_hidden_state  # (B, T, 768)
        lstm_out, _ = self.bilstm(seq)  # (B, T, hidden*2)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_out[:, -1, :]  # last timestep
        emotion_enc = self.emotion_encoder(emotion_score.unsqueeze(1))
        combined = self.layer_norm(torch.cat([pooled, emotion_enc], dim=1))
        x = self.dropout(combined)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        return self.sigmoid(self.fc3(x)).squeeze()


# ══════════════════════════════════════════════════════════════════════════════
# LOSS
# ══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING / EVALUATION HELPERS  (same as original)
# ══════════════════════════════════════════════════════════════════════════════
def train_epoch(model, loader, optimizer, criterion, scheduler):
    model.train()
    total_loss, correct = 0, 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        emo = batch["emotion_score"].to(device)
        lbl = batch["label"].to(device)

        optimizer.zero_grad()
        out = model(ids, mask, emo)
        loss = criterion(out, lbl)
        total_loss += loss.item()
        correct += ((out >= 0.5).float() == lbl).sum().item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
    return total_loss / len(loader), correct / len(loader.dataset)


def evaluate_model(model, loader, threshold=0.5):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            emo = batch["emotion_score"].to(device)
            lbl = batch["label"].to(device)
            out = model(ids, mask, emo)
            all_probs.extend(out.cpu().numpy())
            all_preds.extend((out >= threshold).float().cpu().numpy())
            all_labels.extend(lbl.cpu().numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    auc = roc_auc_score(all_labels, all_probs)
    return acc, prec, rec, f1, auc


def find_optimal_threshold(model, loader):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            out = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["emotion_score"].to(device),
            )
            all_probs.extend(out.cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    best_f1, best_thr = 0, 0.5
    for thr in np.arange(0.3, 0.7, 0.01):
        preds = (all_probs >= thr).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            all_labels, preds, average="binary", zero_division=0
        )
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


# ══════════════════════════════════════════════════════════════════════════════
# FULL TRAINING PIPELINE  (used both standalone and inside metaheuristic)
# ══════════════════════════════════════════════════════════════════════════════
def train_full_pipeline(
    train_loader,
    val_loader,
    lr=2e-5,
    dropout=0.3,
    hidden_size=256,
    n_layers=2,
    alpha=0.25,
    gamma=2.0,
    epochs=8,
    bert_model=None,
    verbose=True,
):
    model = EnhancedBertBiLSTM(
        hidden_size=hidden_size,
        dropout=dropout,
        n_layers=n_layers,
        bert_model=bert_model,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=len(train_loader), T_mult=2)
    criterion = FocalLoss(alpha=alpha, gamma=gamma)
    bce_loss = nn.BCELoss()

    best_f1, best_state, patience_count = 0, None, 0
    patience = 3  # tighter patience for metaheuristic runs

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, scheduler
        )
        acc, prec, rec, f1, auc = evaluate_model(model, val_loader)

        if verbose:
            print(
                f"  Epoch {epoch + 1}/{epochs} | "
                f"train_loss={train_loss:.4f} acc={train_acc:.4f} | "
                f"val_f1={f1:.4f} auc={auc:.4f}"
            )

        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                if verbose:
                    print(f"  Early stop at epoch {epoch + 1}")
                break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_f1


# ══════════════════════════════════════════════════════════════════════════════
# METAHEURISTIC FITNESS FUNCTION
# ══════════════════════════════════════════════════════════════════════════════
_TRAIN_LOADER = None  # set in __main__
_VAL_LOADER = None
_SHARED_BERT = None  # frozen BERT shared across evaluations


def fitness_function(params):
    """
    Minimization target for pyMetaheuristic.

    Search space
    ─────────────────────────────────────────────────────────
    params[0]  log10(lr)       [-5, -4]   → lr in [1e-5, 1e-4]
    params[1]  dropout         [0.1, 0.5]
    params[2]  hidden_size     [128, 512]  (cast to int)
    params[3]  n_layers        [1, 3]      (cast to int)
    params[4]  focal alpha     [0.1, 0.5]
    params[5]  focal gamma     [1.0, 3.0]
    ─────────────────────────────────────────────────────────
    Returns  1 - val_F1   (pyMetaheuristic minimises)
    """
    lr = float(10 ** params[0])
    dropout = float(params[1])
    hidden_size = int(params[2])
    if (hidden_size * 2) % 4 != 0:
        hidden_size = (hidden_size // 2) * 2
    if hidden_size < 128:
        hidden_size = 128
    n_layers = int(params[3])
    alpha = float(params[4])
    gamma = float(params[5])

    print(
        f"\n  → lr={lr:.2e}  dropout={dropout:.2f}  "
        f"hidden={hidden_size}  layers={n_layers}  "
        f"alpha={alpha:.2f}  gamma={gamma:.2f}"
    )

    try:
        _, best_f1 = train_full_pipeline(
            _TRAIN_LOADER,
            _VAL_LOADER,
            lr=lr,
            dropout=dropout,
            hidden_size=hidden_size,
            n_layers=n_layers,
            alpha=alpha,
            gamma=gamma,
            epochs=3,  # 3 epochs per evaluation keeps runtime sane
            bert_model=_SHARED_BERT,  # reuse frozen BERT
            verbose=False,
        )
    except Exception as e:
        print(f"  [WARN] evaluation failed: {e}")
        best_f1 = 0.0

    print(f"  → val_F1={best_f1:.4f}  fitness={1 - best_f1:.4f}")
    return 1.0 - best_f1


# ══════════════════════════════════════════════════════════════════════════════
# METAHEURISTIC RUNNER
# ══════════════════════════════════════════════════════════════════════════════
def run_metaheuristic(method="GWO", iterations=5, pop_size=5):
    """
    method     : 'GWO' | 'MVO' | 'SCA'
    iterations : number of search iterations
    pop_size   : population / pack size
    """

    # ── Dynamically ensure/import pyMetaheuristic local to the function ──
    try:
        from pyMetaheuristic.algorithm import (
            grey_wolf_optimizer,
            muti_verse_optimizer,
            sine_cosine_algorithm,
        )
    except ImportError:
        print("Installing pyMetaheuristic...")
        import subprocess, sys

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "pyMetaheuristic"]
        )
        from pyMetaheuristic.algorithm import (
            grey_wolf_optimizer,
            muti_verse_optimizer,
            sine_cosine_algorithm,
        )

    min_values = [-5.0, 0.1, 128, 1, 0.1, 1.0]
    max_values = [-4.0, 0.5, 512, 3, 0.5, 3.0]

    print(f"\n{'=' * 60}")
    print(f"  Metaheuristic: {method} | pop={pop_size} | iter={iterations}")
    print(f"{'=' * 60}")

    if method == "GWO":
        result = grey_wolf_optimizer(
            target_function=fitness_function,
            pack_size=pop_size,
            min_values=min_values,
            max_values=max_values,
            iterations=iterations,
            verbose=True,
        )
    elif method == "MVO":
        result = muti_verse_optimizer(
            target_function=fitness_function,
            universes=pop_size,
            min_values=min_values,
            max_values=max_values,
            iterations=iterations,
            verbose=True,
        )
    elif method == "SCA":
        result = sine_cosine_algorithm(
            target_function=fitness_function,
            solutions=pop_size,
            min_values=min_values,
            max_values=max_values,
            iterations=iterations,
            verbose=True,
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    return result


def print_best_params(result, method_name):
    best_f1 = 1.0 - result[-1]
    print(f"\n{'=' * 60}")
    print(f"  {method_name} Best Result")
    print(f"{'=' * 60}")
    print(f"  Best Val F1  : {best_f1:.4f}")
    print(f"  LR           : {10 ** result[0]:.2e}")
    print(f"  Dropout      : {result[1]:.2f}")
    print(f"  Hidden Size  : {int(result[2])}")
    print(f"  Layers       : {int(result[3])}")
    print(f"  Focal Alpha  : {result[4]:.2f}")
    print(f"  Focal Gamma  : {result[5]:.2f}")
    return {
        "method": method_name,
        "val_f1": best_f1,
        "lr": 10 ** result[0],
        "dropout": result[1],
        "hidden_size": int(result[2]),
        "n_layers": int(result[3]),
        "alpha": result[4],
        "gamma": result[5],
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Guard imports — ensures this block works even when run as a separate
    # Kaggle cell after a kernel restart
    import os, sys, random, warnings
    import numpy as np
    import pandas as pd

    warnings.filterwarnings("ignore")

    # ── 1. Data paths ──────────────────────────────────────────────────────────
    PATHS = [
        "/kaggle/input/datasets/infamouscoder/mental-health-social-media/Mental-Health-Twitter.csv",
        "/kaggle/input/mental-health-social-media/Mental-Health-Twitter.csv",
    ]
    data_path = next((p for p in PATHS if os.path.exists(p)), None)

    if data_path:
        print(f"Loading data: {data_path}")
        df = pd.read_csv(data_path)
    else:
        print("Data not found — using dummy data for smoke-test")
        df = pd.DataFrame(
            {
                "post_text": [
                    "i am so sad today",
                    "feeling great and happy",
                    "alone and lonely nights",
                    "i love my life",
                ]
                * 100,
                "label": [1, 0, 1, 0] * 100,
            }
        )

    # ── 2. Preprocess ──────────────────────────────────────────────────────────
    preprocessor = EnhancedTextPreprocessor()
    augmenter = TextAugmenter()

    df["cleaned_text"] = df["post_text"].apply(preprocessor.preprocess_pipeline)
    df["emotion_score"] = df["cleaned_text"].apply(extract_emotion_score)
    df = df[df["cleaned_text"].str.len() > 0].reset_index(drop=True)
    print(f"After preprocessing: {df.shape}")

    # Augmentation (same as original baseline)
    print("Applying data augmentation...")
    augmented_rows = []
    for _, row in df.iterrows():
        augmented_rows.append(
            {
                "cleaned_text": row["cleaned_text"],
                "emotion_score": row["emotion_score"],
                "label": row["label"],
            }
        )
        aug_text = augmenter.augment(row["cleaned_text"])
        augmented_rows.append(
            {
                "cleaned_text": aug_text,
                "emotion_score": extract_emotion_score(aug_text),
                "label": row["label"],
            }
        )
    df_aug = pd.DataFrame(augmented_rows)
    print(f"After augmentation: {df_aug.shape}")

    # ── 3. Split ───────────────────────────────────────────────────────────────
    tv_texts, test_texts, tv_scores, test_scores, tv_labels, test_labels = (
        train_test_split(
            df_aug["cleaned_text"].values,
            df_aug["emotion_score"].values,
            df_aug["label"].values,
            test_size=0.2,
            random_state=42,
            stratify=df_aug["label"],
        )
    )
    train_texts, val_texts, train_scores, val_scores, train_labels, val_labels = (
        train_test_split(
            tv_texts,
            tv_scores,
            tv_labels,
            test_size=0.2,
            random_state=42,
            stratify=tv_labels,
        )
    )
    print(
        f"Splits — train:{len(train_texts)} val:{len(val_texts)} test:{len(test_texts)}"
    )

    # ── 4. Tokenizer & DataLoaders ─────────────────────────────────────────────
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    BATCH_SIZE = 32
    train_ds = DepressionDataset(train_texts, train_scores, train_labels, tokenizer)
    val_ds = DepressionDataset(val_texts, val_scores, val_labels, tokenizer)
    test_ds = DepressionDataset(test_texts, test_scores, test_labels, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2
    )

    # ── 5. Shared frozen BERT (loaded once for all metaheuristic evaluations) ──
    print("Loading shared BERT model (frozen)...")
    _SHARED_BERT = BertModel.from_pretrained("bert-base-uncased").to(device)
    for p in _SHARED_BERT.parameters():
        p.requires_grad = False
    _TRAIN_LOADER = train_loader
    _VAL_LOADER = val_loader

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 1 — Baseline training (original hyperparameters, reproduce 0.9328)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 1 — BASELINE TRAINING (reproducing original results)")
    print("=" * 70)

    baseline_model, _ = train_full_pipeline(
        train_loader,
        val_loader,
        lr=2e-5,
        dropout=0.3,
        hidden_size=256,
        n_layers=2,
        alpha=0.25,
        gamma=2.0,
        epochs=8,
        bert_model=None,  # load fresh BERT with gradients for full training
        verbose=True,
    )

    opt_thr, _ = find_optimal_threshold(baseline_model, val_loader)
    acc, prec, rec, f1, auc = evaluate_model(
        baseline_model, test_loader, threshold=opt_thr
    )

    print(f"\nBaseline Test Results (threshold={opt_thr:.3f}):")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  AUC      : {auc:.4f}")
    BASELINE_F1 = f1

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 2 — Metaheuristic Hyperparameter Optimization
    # NOTE: Increase pop_size and iterations for better results on Kaggle.
    #       Suggested: pop_size=5, iterations=10
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2 — METAHEURISTIC HYPERPARAMETER OPTIMIZATION")
    print("=" * 70)

    results_summary = {}

    for METHOD in ["GWO", "MVO", "SCA"]:
        result = run_metaheuristic(
            method=METHOD,
            pop_size=3,  # ← increase for production run
            iterations=5,  # ← increase for production run
        )
        best = print_best_params(result, METHOD)
        results_summary[METHOD] = best

    # ── Best method ────────────────────────────────────────────────────────────
    best_method = max(results_summary, key=lambda m: results_summary[m]["val_f1"])
    bp = results_summary[best_method]

    print(f"\n{'=' * 70}")
    print(f"  BEST METHOD: {best_method}  (val_F1={bp['val_f1']:.4f})")
    print(f"{'=' * 70}")

    # ══════════════════════════════════════════════════════════════════════════
    # PHASE 3 — Final training with best hyperparameters (full epochs)
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3 — FINAL TRAINING WITH OPTIMIZED HYPERPARAMETERS")
    print("=" * 70)

    optimized_model, _ = train_full_pipeline(
        train_loader,
        val_loader,
        lr=bp["lr"],
        dropout=bp["dropout"],
        hidden_size=bp["hidden_size"],
        n_layers=bp["n_layers"],
        alpha=bp["alpha"],
        gamma=bp["gamma"],
        epochs=8,
        bert_model=None,  # full fine-tuning with gradients
        verbose=True,
    )

    opt_thr, _ = find_optimal_threshold(optimized_model, val_loader)
    acc2, prec2, rec2, f12, auc2 = evaluate_model(
        optimized_model, test_loader, threshold=opt_thr
    )

    # ── Final comparison ───────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(f"{'Metric':<12} {'Paper':<10} {'Baseline':<12} {'Optimized':<12} {'Δ'}")
    print(f"{'-' * 70}")
    paper_vals = dict(
        Accuracy=0.8895, Precision=0.8566, Recall=0.9381, F1=0.8955, AUC=0.8890
    )
    rows = [
        ("Accuracy", BASELINE_F1, acc2, acc2),  # reuse same var naming for simplicity
        ("Precision", prec, prec2, prec2),
        ("Recall", rec, rec2, rec2),
        ("F1", BASELINE_F1, f12, f12),
        ("AUC", auc, auc2, auc2),
    ]
    baseline_map = dict(
        zip(
            ["Accuracy", "Precision", "Recall", "F1", "AUC"],
            [acc, prec, rec, BASELINE_F1, auc],
        )
    )
    optimized_map = dict(
        zip(
            ["Accuracy", "Precision", "Recall", "F1", "AUC"],
            [acc2, prec2, rec2, f12, auc2],
        )
    )
    for metric in ["Accuracy", "Precision", "Recall", "F1", "AUC"]:
        b = baseline_map[metric]
        o = optimized_map[metric]
        p = paper_vals[metric]
        delta = (o - b) * 100
        gap = (o - p) * 100
        print(
            f"{metric:<12} {p:<10.4f} {b:<12.4f} {o:<12.4f} "
            f"{delta:+.2f}% (vs paper: {gap:+.2f}%)"
        )
    print(f"{'=' * 70}")

    print(f"""
Summary
───────
• Baseline F1  : {BASELINE_F1:.4f}
• Optimized F1 : {f12:.4f}
• Best method  : {best_method}
• Best params  : LR={bp["lr"]:.2e}  dropout={bp["dropout"]:.2f}
                 hidden={bp["hidden_size"]}  layers={bp["n_layers"]}
                 focal_alpha={bp["alpha"]:.2f}  focal_gamma={bp["gamma"]:.2f}
""")

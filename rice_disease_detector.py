

import os, random, math, json, copy, warnings, sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import v2 as T2
from PIL import Image
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, f1_score, accuracy_score)
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────
CFG = dict(
    # Paths
    data_root = r"C:\Users\TOTAN\Documents\crop_iq\Crop Diseases Dataset\Crop Diseases\Crop___Disease",          # set via CLI or change here
    output_dir    = "./rice_model_output",

    # Model
    backbone      = "efficientnet_b0",         # timm model name
    num_classes   = 4,
    img_size      = 224,
    dropout       = 0.4,

    # Training
    epochs        = 40,
    batch_size    = 16,
    lr            = 3e-4,
    weight_decay  = 1e-4,
    label_smooth  = 0.1,
    mixup_alpha   = 0.4,
    cutmix_alpha  = 1.0,
    warmup_epochs = 3,

    # Data split
    val_split     = 0.15,
    test_split    = 0.10,

    # Misc
    seed          = 42,
    num_workers   = 2,
    device        = "cuda" if torch.cuda.is_available() else "cpu",
    amp           = torch.cuda.is_available(),   # mixed precision
    early_stop    = 8,                           # patience epochs
    tta_steps     = 5,
)

CLASS_NAMES = ["Rice__Brown_Spot", "Rice__Healthy",
               "Rice__Leaf_Blast", "Rice__Neck_Blast"]

# ──────────────────────────────────────────────
# REPRODUCIBILITY
# ──────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

seed_everything(CFG["seed"])

# ──────────────────────────────────────────────
# DATASET
# ──────────────────────────────────────────────
def collect_samples(data_root: str):
    """Walk folder structure: data_root/ClassName/*.jpg|png"""
    root = Path(data_root)
    paths, labels = [], []
    for idx, cls in enumerate(CLASS_NAMES):
        cls_dir = root / cls
        if not cls_dir.exists():
            print(f"  [WARN] Missing class folder: {cls_dir}")
            continue
        imgs = list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.jpeg")) + \
               list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.JPG"))
        paths.extend(imgs)
        labels.extend([idx] * len(imgs))
        print(f"  {cls}: {len(imgs)} images")
    return paths, labels


def get_transforms(split: str, img_size: int):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3,
                                   saturation=0.3, hue=0.05),
            transforms.RandomRotation(30),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2)),
        ])
    else:  # val / test / tta
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

def get_tta_transforms(img_size: int):
    """5 deterministic augmentations for test-time augmentation."""
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    base = [transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)]
    return [
        transforms.Compose(base),
        transforms.Compose([transforms.Resize((img_size, img_size)),
                            transforms.RandomHorizontalFlip(p=1),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize((img_size, img_size)),
                            transforms.RandomVerticalFlip(p=1),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize((img_size + 32, img_size + 32)),
                            transforms.CenterCrop(img_size),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
        transforms.Compose([transforms.Resize((img_size, img_size)),
                            transforms.RandomRotation((90, 90)),
                            transforms.ToTensor(), transforms.Normalize(mean, std)]),
    ]


class RiceDataset(Dataset):
    def __init__(self, paths, labels, transform):
        self.paths     = paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img   = Image.open(self.paths[idx]).convert("RGB")
        img   = self.transform(img)
        label = self.labels[idx]
        return img, label


# ──────────────────────────────────────────────
# MODEL
# ──────────────────────────────────────────────
class RiceDiseaseCNN(nn.Module):
    def __init__(self, backbone: str, num_classes: int, dropout: float):
        super().__init__()
        # pretrained=True downloads ImageNet weights (set False if offline)
        try:
            self.backbone = timm.create_model(
                backbone, pretrained=True, num_classes=0, global_pool="avg")
        except Exception:
            print("  [INFO] Pretrained weights unavailable — training from scratch")
            self.backbone = timm.create_model(
                backbone, pretrained=False, num_classes=0, global_pool="avg")
        feat_dim = self.backbone.num_features

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.classifier(feats)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# ──────────────────────────────────────────────
# LOSS: Focal + Label Smoothing
# ──────────────────────────────────────────────
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, smooth=0.1, num_classes=4):
        super().__init__()
        self.gamma  = gamma
        self.smooth = smooth
        self.nc     = num_classes

    def forward(self, logits, targets):
        # label smoothing
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, self.smooth / (self.nc - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.smooth)

        log_prob = F.log_softmax(logits, dim=1)
        prob     = log_prob.exp()
        # focal weight
        focal_w  = (1 - prob) ** self.gamma
        loss     = -(smooth_targets * focal_w * log_prob).sum(dim=1).mean()
        return loss


# ──────────────────────────────────────────────
# MIXUP / CUTMIX
# ──────────────────────────────────────────────
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w   = int(W * cut_rat)
    cut_h   = int(H * cut_rat)
    cx      = np.random.randint(W)
    cy      = np.random.randint(H)
    x1 = max(cx - cut_w // 2, 0);  x2 = min(cx + cut_w // 2, W)
    y1 = max(cy - cut_h // 2, 0);  y2 = min(cy + cut_h // 2, H)
    return x1, y1, x2, y2


def mixup_cutmix(images, labels, num_classes, alpha_mix=0.4, alpha_cut=1.0):
    """Randomly apply either MixUp or CutMix."""
    use_cutmix = random.random() < 0.5
    alpha = alpha_cut if use_cutmix else alpha_mix
    lam   = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx   = torch.randperm(images.size(0), device=images.device)

    onehot = F.one_hot(labels, num_classes).float()
    mixed_labels = lam * onehot + (1 - lam) * onehot[idx]

    if use_cutmix:
        x1, y1, x2, y2 = rand_bbox(images.size(), lam)
        mixed_imgs = images.clone()
        mixed_imgs[:, :, x1:x2, y1:y2] = images[idx, :, x1:x2, y1:y2]
        lam_adj = 1 - (x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2))
        mixed_labels = lam_adj * onehot + (1 - lam_adj) * onehot[idx]
    else:
        mixed_imgs = lam * images + (1 - lam) * images[idx]

    return mixed_imgs, mixed_labels


def soft_cross_entropy(logits, soft_labels):
    return -(soft_labels * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


# ──────────────────────────────────────────────
# TRAINING UTILITIES
# ──────────────────────────────────────────────
def make_scheduler(optimizer, epochs, warmup_epochs, steps_per_epoch):
    total = epochs * steps_per_epoch
    warm  = warmup_epochs * steps_per_epoch

    def lr_lambda(step):
        if step < warm:
            return step / max(warm, 1)
        progress = (step - warm) / max(total - warm, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, scheduler, criterion,
                device, scaler, cfg):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        # MixUp / CutMix (50 % of batches)
        use_aug = random.random() < 0.5
        if use_aug:
            imgs, soft_labels = mixup_cutmix(
                imgs, labels, cfg["num_classes"],
                cfg["mixup_alpha"], cfg["cutmix_alpha"])

        optimizer.zero_grad()
        with torch.autocast(device_type=device, enabled=cfg["amp"]):
            logits = model(imgs)
            if use_aug:
                loss = soft_cross_entropy(logits, soft_labels)
            else:
                loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler:
            scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        preds       = logits.argmax(1)
        if not use_aug:
            correct += (preds == labels).sum().item()
            n       += imgs.size(0)

    avg_loss = total_loss / len(loader.dataset)
    acc      = correct / n if n > 0 else 0.0
    return avg_loss, acc


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, cfg):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.autocast(device_type=device, enabled=cfg["amp"]):
            logits = model(imgs)
            loss   = criterion(logits, labels)
        probs  = F.softmax(logits, dim=1)
        preds  = probs.argmax(1)
        total_loss += loss.item() * imgs.size(0)
        correct    += (preds == labels).sum().item()
        n          += imgs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (total_loss / len(loader.dataset),
            correct / n,
            np.array(all_preds),
            np.array(all_labels),
            np.array(all_probs))


# ──────────────────────────────────────────────
# GRAD-CAM
# ──────────────────────────────────────────────
class GradCAM:
    def __init__(self, model):
        self.model    = model
        self.grads    = None
        self.acts     = None
        self._hook()

    def _hook(self):
        # EfficientNet last conv block
        target = list(self.model.backbone.modules())[-3]   # heuristic
        target.register_forward_hook(lambda m, i, o: setattr(self, 'acts', o))
        target.register_full_backward_hook(lambda m, gi, go: setattr(self, 'grads', go[0]))

    def __call__(self, img_tensor, class_idx=None):
        self.model.eval()
        img_tensor = img_tensor.unsqueeze(0)
        logits     = self.model(img_tensor)
        if class_idx is None:
            class_idx = logits.argmax(1).item()
        self.model.zero_grad()
        logits[0, class_idx].backward()
        weights   = self.grads.mean(dim=[2, 3], keepdim=True)
        cam       = (weights * self.acts).sum(1, keepdim=True).relu()
        cam       = F.interpolate(cam, img_tensor.shape[-2:],
                                  mode="bilinear", align_corners=False)
        cam       = cam.squeeze().detach().cpu().numpy()
        cam       = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx


# ──────────────────────────────────────────────
# TTA INFERENCE
# ──────────────────────────────────────────────
@torch.no_grad()
def tta_predict(model, img_path: str, cfg: dict, device):
    tta_tfms = get_tta_transforms(cfg["img_size"])
    img      = Image.open(img_path).convert("RGB")
    probs    = []
    model.eval()
    for tfm in tta_tfms:
        tensor = tfm(img).unsqueeze(0).to(device)
        with torch.autocast(device_type=device, enabled=cfg["amp"]):
            logit = model(tensor)
        probs.append(F.softmax(logit, dim=1).cpu().numpy())
    mean_prob  = np.mean(probs, axis=0)[0]
    pred_cls   = mean_prob.argmax()
    confidence = mean_prob[pred_cls]
    return CLASS_NAMES[pred_cls], confidence, mean_prob


# ──────────────────────────────────────────────
# VISUALISATION HELPERS
# ──────────────────────────────────────────────
def plot_history(history, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in zip(axes,
                               [("train_loss","val_loss"), ("train_acc","val_acc")],
                               ["Loss", "Accuracy"]):
        ax.plot(history[key[0]], label="Train")
        ax.plot(history[key[1]], label="Val")
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/training_history.png", dpi=150)
    plt.close()


def plot_confusion(labels, preds, out_dir):
    cm = confusion_matrix(labels, preds)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, data, fmt, title in zip(
            axes, [cm, cm_norm], ["d", ".2f"],
            ["Confusion Matrix (counts)", "Confusion Matrix (normalised)"]):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues",
                    xticklabels=[c.split("__")[-1] for c in CLASS_NAMES],
                    yticklabels=[c.split("__")[-1] for c in CLASS_NAMES], ax=ax)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=150)
    plt.close()


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────
def build_dataset_from_demo_images(cfg):
    """
    When no full dataset folder exists, create a synthetic small dataset
    from the 6 demo images so the pipeline runs end-to-end for demonstration.
    """
    demo_imgs = sorted(Path("/mnt/user-data/uploads").glob("*.jpg"))
    print(f"  Found {len(demo_imgs)} demo images — synthetic expansion mode")

    # Assign round-robin labels
    paths, labels = [], []
    for i, p in enumerate(demo_imgs * 40):   # repeat to get ~240 samples
        paths.append(p)
        labels.append(i % cfg["num_classes"])
    return paths, labels


def main(cfg: dict):
    out = Path(cfg["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    device = cfg["device"]
    print(f"\n{'='*60}")
    print(f"  Rice Disease Detection — {cfg['backbone'].upper()}")
    print(f"  Device: {device.upper()}  |  AMP: {cfg['amp']}")
    print(f"{'='*60}\n")

    # ── DATA ──────────────────────────────────
    data_root = Path(cfg["data_root"])
    if data_root.exists():
        print("[DATA] Loading from dataset folder …")
        paths, labels = collect_samples(str(data_root))
    else:
        print("[DATA] Dataset folder not found — using demo images …")
        paths, labels = build_dataset_from_demo_images(cfg)

    print(f"  Total samples: {len(paths)}")

    paths  = [str(p) for p in paths]
    labels = list(labels)

    # Stratified split
    tr_p, tmp_p, tr_l, tmp_l = train_test_split(
        paths, labels, test_size=cfg["val_split"] + cfg["test_split"],
        stratify=labels, random_state=cfg["seed"])
    vl_p, te_p, vl_l, te_l   = train_test_split(
        tmp_p, tmp_l,
        test_size=cfg["test_split"] / (cfg["val_split"] + cfg["test_split"]),
        stratify=tmp_l, random_state=cfg["seed"])

    print(f"\n  Train: {len(tr_p)} | Val: {len(vl_p)} | Test: {len(te_p)}\n")

    # Weighted sampler (handle class imbalance)
    class_counts = np.bincount(tr_l, minlength=cfg["num_classes"]).astype(float)
    weights = [1.0 / class_counts[l] for l in tr_l]
    sampler = WeightedRandomSampler(weights, num_samples=len(tr_l), replacement=True)

    tfm_train = get_transforms("train", cfg["img_size"])
    tfm_val   = get_transforms("val",   cfg["img_size"])

    train_ds = RiceDataset(tr_p, tr_l, tfm_train)
    val_ds   = RiceDataset(vl_p, vl_l, tfm_val)
    test_ds  = RiceDataset(te_p, te_l, tfm_val)

    kw = dict(num_workers=cfg["num_workers"], pin_memory=(device == "cuda"))
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              sampler=sampler, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=cfg["batch_size"],
                              shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=cfg["batch_size"],
                              shuffle=False, **kw)

    # ── MODEL ─────────────────────────────────
    model = RiceDiseaseCNN(cfg["backbone"], cfg["num_classes"], cfg["dropout"])
    model = model.to(device)
    criterion = FocalLoss(gamma=2.0, smooth=cfg["label_smooth"],
                          num_classes=cfg["num_classes"])
    scaler    = torch.amp.GradScaler(enabled=cfg["amp"])

    # Phase 1: freeze backbone, train head only (3 epochs)
    model.freeze_backbone()
    head_params = [p for p in model.parameters() if p.requires_grad]
    opt1 = torch.optim.AdamW(head_params, lr=cfg["lr"] * 5,
                              weight_decay=cfg["weight_decay"])
    sched1 = make_scheduler(opt1, 3, 1, len(train_loader))

    print("[Phase 1] Warming up classifier head …")
    for ep in range(3):
        tr_loss, tr_acc = train_epoch(model, train_loader, opt1, sched1,
                                      criterion, device, scaler, cfg)
        vl_loss, vl_acc, *_ = eval_epoch(model, val_loader, criterion, device, cfg)
        print(f"  Ep {ep+1:02d}/03 | tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f}"
              f" | vl_loss={vl_loss:.4f} vl_acc={vl_acc:.3f}")

    # Phase 2: unfreeze all, fine-tune end-to-end
    model.unfreeze_backbone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    scheduler = make_scheduler(optimizer, cfg["epochs"], cfg["warmup_epochs"],
                               len(train_loader))

    history = dict(train_loss=[], val_loss=[], train_acc=[], val_acc=[])
    best_val_acc, patience_cnt = 0.0, 0
    best_state = None

    print(f"\n[Phase 2] End-to-end fine-tuning for {cfg['epochs']} epochs …")
    for ep in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, scheduler,
                                      criterion, device, scaler, cfg)
        vl_loss, vl_acc, vl_preds, vl_labels, _ = eval_epoch(
            model, val_loader, criterion, device, cfg)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        improved = vl_acc > best_val_acc
        mark     = " ★" if improved else ""
        print(f"  Ep {ep:02d}/{cfg['epochs']} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.3f} | "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.3f}{mark}")

        if improved:
            best_val_acc = vl_acc
            best_state   = copy.deepcopy(model.state_dict())
            torch.save(best_state, out / "best_model.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= cfg["early_stop"]:
                print(f"  Early stopping triggered (patience={cfg['early_stop']})")
                break

    # ── EVALUATION ────────────────────────────
    model.load_state_dict(best_state)
    plot_history(history, str(out))

    print("\n[TEST] Evaluating on held-out test set …")
    _, te_acc, te_preds, te_labels, te_probs = eval_epoch(
        model, test_loader, criterion, device, cfg)

    plot_confusion(te_labels, te_preds, str(out))

    report = classification_report(te_labels, te_preds,
                                   target_names=[c.split("__")[-1] for c in CLASS_NAMES],
                                   digits=4)
    print("\n" + report)

    # AUC
    try:
        auc = roc_auc_score(te_labels, te_probs, multi_class="ovr", average="macro")
        print(f"  Macro AUC: {auc:.4f}")
    except Exception:
        auc = None

    f1  = f1_score(te_labels, te_preds, average="macro")
    summary = dict(
        test_accuracy=round(float(te_acc), 4),
        macro_f1=round(float(f1), 4),
        macro_auc=round(float(auc), 4) if auc else None,
        best_val_accuracy=round(float(best_val_acc), 4),
        epochs_trained=len(history["train_loss"]) + 3,
    )
    with open(out / "results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  ✓ Test Accuracy : {te_acc:.4f}")
    print(f"  ✓ Macro F1      : {f1:.4f}")
    if auc:
        print(f"  ✓ Macro AUC     : {auc:.4f}")

    # ── ONNX EXPORT ───────────────────────────
    try:
        dummy = torch.randn(1, 3, cfg["img_size"], cfg["img_size"]).to(device)
        torch.onnx.export(model, dummy, str(out / "rice_model.onnx"),
                          input_names=["image"], output_names=["logits"],
                          opset_version=17,
                          dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}})
        print("  ✓ ONNX model exported: rice_model.onnx")
    except Exception as e:
        print(f"  [WARN] ONNX export failed: {e}")

    # ── SAMPLE INFERENCE ──────────────────────
    print("\n[INFERENCE] Running TTA on sample images …")
    demo_imgs = sorted(Path("/mnt/user-data/uploads").glob("*.jpg"))[:3]
    inf_results = []
    for p in demo_imgs:
        cls, conf, probs = tta_predict(model, str(p), cfg, device)
        short = cls.split("__")[-1]
        print(f"  {p.name:<35} → {short:<15} ({conf*100:.1f}%)")
        inf_results.append(dict(image=p.name, prediction=cls,
                                confidence=round(float(conf), 4),
                                probabilities={
                                    CLASS_NAMES[i].split("__")[-1]: round(float(probs[i]), 4)
                                    for i in range(len(CLASS_NAMES))}))

    with open(out / "sample_predictions.json", "w") as f:
        json.dump(inf_results, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  Output artefacts saved to: {out.resolve()}")
    print(f"  • best_model.pth")
    print(f"  • rice_model.onnx")
    print(f"  • training_history.png")
    print(f"  • confusion_matrix.png")
    print(f"  • results.json")
    print(f"  • sample_predictions.json")
    print(f"{'='*60}\n")

    return summary


if __name__ == "__main__":
    # Optionally pass data_root as first argument
    if len(sys.argv) > 1:
        CFG["data_root"] = sys.argv[1]
    main(CFG)
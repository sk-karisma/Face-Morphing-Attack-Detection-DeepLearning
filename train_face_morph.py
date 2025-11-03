# ==============================================================================
# 1. SETUP AND DEPENDENCY IMPORTS
# ==============================================================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from dataclasses import dataclass
import numpy as np
import pandas as pd
import glob
import logging
import sys
import clip
# --- Added imports for testing ---
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import csv

# Configure logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

print("Dependencies imported successfully.")

# ==============================================================================
# 2. DATA PATH MAPPING AND CSV GENERATION
# ==============================================================================

# This path must match your attached dataset's folder name.
DATASET_ROOT = "/kaggle/input/amsl-facemorphimagedataset"

# Map folder names to binary labels: 0 for Genuine, 1 for Morph
LABEL_MAP = {
    'londondb_genuine_smiling_passport-scale_15kb': 0,
    'londondb_genuine_neutral_passport-scale_15kb': 0,
    'londondb_morph_combined_alpha0_5_passport-s': 1,
}

# --- CSV GENERATION ---
all_data = []

print("Generating training CSV from dataset folders...")
for folder_name, label in LABEL_MAP.items():
    folder_path = os.path.join(DATASET_ROOT, folder_name)

    image_paths = glob.glob(os.path.join(folder_path, '**/*.jpg'), recursive=True)
    image_paths += glob.glob(os.path.join(folder_path, '**/*.png'), recursive=True)

    for path in image_paths:
        all_data.append({
            'image_path': path,
            'label': label
        })

df = pd.DataFrame(all_data)
if df.empty:
    raise ValueError(f"No images found. Check the DATASET_ROOT: {DATASET_ROOT} and folder names in LABEL_MAP. Found {len(all_data)} files.")

df_train = df.sample(frac=0.8, random_state=42).reset_index(drop=True)
df_val = df.drop(df_train.index).reset_index(drop=True)

df_train.to_csv("train_data.csv", index=False)
df_val.to_csv("val_data.csv", index=False)
print(f"Training data size: {len(df_train)}")
print(f"Validation data size: {len(df_val)}")

# ==============================================================================
# 3. PYTORCH CUSTOM DATASET
# ==============================================================================

class FaceMorphDataset(Dataset):
    """
    Custom Dataset reads paths from CSV and loads images.
    FIX: This class now correctly returns 3 items: (image, label, img_path)
    """
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_path = self.data_frame.loc[idx, 'image_path']
        image = Image.open(img_path).convert('RGB')
        label = self.data_frame.loc[idx, 'label']
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        # This return format now matches all training/testing loops
        return image, label, img_path


# ==============================================================================
# 4. CORE MADATION MODEL DEFINITIONS
# ==============================================================================

# --- Utility Mocks ---
class AverageMeter:
    def __init__(self): self.reset()
    def reset(self): self.avg, self.sum, self.count = 0, 0, 0
    def update(self, val, n=1): self.sum += val * n; self.count += n; self.avg = self.sum / self.count

def get_scheduler(optimizer_model, **kwargs):
    class MockScheduler:
        def __init__(self, optimizer): self.optimizer = optimizer
        def step(self): pass
        def get_last_lr(self): return [self.optimizer.param_groups[0]['lr']]
    return MockScheduler(optimizer_model)

# --- Model Definitions ---
class CLIP_Model(nn.Module):
    """CLIP backbone with robust LoRA-like parameter selection."""
    def __init__(self):
        super().__init__()
        self.model, _ = clip.load("ViT-B/16", device="cpu")
        self.encoder_output_dim = self.model.visual.output_dim

        for param in self.model.parameters():
            param.requires_grad = False

        trainable_count = 0
        for name, param in self.model.visual.named_parameters():
             if ('attn.q_proj' in name or 'attn.v_proj' in name):
                param.requires_grad = True
                trainable_count += param.numel()

        if trainable_count == 0:
            print("WARNING: Primary LoRA targets yielded ZERO parameters. Falling back to unfreezing final attention block.")
            for name, param in self.model.visual.named_parameters():
                if 'resblocks.11.attn' in name:
                    param.requires_grad = True
                    trainable_count += param.numel()

        if trainable_count == 0:
            print("ERROR: Fallback failed. Check model structure.")
        else:
            print(f"Total trainable parameters (LoRA-like): {trainable_count}")

    def encode_image(self, x):
        return self.model.encode_image(x)

class ClassificationHeader(nn.Module):
    """Binary classification head for the MAD task."""
    def __init__(self, feature_dim):
        super().__init__()
        self.fc = nn.Linear(feature_dim, 2)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, features, target):
        output = self.fc(features)
        loss_image = self.criterion(output, target.long())
        return output, loss_image

@dataclass
class Config:
    num_epoch: int = 5
    lr_model: float = 1e-5
    lr_header: float = 1e-4
    weight_decay: float = 0.05
    max_norm: float = 2.0
    scheduler_type: str = 'cosine'
    warmup: bool = True
    num_warmup_epochs: int = 5
    # --- Added fields for testing ---
    output_path: str = "./results"
    test_data_name: str = "validation_set"

# --- Added Helper Functions for Testing ---

def evaluate_mad_performance(scores, labels):
    """Calculates AUC and EER."""
    try:
        auc_score = roc_auc_score(labels, scores)
    except ValueError:
        auc_score = 0.5

    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr

    try:
        eer_interp = interp1d(fpr, fnr)
        eer = brentq(lambda x: eer_interp(x) - x, 0, 1)
    except (ValueError, RuntimeError):
        eer_idx = np.argmin(np.abs(fnr - fpr))
        eer = fpr[eer_idx]

    # Return placeholder values for other metrics
    return {
        "auc_score": auc_score,
        "eer": eer,
        "apcer_bpcer20": 0.0,
        "apcer_bpcer10": 0.0,
        "apcer_bpcer1": 0.0,
        "bpcer_apcer20": 0.0,
        "bpcer_apcer10": 0.0,
        "bpcer_apcer1": 0.0,
    }

def write_scores(img_paths, scores, labels, out_path):
    """Writes image paths, scores, and labels to a CSV file."""
    with open(out_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_path', 'score', 'label'])
        for path, score, label in zip(img_paths, scores, labels):
            writer.writerow([path, score, label])

# --- Single-GPU Trainer ---
class TrainerClip:
    def __init__(self, model, dataloader, val_dataloader, config, header):
        """
        FIX: Updated init to accept val_dataloader
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.header = header.to(self.device)
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader # ADDED
        self.config = config

        self.start_epoch = 0
        self.global_step = 0
        self.loss_log = AverageMeter()

        print(f"Trainer initialized. Device: {self.device}")

    def start_training(self):
        self.MAD_training()

    def MAD_training(self):
        config = self.config

        optimizer_model = optim.AdamW(
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
            betas=(0.9, 0.999),
            lr=config.lr_model, weight_decay=config.weight_decay
        )
        optimizer_header = optim.AdamW(
            params=self.header.parameters(), betas=(0.9, 0.999),
            lr=config.lr_header, weight_decay=config.weight_decay
        )

        scheduler_model = get_scheduler(optimizer_model=optimizer_model)
        scheduler_header = get_scheduler(optimizer_model=optimizer_header)

        for epoch in range(self.start_epoch, config.num_epoch):
            self.model.train()
            self.header.train()
            self.loss_log.reset()

            # FIX: Unpack 3 items from dataset
            for i, (images, target, _) in enumerate(self.dataloader):
                self.global_step += 1

                images = images.to(self.device)
                target = target.to(self.device)

                features_image = self.model.encode_image(images).float()

                _, loss_image = self.header(F.normalize(features_image, dim=-1), target)

                loss_image.backward()
                clip_grad_norm_(self.model.parameters(), max_norm=config.max_norm, norm_type=2)
                clip_grad_norm_(self.header.parameters(), max_norm=config.max_norm, norm_type=2)

                optimizer_model.step()
                optimizer_header.step()

                self.loss_log.update(loss_image.item(), images.size(0))

                optimizer_model.zero_grad()
                optimizer_header.zero_grad()

                if self.global_step % 10 == 0:
                    current_lr = optimizer_model.param_groups[0]['lr']
                    print(f"Epoch: {epoch}/{config.num_epoch}, Step: {self.global_step}, Loss: {self.loss_log.avg:.4f}, LR: {current_lr:.6f}")

            scheduler_model.step()
            scheduler_header.step()

            print(f"\n--- Epoch {epoch} Finished. Average Loss: {self.loss_log.avg:.4f} ---")

        print("Training (LoRA + Header) finished.")

    def MAD_training_header_only(self):
        """
        Trains *only* the classification header.
        """
        config = self.config

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        self.header.train()

        optimizer_header = optim.AdamW(
            params=self.header.parameters(), betas=(0.9, 0.999),
            lr=config.lr_header, weight_decay=config.weight_decay
        )

        scheduler_header = get_scheduler(optimizer_model=optimizer_header)

        self.global_step = 0
        print("\n--- Starting Training (Header Only) ---")

        for epoch in range(self.start_epoch, config.num_epoch):
            self.loss_log.reset()

            # FIX: Unpack 3 items from dataset
            for i, (images, target, _) in enumerate(self.dataloader):
                self.global_step += 1

                images = images.to(self.device)
                target = target.to(self.device)

                with torch.no_grad():
                    features_image = self.model.encode_image(images).float()

                _, loss_image = self.header(F.normalize(features_image, dim=-1), target)

                loss_image.backward()
                clip_grad_norm_(self.header.parameters(), max_norm=config.max_norm, norm_type=2)
                optimizer_header.step()
                self.loss_log.update(loss_image.item(), images.size(0))
                optimizer_header.zero_grad()

                if self.global_step % 10 == 0:
                    current_lr = optimizer_header.param_groups[0]['lr']
                    print(f"Epoch: {epoch}/{config.num_epoch}, Step: {self.global_step}, Loss: {self.loss_log.avg:.4f}, LR: {current_lr:.6f}")

            scheduler_header.step()
            print(f"\n--- Epoch {epoch} Finished. Average Loss: {self.loss_log.avg:.4f} ---")

        print("Training (Header Only) finished.")

    def test_clip(self):
        """
        Runs zero-shot testing using text prompts.
        """
        print("\n--- Starting Zero-Shot Testing (test_clip) ---")
        self.model.eval()
        self.header.eval()

        prompts = ["face image morphing attack", "bona-fide presentation"]

        try:
            text_inputs = torch.cat([clip.tokenize(p) for p in prompts]).to(self.device)
            text_features = self.model.model.encode_text(text_inputs)
        except Exception as e:
            print(f"Error tokenizing or encoding text: {e}.")
            return

        text_features /= text_features.norm(dim=-1, keepdim=True)

        raw_test_scores, gt_labels = [], []
        raw_test_img_pths = []

        with torch.no_grad():
            # FIX: Loop unpacks 3 items, matching dataset
            for i, (raw, labels, img_paths) in enumerate(self.val_dataloader):
                if i % 5 == 0:
                    print(f"Testing batch {i}/{len(self.val_dataloader)}")

                raw = raw.to(self.device)
                labels = labels.to(self.device)

                image_features = self.model.encode_image(raw).float()
                image_features /= image_features.norm(dim=-1, keepdim=True)

                logits_per_image = (100.0 * image_features @ text_features.T).softmax(
                    dim=-1
                )

                # Score is probability of "morphing attack" (prompt at index 0)
                raw_scores = logits_per_image[:, 0]

                raw_test_scores.append(raw_scores.cpu())
                gt_labels.append(labels.cpu())
                raw_test_img_pths.extend(img_paths)

        raw_test_scores = torch.cat(raw_test_scores).numpy()
        gt_labels = torch.cat(gt_labels).numpy()

        os.makedirs(self.config.output_path, exist_ok=True)

        out_path = os.path.join(self.config.output_path, self.config.test_data_name + ".csv")
        write_scores(raw_test_img_pths, raw_test_scores, gt_labels, out_path)
        print(f"Scores written to {out_path}")

        results_dict = evaluate_mad_performance(raw_test_scores, gt_labels)

        print(f"--- Testing Finished ---")
        print(f"Results for: {self.config.test_data_name}")

        txt_out_path = os.path.join(self.config.output_path, self.config.test_data_name + ".txt")
        with open(txt_out_path, "w") as f:
            for key, value in results_dict.items():
                print(f"{key.upper()}: {value:.4f}")
                f.write(f"{key.upper()}: {value:.4f}\n")
        print(f"Metrics written to {txt_out_path}")


# ==============================================================================
# 5. EXECUTION BLOCK
# ==============================================================================
if __name__ == '__main__':
    if not torch.cuda.is_available():
        print("CUDA not available. Please ensure a GPU is enabled in your Kaggle notebook settings.")
    else:
        config = Config()

        # --- FIX: Create BOTH dataloaders inside __main__ ---
        # This ensures they use the updated FaceMorphDataset class

        # 1. Create Train Dataloader
        CLIP_TRANSFORMS = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_dataset = FaceMorphDataset(csv_file='train_data.csv', transform=CLIP_TRANSFORMS)
        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        print("Training DataLoader is ready.")

        # 2. Create Validation Dataloader
        VAL_TRANSFORMS = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        val_dataset = FaceMorphDataset(csv_file='val_data.csv', transform=VAL_TRANSFORMS)
        val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
        print("Validation DataLoader is ready.")


        model = CLIP_Model()
        header = ClassificationHeader(model.encoder_output_dim)

        # Pass all dataloaders to the trainer
        trainer = TrainerClip(model, train_dataloader, val_dataloader, config, header)

        # --- RUNNING ALL THREE FUNCTIONS SEQUENTIALLY ---

        print("\nRUNNING MODE 1: Training LoRA + Header (MAD_training)")
        trainer.start_training() # This calls the original MAD_training

        print("\nRUNNING MODE 2: Training Header Only (MAD_training_header_only)")
        # Re-initialize model and header to reset weights before next training
        model = CLIP_Model()
        header = ClassificationHeader(model.encoder_output_dim)
        trainer = TrainerClip(model, train_dataloader, val_dataloader, config, header)
        trainer.MAD_training_header_only()

        print("\nRUNNING MODE 3: Zero-Shot Testing (test_clip)")
        # This will test the weights from the "Header Only" training.
        trainer.test_clip()

        print("\n--- All three modes finished. ---")
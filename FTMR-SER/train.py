import sys
import argparse
import pickle
from data_utils import SERDataset, _concatenate_speaker_data, SERDataset_Custom
import torch
import numpy as np
from models.ser_model import Ser_Model
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import random
import logging
from collections import Counter, defaultdict
from torch.backends import cudnn
from sklearn.model_selection import train_test_split
import time
import pandas as pd
from tqdm import tqdm


# =================================================================
# --- 0. CONFIGURATION CLASS (centralized management of all parameters and paths) ---
# =================================================================

class ProjectConfig:
    """Centrally manage all fixed parameters, file paths, and data-split settings."""

    # --- Target CASIA path settings ---
    BASE_PATH = '/root/'
    FEATURES_FILE = '/root/CASIA_CASIA_4speaker200.pkl'  # feature file containing utter_id

    # --- Emotion label mapping (CASIA 6-class) ---
    EMOTION_LABELS = {
        'angry': 0,
        'fear': 1,
        'sad': 2,
        'happy': 3,
        'neutral': 4,
        'surprise': 5
    }
    LABELS = list(EMOTION_LABELS.keys())
    NUM_CLASSES = 6

    # --- Data split settings (LOSO: split by speaker) ---
    TEST_SIZE = 0.2  # 20% used for the test set (at the utterance level)
    RANDOM_SEED = 111

    # --- Training hyperparameters ---
    NUM_EPOCHS = 60
    EARLY_STOP = 60
    BATCH_SIZE = 32
    LR = 0.00001

    # --- Model and hardware settings ---
    GPU = 1
    GPU_IDS = [0]
    PRETRAINED = True
    OVERSAMPLING = False

    # --- Multi-task learning weights (fully aligned with IEMOCAP) ---
    ALPHA_CE = 1.0  # main classification loss weight
    ALPHA_STUDENT = 0.3  # student-branch CE loss weight
    ALPHA_CL = 0.2  # contrastive loss weight
    ALPHA_SD = 0.5  # self-distillation loss weight (KLDivLoss)
    TEMPERATURE = 8.0  # distillation temperature T

    # --- Output directory generation ---
    @staticmethod
    def setup_output_dir():
        localtime = time.localtime(time.time())
        str_time = f'{str(localtime.tm_year)}-{str(localtime.tm_mon)}-{str(localtime.tm_mday)}-{str(localtime.tm_hour)}-{str(localtime.tm_min)}'

        OUTPUT_DIR = os.path.join(ProjectConfig.BASE_PATH, 'casia_train_test_results', str_time)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        return OUTPUT_DIR, str_time


# =================================================================
# --- Helper function: parse the true emotion label from utter_id ---
# =================================================================
def extract_emotion_from_utter_id(utter_id):
    """
    Parse the emotion label from the utter_id.
    Supported utter_id format examples:
    - "wangzhe_angry_001" -> 0 (angry)
    - "zhangsan_fear_002.wav" -> 1 (fear)
    - "lisi_sad_003.pkl" -> 2 (sad)
    - "happy_004" -> 3 (happy)
    - "neutral_005" -> 4 (neutral)
    - "surprise_006" -> 5 (surprise)
    """
    # Convert to lowercase and remove the file extension
    utter_id_clean = utter_id.lower().split('.')[0]

    # Iterate over the emotion labels and match the keyword
    for emotion, label in ProjectConfig.EMOTION_LABELS.items():
        if emotion in utter_id_clean:
            return label

    # If no match is found, return the default value (or raise an exception)
    logging.warning(f"Unable to parse an emotion label from utter_id {utter_id}; defaulting to neutral (4)")
    return 4


# =================================================================
# --- 1. MODEL, LOSS, AND TRAIN/TEST FUNCTIONS ---
# =================================================================

# --- Contrastive loss function (InfoNCE style) ---
class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=1)

    def forward(self, f_freq, f_time):
        self.target = torch.tensor(1.0).to(f_freq.device)
        sim = self.cosine_similarity(f_freq, f_time)
        loss = F.mse_loss(sim, self.target.expand_as(sim))
        return loss


# --- KULLBACK-LEIBLER DIVERGENCE LOSS for SD (FIXED) ---
def kd_loss(student_logits, teacher_logits, temperature):
    """Computes the Knowledge Distillation loss (KL Divergence)."""
    soft_teacher = F.softmax(teacher_logits.detach() / temperature, dim=1)
    log_soft_student = F.log_softmax(student_logits / temperature, dim=1)
    return F.kl_div(log_soft_student, soft_teacher, reduction='batchmean') * (temperature ** 2)


# seeding function for reproducibility
def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# to count the number of trainable parameter in the model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test(mode, params, model, criterion_ce, criterion_mml, dataset, batch_size, device,
         return_preds=False):
    """Test an SER model. Additionally returns utter_id for traceability."""
    total_loss = 0
    test_preds_segs = []
    test_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    best_epoch = 0  # Dummy initialization for original code compatibility

    # New: save the utter_id of the test set
    test_utter_ids = []

    for test_batch in test_loader:
        test_data_spec_batch = test_batch['seg_spec'].to(device)
        test_data_mfcc_batch = test_batch['seg_mfcc'].to(device)
        test_data_audio_batch = test_batch['seg_audio'].to(device)
        test_labels_batch = test_batch['seg_label'].to(device, dtype=torch.long)

        # Read utter_id (if present)
        if 'utter_id' in test_batch:
            test_utter_ids.extend(test_batch['utter_id'])

        test_outputs = model(test_data_spec_batch, test_data_mfcc_batch, test_data_audio_batch)
        test_preds_segs.append(F.softmax(test_outputs['M'], dim=1).detach().cpu().numpy())

        # Test loss calculation uses the main output 'M'
        test_loss_ce = criterion_ce(test_outputs['M'], test_labels_batch)
        test_loss = test_loss_ce

        total_loss += test_loss.item()

    test_loss = total_loss / len(test_loader)
    test_preds_segs = np.vstack(test_preds_segs)
    test_preds = dataset.get_preds(test_preds_segs)  # obtain utterance-level predictions

    test_wa = dataset.weighted_accuracy(test_preds)
    test_ua = dataset.unweighted_accuracy(test_preds)

    results = (test_loss, test_wa * 100, test_ua * 100)

    if return_preds:  # additionally return utter_id
        return results, test_preds, dataset.actual_target, test_utter_ids
    else:
        return results


def train(dataset_wrapper, params, save_label='default'):  # receive the dataset wrapper

    # get dataset
    train_dataset = dataset_wrapper.get_train_dataset()
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=params['batch_size'],
                                               shuffle=True)

    val_dataset = dataset_wrapper.get_val_dataset()  # validation set
    test_dataset = dataset_wrapper.get_test_dataset()  # test set

    # select device
    if params['use_gpu'] == 1:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = Ser_Model().to(device)

    # Set loss criterion and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=params['lr'])
    criterion_ce = nn.CrossEntropyLoss()
    criterion_mml = nn.MultiMarginLoss(margin=0.5)
    criterion_cl = ContrastiveLoss()
    criterion_kld = nn.KLDivLoss(reduction='batchmean')

    # Read all loss-weight parameters from params
    alpha_ce = params['alpha_ce']
    alpha_student = params['alpha_student']
    alpha_cl = params['contrastive_weight']
    alpha_sd = params['sd_weight']
    temperature = params['sd_temperature']

    logging.info(
        f"Loss Weights - CE: {alpha_ce}, Student CE: {alpha_student}, CL: {alpha_cl}, SD: {alpha_sd}, Temperature: {temperature}"
    )

    loss_format = "{:.04f}"
    acc_format = "{:.02f}%"
    acc_format2 = "{:.02f}"
    best_val_wa = 0
    best_val_ua = 0
    save_path = save_label + '.pth'
    best_val_loss = 1e8
    best_val_acc = -1e8
    best_epoch = 0

    all_train_loss = []
    all_train_wa = []
    all_train_ua = []
    all_val_loss = []
    all_val_wa = []
    all_val_ua = []
    train_preds = []

    logging.info("Start Training...")

    for epoch in range(params['num_epochs']):

        # get current learning rate
        for param_group in optimizer.param_groups:
            current_lr = param_group['lr']

        total_loss = 0
        model.train()

        # Show a batch-level progress bar for each epoch, displaying the current batch loss in real time
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{params['num_epochs']}",
            unit="batch",
            dynamic_ncols=True,
            leave=False
        )
        for train_batch in progress_bar:
            optimizer.zero_grad(set_to_none=True)

            train_data_spec_batch = train_batch['seg_spec'].to(device)
            train_data_mfcc_batch = train_batch['seg_mfcc'].to(device)
            train_data_audio_batch = train_batch['seg_audio'].to(device)
            train_labels_batch = train_batch['seg_label'].to(device, dtype=torch.long)

            outputs = model(train_data_spec_batch, train_data_mfcc_batch, train_data_audio_batch)

            # --- LOSS COMPUTATION ---
            train_loss_ce = criterion_ce(outputs['M'], train_labels_batch)
            train_loss_student_ce = (
                    criterion_ce(outputs['S_spec'], train_labels_batch)
                    + criterion_ce(outputs['S_mfcc'], train_labels_batch)
                    + criterion_ce(outputs['S_wav'], train_labels_batch)
            )
            train_loss_contrastive = criterion_cl(outputs['F_freq'], outputs['F_time'])
            teacher_logits = outputs['M']
            loss_sd_spec = kd_loss(outputs['S_spec'], teacher_logits, temperature)
            loss_sd_mfcc = kd_loss(outputs['S_mfcc'], teacher_logits, temperature)
            loss_sd_wav = kd_loss(outputs['S_wav'], teacher_logits, temperature)
            train_loss_distill = loss_sd_spec + loss_sd_mfcc + loss_sd_wav

            # Total Loss
            train_loss = (
                    alpha_ce * train_loss_ce
                    + alpha_student * train_loss_student_ce
                    + alpha_cl * train_loss_contrastive
                    + alpha_sd * train_loss_distill
            )

            train_loss.backward()
            optimizer.step()
            batch_loss = train_loss.item()
            total_loss += batch_loss
            progress_bar.set_postfix(loss=f"{batch_loss:.4f}")
        progress_bar.close()

        # After each epoch, re-evaluate the full training set in a fixed order.
        # Predictions collected during training with shuffle=True must not be compared
        # directly against the original labels; otherwise the prediction order would not
        # match the label order and WA/UA would be distorted.
        train_loss = total_loss / max(1, len(train_loader))
        train_eval_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=params['batch_size'],
            shuffle=False,
            num_workers=0
        )
        model.eval()
        train_eval_probs = []
        with torch.no_grad():
            for eval_batch in train_eval_loader:
                eval_spec = eval_batch['seg_spec'].to(device)
                eval_mfcc = eval_batch['seg_mfcc'].to(device)
                eval_audio = eval_batch['seg_audio'].to(device)
                eval_outputs = model(eval_spec, eval_mfcc, eval_audio)
                train_eval_probs.append(
                    F.softmax(eval_outputs['M'], dim=1).cpu().numpy()
                )

        if not train_eval_probs:
            raise RuntimeError('The training set is empty; cannot compute training WA/UA.')
        train_eval_probs = np.vstack(train_eval_probs)
        train_preds = train_dataset.get_preds(train_eval_probs)
        train_wa = train_dataset.weighted_accuracy(train_preds) * 100
        train_ua = train_dataset.unweighted_accuracy(train_preds) * 100
        all_train_loss.append(loss_format.format(train_loss))
        all_train_wa.append(acc_format2.format(train_wa))
        all_train_ua.append(acc_format2.format(train_ua))
        model.train()

        # Per-epoch evaluation: in LOSO mode, evaluate the test set directly as requested;
        # in normal mode, evaluate the validation set.
        # Note: under LOSO, using the test set to select the best Epoch makes the test set
        # participate in model selection, so the final metrics will be optimistically biased.
        eval_dataset = test_dataset if params.get('loso_mode', False) else val_dataset
        eval_mode = 'TEST' if params.get('loso_mode', False) else 'VAL'

        with torch.no_grad():
            eval_result = test(
                eval_mode,
                params,
                model,
                criterion_ce,
                criterion_mml,
                eval_dataset,
                batch_size=64,
                device=device
            )

        eval_loss, eval_wa, eval_ua = eval_result
        current_score = eval_wa + eval_ua

        if current_score > best_val_acc:
            best_val_acc = current_score
            best_val_ua = eval_ua
            best_val_wa = eval_wa
            best_val_loss = eval_loss
            best_epoch = epoch
            torch.save(model.state_dict(), save_path)

        all_val_loss.append(loss_format.format(eval_loss))
        all_val_wa.append(acc_format2.format(eval_wa))
        all_val_ua.append(acc_format2.format(eval_ua))

        if params.get('loso_mode', False):
            logging.info(
                f"Epoch {epoch + 1}/{params['num_epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train WA: {train_wa:.2f}% | Train UA: {train_ua:.2f}% | "
                f"Test Loss: {eval_loss:.4f} | Test WA: {eval_wa:.2f}% | "
                f"Test UA: {eval_ua:.2f}% | Best Epoch: {best_epoch + 1} | "
                f"Best Test WA: {best_val_wa:.2f}% | Best Test UA: {best_val_ua:.2f}%"
            )
        else:
            logging.info(
                f"Epoch {epoch + 1}/{params['num_epochs']} (Best:{best_epoch + 1}) | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train WA: {train_wa:.2f}% | Train UA: {train_ua:.2f}% | "
                f"Val Loss: {eval_loss:.4f} | Val WA: {eval_wa:.2f}% "
                f"<Best {best_val_wa:.2f}%> | Val UA: {eval_ua:.2f}% "
                f"<Best {best_val_ua:.2f}%>"
            )

        if (epoch - best_epoch >= params['early_stop']) and (epoch > 5):
            logging.info(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    # Test on best model
    with torch.no_grad():
        model.load_state_dict(torch.load(save_path, map_location=device))
        # New: return utter_id for traceability
        test_result, test_preds, test_targets, test_utter_ids = test(
            'TEST', params, model, criterion_ce, criterion_mml, test_dataset,
            batch_size=64, device=device, return_preds=True)

    test_loss_print = loss_format.format(test_result[0])
    test_wa_print = acc_format2.format(test_result[1])
    test_ua_print = acc_format2.format(test_result[2])

    logging.info("*" * 40)
    logging.info("RESULTS ON TEST SET (utterance-level WA/UA):")
    logging.info("Loss:{}\tWA: {}\tUA: {}".format(test_loss_print, test_wa_print, test_ua_print))

    # New: return utter_id
    return (epoch, best_epoch, all_train_loss, all_train_wa, all_train_ua,
            all_val_loss, all_val_wa, all_val_ua,
            test_loss_print, test_wa_print, test_ua_print,
            test_dataset, test_preds, test_targets,
            test_utter_ids)


# =================================================================
# --- 2. New: group data by Utterance ID (final fixed version) ---
# =================================================================
def group_data_by_utterance(features_data):
    """
    Group data by utter_id, adapted to the pure-numeric utter_id scenario.
    Key improvements:
    1. No longer attempts to parse pure-numeric utter_ids; directly use the seg_label of the
       first segment under each utterance as the utterance-level label.
    2. Keep the segment-level labels for training to preserve the 6-class distribution.
    3. Strengthen data validation and logging output.
    """
    utter_groups = defaultdict(dict)
    all_utter_ids = []
    all_utter_labels = []

    for speaker_id in features_data.keys():
        speaker_data = features_data[speaker_id]

        # 1. Basic data validation
        assert 'utter_id' in speaker_data, f"Speaker {speaker_id} is missing the utter_id field!"
        assert 'seg_label' in speaker_data, f"Speaker {speaker_id} is missing the seg_label field!"

        # 2. Group all segments by utter_id
        seg_indices_map = defaultdict(list)
        for seg_idx, utter_id in enumerate(speaker_data['utter_id']):
            seg_indices_map[utter_id].append(seg_idx)

        # 3. Build utter_groups and extract the true utterance-level labels (core fix)
        for utter_id, seg_indices in seg_indices_map.items():
            # Core change: directly take the seg_label of the first segment under this utterance
            # as the utterance-level label.
            # Logs have verified that seg_label contains all 6 classes, so this logic guarantees
            # that utter_label also follows the 6-class distribution.
            utter_label = speaker_data['seg_label'][seg_indices[0]]

            utter_groups[utter_id] = {
                'seg_spec': speaker_data['seg_spec'][seg_indices],
                'seg_mfcc': speaker_data['seg_mfcc'][seg_indices],
                'seg_audio': speaker_data['seg_audio'][seg_indices],
                'seg_label': speaker_data['seg_label'][seg_indices],
                'utter_label': utter_label,  # true emotion label (extracted from segment labels)
                'seg_num': len(seg_indices)
            }
            all_utter_ids.append(utter_id)
            all_utter_labels.append(utter_label)

    # Final data validation
    assert len(all_utter_ids) == len(all_utter_labels), "Total utter_id count does not match the label count!"
    logging.info(
        f"Grouping complete: {len(all_utter_ids)} utterances in total, utterance-level label distribution: {Counter(all_utter_labels)}")

    # Verify whether the label distribution is complete
    label_dist = Counter(all_utter_labels)
    missing_labels = [label for label in range(6) if label not in label_dist]
    if missing_labels:
        logging.warning(
            f"Warning: the following emotion labels were not found at the utterance level: {missing_labels}")
    else:
        logging.info("All 6 emotion label classes are present in the distribution")

    return utter_groups, np.array(all_utter_ids), np.array(all_utter_labels)


# =================================================================
# --- 3. Leave-One-Speaker-Out cross-validation by speaker ---
def build_dataset_from_speakers(speaker_ids, features_data):
    """
    Build data by speaker. All utterances of a given speaker belong to the same dataset,
    and all segments of the same utterance are always kept together to avoid
    speaker/utterance leakage.
    """
    spec_list, mfcc_list, audio_list, seg_label_list = [], [], [], []
    utter_label_list, seg_num_list, utter_id_list = [], [], []

    for speaker_id in speaker_ids:
        # Compatible with keys in the pickle dict being either strings or convertible to strings
        sid = speaker_id
        if sid not in features_data:
            sid = next((k for k in features_data.keys() if str(k) == str(speaker_id)), None)
        if sid is None:
            raise KeyError(f"Cannot find speaker ID: {speaker_id}")

        data = features_data[sid]
        if 'utter_id' not in data:
            raise KeyError(f"The feature data of speaker {sid} is missing utter_id; cannot aggregate by utterance.")

        ids = np.asarray(data['utter_id'])
        # Obtain the utterance IDs within this speaker in order of first appearance to avoid
        # collisions caused by identically named IDs across different speakers
        unique_ids = list(dict.fromkeys(ids.tolist()))
        for uid in unique_ids:
            idx = np.where(ids == uid)[0]
            spec_list.append(np.asarray(data['seg_spec'])[idx].astype(np.float32))
            mfcc_list.append(np.asarray(data['seg_mfcc'])[idx].astype(np.float32))
            audio_list.append(np.asarray(data['seg_audio'])[idx].astype(np.float32))
            labels = np.asarray(data['seg_label'])[idx].astype(np.int64)
            if labels.size == 0:
                continue
            if np.any((labels < 0) | (labels >= ProjectConfig.NUM_CLASSES)):
                raise ValueError(
                    f'Utterance {uid} of speaker {sid} contains labels outside 0~{ProjectConfig.NUM_CLASSES - 1}: '
                    f'{np.unique(labels).tolist()}'
                )
            # Utterance-level classification requires all segments of the same utterance to share
            # the same label. If they are inconsistent, raise an error directly to avoid incorrectly
            # substituting the first segment label for the whole-utterance label.
            if not np.all(labels == labels[0]):
                raise ValueError(
                    f'The segment labels of utterance {sid}::{uid} are inconsistent: {np.unique(labels).tolist()}. '
                    'Please check seg_label / utter_id in the feature file.'
                )
            seg_label_list.append(labels)
            utter_label_list.append(int(labels[0]))
            seg_num_list.append(len(idx))
            utter_id_list.extend([f"{sid}::{uid}"] * len(idx))

    if not spec_list:
        raise ValueError(f"No data was read from speakers {speaker_ids}.")

    return {
        'seg_spec': np.concatenate(spec_list, axis=0),
        'seg_mfcc': np.concatenate(mfcc_list, axis=0),
        'seg_audio': np.concatenate(audio_list, axis=0),
        'seg_label': np.concatenate(seg_label_list, axis=0),
        'utter_label': np.asarray(utter_label_list, dtype=np.int64),
        'seg_num': np.asarray(seg_num_list, dtype=np.int64),
        'utter_id': np.asarray(utter_id_list)
    }


def train_test_split_casia():
    """CASIA Leave-One-Speaker-Out cross-validation (LOSO): each fold holds out one speaker as the test set."""
    torch.autograd.set_detect_anomaly(True)

    OUTPUT_DIR, str_time = ProjectConfig.setup_output_dir()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log_file_name = os.path.join(OUTPUT_DIR, 'loso_results.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler(log_file_name, mode='w'), logging.StreamHandler(sys.stdout)],
        force=True
    )

    features_file = ProjectConfig.FEATURES_FILE
    num_classes = ProjectConfig.NUM_CLASSES
    params = {
        'features_file': features_file,
        'num_epochs': ProjectConfig.NUM_EPOCHS,
        'early_stop': ProjectConfig.EARLY_STOP,
        'batch_size': ProjectConfig.BATCH_SIZE,
        'lr': ProjectConfig.LR,
        'seed': ProjectConfig.RANDOM_SEED,
        'gpu': ProjectConfig.GPU,
        'use_gpu': ProjectConfig.GPU,
        'alpha_ce': ProjectConfig.ALPHA_CE,
        'alpha_student': ProjectConfig.ALPHA_STUDENT,
        'contrastive_weight': ProjectConfig.ALPHA_CL,
        'sd_weight': ProjectConfig.ALPHA_SD,
        'sd_temperature': ProjectConfig.TEMPERATURE,
        # LOSO: evaluate the test set every epoch and select the best epoch, as requested
        'loso_mode': True
    }

    seed_everything(ProjectConfig.RANDOM_SEED)
    with open(features_file, 'rb') as fin:
        features_data = pickle.load(fin)

    speaker_ids = list(features_data.keys())
    if len(speaker_ids) < 2:
        raise ValueError(f"LOSO requires at least 2 speakers; currently there are only {len(speaker_ids)}.")

    logging.info("=" * 70)
    logging.info("Starting CASIA Leave-One-Speaker-Out (LOSO) cross-validation")
    logging.info(f"Number of speakers: {len(speaker_ids)}; each fold tests 1 speaker and trains on the rest")
    logging.info(f"Speaker list: {[str(s) for s in speaker_ids]}")
    logging.info(f"Hyperparameters: LR={params['lr']}, BatchSize={params['batch_size']}, Epochs={params['num_epochs']}")
    logging.info(
        "LOSO mode: as requested, evaluate the held-out speaker\'s test set every epoch and save the best model by Test WA + Test UA; the test set participates in model selection.")

    fold_rows = []

    for fold_idx, test_speaker in enumerate(speaker_ids, start=1):
        train_speakers = [sid for sid in speaker_ids if sid != test_speaker]
        fold_dir = os.path.join(OUTPUT_DIR, f"fold_{fold_idx:02d}_test_{str(test_speaker)}")
        os.makedirs(fold_dir, exist_ok=True)
        save_label = os.path.join(fold_dir, f"casia_loso_fold_{fold_idx:02d}")

        logging.info("\n" + "#" * 70)
        logging.info(f"Fold {fold_idx}/{len(speaker_ids)} | test speaker: {test_speaker}")
        logging.info(f"training speakers: {train_speakers}")

        # Reset the random seed for each fold to ensure reproducibility and consistent initialization across folds
        seed_everything(ProjectConfig.RANDOM_SEED)
        train_data_split = build_dataset_from_speakers(train_speakers, features_data)
        test_data_split = build_dataset_from_speakers([test_speaker], features_data)

        logging.info(f"Number of training segments: {len(train_data_split['seg_label'])}, "
                     f"label distribution: {Counter(train_data_split['seg_label'])}")
        logging.info(f"Number of test segments: {len(test_data_split['seg_label'])}, "
                     f"label distribution: {Counter(test_data_split['seg_label'])}")

        dataset_wrapper = SERDataset_Custom(
            train_data_split, test_data_split, num_classes=num_classes
        )

        stat = train(dataset_wrapper, params, save_label=save_label)
        test_preds = stat[12]
        test_targets = stat[13]
        test_utter_ids = stat[14]

        # Save the per-utterance prediction results of this fold (the test utter_ids already include the speaker prefix).
        # test_utter_ids are segment-level IDs; after de-duplicating by order of first appearance, they align with the
        # utterance-level predictions
        unique_test_utter_ids = list(dict.fromkeys(np.asarray(test_utter_ids).tolist()))
        pred_df = pd.DataFrame({
            'utter_id': unique_test_utter_ids,
            'true_label': test_targets,
            'pred_label': test_preds
        })
        pred_df.to_csv(os.path.join(fold_dir, "utter_predictions.csv"), index=False)

        wa = float(stat[9])
        ua = float(stat[10])
        fold_rows.append({
            'fold': fold_idx,
            'test_speaker': str(test_speaker),
            'train_speakers': ','.join(map(str, train_speakers)),
            'WA': wa,
            'UA': ua,
            'test_utterances': len(test_targets)
        })

        logging.info(f"Fold {fold_idx} complete | utterance-level WA={wa:.2f}% | UA={ua:.2f}%")

    # Aggregate the metrics across folds: report the mean and standard deviation over all folds
    results_df = pd.DataFrame(fold_rows)
    results_df.to_csv(os.path.join(OUTPUT_DIR, "loso_fold_results.csv"), index=False)
    mean_wa, std_wa = results_df['WA'].mean(), results_df['WA'].std(ddof=0)
    mean_ua, std_ua = results_df['UA'].mean(), results_df['UA'].std(ddof=0)
    logging.info("\n" + "=" * 70)
    logging.info("LOSO summary results (utterance level)")
    logging.info(f"WA: {mean_wa:.2f}% +/- {std_wa:.2f}%")
    logging.info(f"UA: {mean_ua:.2f}% +/- {std_ua:.2f}%")

    logging.info(f"Results output directory: {OUTPUT_DIR}")
    return results_df


# =================================================================
# --- 4. Entry point ---
# =================================================================

if __name__ == '__main__':
    train_test_split_casia()

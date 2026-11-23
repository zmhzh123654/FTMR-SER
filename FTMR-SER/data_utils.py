import numpy as np
import torch
from torchvision import transforms
from imblearn.over_sampling import RandomOverSampler
from sklearn import preprocessing
from PIL import Image
from collections import defaultdict


SCALER_TYPE = {
    'standard': 'preprocessing.StandardScaler()',
    'minmax': 'preprocessing.MinMaxScaler(feature_range=(0,1))'
}


def _concatenate_speaker_data(features_data, speaker_ids):
    """Safely concatenates data from a list of speaker IDs."""

    spec, mfcc, audio, seg_labels, utter_labels, seg_nums, utter_ids = None, None, None, None, None, None, None

    for sid in speaker_ids:

        sid = sid.strip()
        if not sid or sid not in features_data:
            continue

        data = features_data[sid]

        if spec is None:
            spec = data['seg_spec'].astype(np.float32)
            mfcc = data['seg_mfcc'].astype(np.float32)
            audio = data['seg_audio'].astype(np.float32)
            seg_labels = data['seg_label'].astype(np.int64)
            utter_labels = data['utter_label'].astype(np.int64)
            seg_nums = data['seg_num']
            utter_ids = data['utter_id'] if 'utter_id' in data else None

        else:
            spec = np.concatenate((spec, data['seg_spec'].astype(np.float32)), axis=0)
            mfcc = np.concatenate((mfcc, data['seg_mfcc'].astype(np.float32)), axis=0)
            audio = np.concatenate((audio, data['seg_audio'].astype(np.float32)), axis=0)
            seg_labels = np.concatenate((seg_labels, data['seg_label'].astype(np.int64)), axis=0)
            utter_labels = np.concatenate((utter_labels, data['utter_label'].astype(np.int64)), axis=0)
            seg_nums = np.concatenate((seg_nums, data['seg_num']), axis=0)
            if 'utter_id' in data and utter_ids is not None:
                utter_ids = np.concatenate((utter_ids, data['utter_id']), axis=0)

    if spec is None:
        raise ValueError(f"No data found for speaker IDs: {speaker_ids}. Check if features_data keys match the speakers or if file is empty.")

    return spec, mfcc, audio, seg_labels, utter_labels, seg_nums, utter_ids


class TrainDataset(torch.utils.data.Dataset):
    """
    data : ndarray
        Input data of shape , where  is the number of examples
        (segments), C is number of input channels (3 in the case of image),  is image height,
         is image width
    target : ndarray
        Labels for segments (note that one utterance might contain more than
        one segments) of shape .
    num_classes :
        Number of classes.
    """
    def __init__(self, data, num_classes=6):
        super().__init__()
        self.data_spec = data['seg_spec']
        self.data_mfcc = data['seg_mfcc']
        self.data_audio = data['seg_audio']
        self.seg_label = data['seg_label']
        self.n_samples = len(self.seg_label)
        self.num_classes = num_classes
        self.utter_id = data.get('utter_id', None)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = {
            'seg_spec': self.data_spec[index],
            'seg_mfcc': self.data_mfcc[index],
            'seg_audio': self.data_audio[index],
            'seg_label': self.seg_label[index]
        }

        if self.utter_id is not None:
            sample['utter_id'] = self.utter_id[index]
        return sample

    def get_preds(self, preds):
        preds = np.argmax(preds, axis=1)
        return preds

    def weighted_accuracy(self, predictions):
        acc = (self.seg_label == predictions).sum() / self.n_samples
        return acc

    def unweighted_accuracy(self, predictions):
        class_acc = 0
        n_classes = 0
        for c in range(self.num_classes):
            class_pred = np.multiply((self.seg_label == predictions),
                                     (self.seg_label == c)).sum()

            if (self.seg_label == c).sum() > 0:
                class_pred /= (self.seg_label == c).sum()
                n_classes += 1
                class_acc += class_pred

        return class_acc / n_classes


class TestDataset(torch.utils.data.Dataset):
    """
    Holds data for a validation/test set.
    """

    def __init__(self, data, num_classes=6):
        super().__init__()
        self.data_spec = data['seg_spec']
        self.data_mfcc = data['seg_mfcc']
        self.data_audio = data['seg_audio']
        self.target = data['seg_label']
        self.n_samples = len(self.target)
        self.actual_target = data['utter_label']
        self.n_actual_samples = len(self.actual_target)
        self.num_segs = data['seg_num']
        self.num_classes = num_classes
        self.labels = ['angry', 'fear', 'sad', 'happy', 'neutral', 'surprise'][:num_classes]

        self.utter_id = data.get('utter_id', None)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        sample = {
            'seg_spec': self.data_spec[index],
            'seg_mfcc': self.data_mfcc[index],
            'seg_audio': self.data_audio[index],
            'seg_label': self.target[index]
        }

        if self.utter_id is not None:
            sample['utter_id'] = self.utter_id[index]
        return sample

    def get_preds(self, seg_preds):
        """
        Get predictions for all utterances from their segments' prediction.
        """
        preds = np.empty(
            shape=(self.n_actual_samples, self.num_classes), dtype="float")

        end = 0
        for v in range(self.n_actual_samples):
            start = end
            end = start + self.num_segs[v]

            preds[v] = np.average(seg_preds[start:end], axis=0)

        preds = np.argmax(preds, axis=1)
        return preds

    def weighted_accuracy(self, utt_preds):
        acc = (self.actual_target == utt_preds).sum() / self.n_actual_samples
        return acc

    def unweighted_accuracy(self, utt_preds):
        class_acc = 0
        n_classes = 0

        for c in range(self.num_classes):
            class_pred = np.multiply((self.actual_target == utt_preds),
                                     (self.actual_target == c)).sum()

            if (self.actual_target == c).sum() > 0:
                class_pred /= (self.actual_target == c).sum()
                n_classes += 1
                class_acc += class_pred

        return class_acc / n_classes


class SERDataset:
    """
    Wrapper for both  and , which loads and pre-process
    the data and applies speaker-level train/validation/test split (LOSO).
    """
    def __init__(self, features_data, num_classes = 4,
                 val_speaker_id='1M', test_speaker_id='1F',
                 oversample=False):

        val_ids = [sid.strip() for sid in val_speaker_id.split(',') if sid.strip()]
        test_ids = [sid.strip() for sid in test_speaker_id.split(',') if sid.strip()]

        exclude_ids = set(val_ids + test_ids)

        train_spec_data, train_mfcc_data, train_audio_data, train_seg_labels, train_labels = None, None, None, None, None

        for speaker_id in features_data.keys():
            if speaker_id in exclude_ids:
                continue

            if train_mfcc_data is None:
                train_spec_data = features_data[speaker_id]['seg_spec'].astype(np.float32)
                train_mfcc_data = features_data[speaker_id]['seg_mfcc'].astype(np.float32)
                train_audio_data = features_data[speaker_id]['seg_audio'].astype(np.float32)
            else:
                train_spec_data = np.concatenate((train_spec_data,
                                                  features_data[speaker_id]['seg_spec'].astype(np.float32) ),
                                                  axis=0)
                train_mfcc_data = np.concatenate((train_mfcc_data,
                                                  features_data[speaker_id]['seg_mfcc'].astype(np.float32) ),
                                                  axis=0)
                train_audio_data = np.concatenate((train_audio_data,
                                                  features_data[speaker_id]['seg_audio'].astype(np.float32) ),
                                                  axis=0)

            if train_seg_labels is None:
                train_seg_labels = features_data[speaker_id]['seg_label'].astype(np.int64)
                train_labels = features_data[speaker_id]['utter_label'].astype(np.int64)
            else:
                train_seg_labels = np.concatenate((train_seg_labels,
                                                   features_data[speaker_id]['seg_label'].astype(np.int64)),
                                                   axis=0)
                train_labels = np.concatenate((train_labels,
                                                   features_data[speaker_id]['utter_label'].astype(np.int64)),
                                                   axis=0)

        self.train_spec_data = train_spec_data
        self.train_mfcc_data = train_mfcc_data
        self.train_audio_data = train_audio_data
        self.train_seg_labels = train_seg_labels
        self.train_labels = train_labels
        self.num_classes = num_classes

        (self.val_spec_data, self.val_mfcc_data, self.val_audio_data,
         self.val_seg_labels, self.val_labels, self.val_num_segs, self.val_utter_ids) = \
            _concatenate_speaker_data(features_data, val_ids)

        (self.test_spec_data, self.test_mfcc_data, self.test_audio_data,
         self.test_seg_labels, self.test_labels, self.test_num_segs, self.test_utter_ids) = \
            _concatenate_speaker_data(features_data, test_ids)

        self._normalize('minmax')

        if oversample == True:
            datar, labelr = random_oversample(self.train_spec_data, self.train_labels)
            datar, labelr = random_oversample(datar, labelr)
            self.train_spec_data = datar
            self.train_labels = labelr

        train_spec_data_shape = self.train_spec_data.shape
        val_spec_data_shape = self.val_spec_data.shape
        test_spec_data_shape = self.test_spec_data.shape

        self.train_spec_data = self._spec_to_gray(self.train_spec_data)
        self.val_spec_data = self._spec_to_gray(self.val_spec_data)
        self.test_spec_data = self._spec_to_gray(self.test_spec_data)
        self.num_in_ch = 1

        self.train_data = defaultdict()
        self.train_data["seg_spec"] = self.train_spec_data
        self.train_data["seg_mfcc"] = self.train_mfcc_data
        self.train_data["seg_audio"] = self.train_audio_data
        self.train_data["seg_label"] = self.train_seg_labels

        self.val_data = defaultdict()
        self.val_data["seg_spec"] = self.val_spec_data
        self.val_data["seg_mfcc"] = self.val_mfcc_data
        self.val_data["seg_audio"] = self.val_audio_data
        self.val_data["seg_label"] = self.val_seg_labels
        self.val_data["utter_label"] = self.val_labels
        self.val_data["seg_num"] = self.val_num_segs

        if self.val_utter_ids is not None:
            self.val_data["utter_id"] = self.val_utter_ids

        self.test_data = defaultdict()
        self.test_data["seg_spec"] = self.test_spec_data
        self.test_data["seg_mfcc"] = self.test_mfcc_data
        self.test_data["seg_audio"] = self.test_audio_data
        self.test_data["seg_label"] = self.test_seg_labels
        self.test_data["utter_label"] = self.test_labels
        self.test_data["seg_num"] = self.test_num_segs

        if self.test_utter_ids is not None:
            self.test_data["utter_id"] = self.test_utter_ids

        assert len(self.train_spec_data) == train_spec_data_shape[0]
        assert len(self.val_spec_data) == val_spec_data_shape[0]
        assert len(self.test_spec_data) == test_spec_data_shape[0]

        assert val_spec_data_shape[0] == self.val_seg_labels.shape[0] == sum(self.val_num_segs)
        assert self.val_labels.shape[0] == self.val_num_segs.shape[0]
        assert test_spec_data_shape[0] == self.test_seg_labels.shape[0] == sum(self.test_num_segs)
        assert self.test_labels.shape[0] == self.test_num_segs.shape[0]


    def _normalize(self, scaling):
        '''
        calculate normalization factor from training dataset and apply to
        the whole dataset
        '''

        #get data range
        self._get_data_range()

        #re-arrange array from (N, C, F, T) to (C, -1, F)
        nch = self.train_spec_data.shape[1]
        nfreq = self.train_spec_data.shape[2]
        ntime = self.train_spec_data.shape[3]
        rearrange = lambda x: x.transpose(1,0,3,2).reshape(nch,-1,nfreq)
        self.train_spec_data = rearrange(self.train_spec_data)
        self.val_spec_data = rearrange(self.val_spec_data)
        self.test_spec_data = rearrange(self.test_spec_data)

        #scaler type
        scaler = eval(SCALER_TYPE[scaling])

        for ch in range(nch):
            #get scaling values from training data
            scaler.fit(self.train_spec_data[ch])

            #apply to all
            self.train_spec_data[ch] = scaler.transform(self.train_spec_data[ch])
            self.val_spec_data[ch] = scaler.transform(self.val_spec_data[ch])
            self.test_spec_data[ch] = scaler.transform(self.test_spec_data[ch])

        #Shape the data back to (N,C,F,T)
        rearrange = lambda x: x.reshape(nch,-1,ntime,nfreq).transpose(1,0,3,2)
        self.train_spec_data = rearrange(self.train_spec_data)
        self.val_spec_data = rearrange(self.val_spec_data)
        self.test_spec_data = rearrange(self.test_spec_data)

    def _get_data_range(self):
        #get data range
        trmin = np.min(self.train_spec_data)
        evmin = np.min(self.val_spec_data)
        tsmin = np.min(self.test_spec_data)
        dmin = np.min(np.array([trmin, evmin, tsmin]))

        trmax = np.max(self.train_spec_data)
        evmax = np.max(self.val_spec_data)
        tsmax = np.max(self.test_spec_data)
        dmax = np.max(np.array([trmax, evmax, tsmax]))

        return [dmin, dmax]

    def _spec_to_gray(self, data):
        """
        Convert normalized spectrogram to 3-channel gray image (identical data on each channel)
        and apply AlexNet image pre-processing

        Input: data
            - shape (N,C,H,W) = (num_spectrogram_segments, 1, Freq, Time)
            - data range [0.0, 1.0]
        """

        #AlexNet preprocessing
        alexnet_preprocess = transforms.Compose([
            transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        #Convert format to uint8, flip the frequency axis to orientate image upward,
        #   duplicate into 3 channels
        data = np.clip(data,0.0, 1.0)
        data = (data*255.0).astype(np.uint8)
        data = np.flip(data,axis=2)
        data = np.moveaxis(data,1,-1)
        data = np.repeat(data,3,axis=-1)

        data_tensor = list()
        for i, seg in enumerate(data):
            img = Image.fromarray(seg, mode='RGB')
            data_tensor.append(alexnet_preprocess(img))

        return data_tensor

    def get_train_dataset(self):
        return TrainDataset(
            self.train_data, num_classes=self.num_classes)

    def get_val_dataset(self):
        return TestDataset(
            self.val_data, num_classes=self.num_classes)

    def get_test_dataset(self):
        return TestDataset(
            self.test_data, num_classes=self.num_classes)


def random_oversample(data, labels):
    """
    Performs random oversampling on the data and corresponding labels
    """
    ros = RandomOverSampler(random_state=0, sampling_strategy='minority')

    n_samples = data.shape[0]
    fh = data.shape[2]
    fw = data.shape[3]
    n_features= fh*fw

    data = np.squeeze(data,axis=1)
    data = np.reshape(data,(n_samples, n_features))
    data_resampled, label_resampled = ros.fit_resample(data, labels)
    n_samples = data_resampled.shape[0]
    data_resampled = np.reshape(data_resampled,(n_samples,fh,fw))
    data_resampled = np.expand_dims(data_resampled, axis=1)

    return data_resampled, label_resampled


class SERDataset_Custom:
    """
    Adapter class to handle Spectrogram pre-processing (Normalization,
    Gray-to-3ch conversion, AlexNet pre-processing) for split data.
    """
    def __init__(self, train_data_raw, test_data_raw, num_classes = 6):

        self.num_classes = num_classes

        self.train_data = defaultdict(lambda: None)
        self.test_data = defaultdict(lambda: None)

        self.train_data["seg_spec"] = train_data_raw['seg_spec']
        self.train_data["seg_mfcc"] = train_data_raw['seg_mfcc']
        self.train_data["seg_audio"] = train_data_raw['seg_audio']
        self.train_data["seg_label"] = train_data_raw['seg_label']

        self.train_data["utter_id"] = train_data_raw.get('utter_id', None)

        self.test_data["seg_spec"] = test_data_raw['seg_spec']
        self.test_data["seg_mfcc"] = test_data_raw['seg_mfcc']
        self.test_data["seg_audio"] = test_data_raw['seg_audio']
        self.test_data["seg_label"] = test_data_raw['seg_label']
        self.test_data["utter_label"] = test_data_raw['utter_label']
        self.test_data["seg_num"] = test_data_raw['seg_num']

        self.test_data["utter_id"] = test_data_raw.get('utter_id', None)

        # Spectrogram normalization
        self._normalize('minmax')

        # Spectrogram gray-to-3ch + AlexNet pre-processing
        self.train_data["seg_spec"] = self._spec_to_gray(self.train_data["seg_spec"])
        self.test_data["seg_spec"] = self._spec_to_gray(self.test_data["seg_spec"])


    def _normalize(self, scaling):
        '''
        calculate normalization factor from training dataset and apply to
        the whole dataset
        '''

        #get data range
        self._get_data_range()

        #re-arrange array from (N, C, F, T) to (C, -1, F)
        train_spec_data = self.train_data["seg_spec"]
        test_spec_data = self.test_data["seg_spec"]

        nch = train_spec_data.shape[1]
        nfreq = train_spec_data.shape[2]
        ntime = train_spec_data.shape[3]
        rearrange = lambda x: x.transpose(1,0,3,2).reshape(nch,-1,nfreq)

        train_spec_data = rearrange(train_spec_data)
        test_spec_data = rearrange(test_spec_data)

        #scaler type
        scaler = eval(SCALER_TYPE[scaling])

        for ch in range(nch):
            #get scaling values from training data
            scaler.fit(train_spec_data[ch])

            #apply to all
            train_spec_data[ch] = scaler.transform(train_spec_data[ch])
            test_spec_data[ch] = scaler.transform(test_spec_data[ch])

        #Shape the data back to (N,C,F,T)
        rearrange = lambda x: x.reshape(nch,-1,ntime,nfreq).transpose(1,0,3,2)
        self.train_data["seg_spec"] = rearrange(train_spec_data)
        self.test_data["seg_spec"] = rearrange(test_spec_data)

    def _get_data_range(self):
        #get data range
        trmin = np.min(self.train_data["seg_spec"])
        tsmin = np.min(self.test_data["seg_spec"])
        dmin = np.min(np.array([trmin, tsmin]))

        trmax = np.max(self.train_data["seg_spec"])
        tsmax = np.max(self.test_data["seg_spec"])
        dmax = np.max(np.array([trmax, tsmax]))

        return [dmin, dmax]

    def _spec_to_gray(self, data):
        """
        Convert normalized spectrogram to 3-channel gray image (identical data on each channel)
        and apply AlexNet image pre-processing
        """
        #AlexNet preprocessing
        alexnet_preprocess = transforms.Compose([
            transforms.Resize(256),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        #Convert format to uint8, flip the frequency axis to orientate image upward,
        #   duplicate into 3 channels
        data = np.clip(data,0.0, 1.0)
        data = (data*255.0).astype(np.uint8)
        data = np.flip(data,axis=2)
        data = np.moveaxis(data,1,-1)
        data = np.repeat(data,3,axis=-1)

        data_tensor = list()
        for i, seg in enumerate(data):
            img = Image.fromarray(seg, mode='RGB')
            data_tensor.append(alexnet_preprocess(img))

        return data_tensor

    def get_train_dataset(self):
        return TrainDataset(
            self.train_data, num_classes=self.num_classes)

    def get_val_dataset(self):
        return TestDataset(
            self.test_data, num_classes=self.num_classes)

    def get_test_dataset(self):
        return TestDataset(
            self.test_data, num_classes=self.num_classes)

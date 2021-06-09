from typing import Any, Callable, Optional, Tuple
import torchaudio
from src.utils.datasets.AudioFolder import AudioFolder
import torch
import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


class PerceiverAudioFolder(AudioFolder):

    def __init__(self,
            root: str,
            audio_cfg = None,
            transform: Optional[Callable] = None,
            is_valid_file: Optional[Callable[[str], bool]] = None,
            norm_mean_std=None
    ):
        super().__init__(root, transform=transform, is_valid_file=is_valid_file)
        self.clip_length = audio_cfg.clip_length
        self.sample_rate = audio_cfg.sample_rate
        self.hop_length = audio_cfg.hop_length
        self.n_fft = audio_cfg.n_fft
        self.n_mels = audio_cfg.n_mels
        self.f_min = audio_cfg.f_min
        self.f_max = audio_cfg.f_max
        
        self.unit_length = int((audio_cfg.clip_length * audio_cfg.sample_rate + audio_cfg.hop_length - 1) // audio_cfg.hop_length)
        self.norm_mean_std = norm_mean_std
        self.preprocess(root)

    def preprocess(self, root):
        preprocessed_root = root + '_preprocessed'
        to_mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate, n_fft=self.n_fft, n_mels=self.n_mels,
            hop_length=self.hop_length, f_min=self.f_min, f_max=self.f_max)
        self.to_mel_spectrogram = to_mel_spectrogram
        if not os.path.exists(preprocessed_root):
            for index, sample in enumerate(self.samples):
                path, target = sample
                # Load waveform
                waveform, sample_rate = torchaudio.load(path)
                # Resample
                if sample_rate != self.sample_rate:
                    waveform = torchaudio.transforms.Resample(sample_rate, self.sample_rate)(waveform)
                # To log-mel spectrogram
                log_mel_spec = (to_mel_spectrogram(waveform) + torch.finfo(torch.float).eps).log()
                # Write to work folder
                if not os.path.isdir(preprocessed_root):
                    os.mkdir(preprocessed_root)
                filename = os.path.basename(path).split('.')[0] + '.npy'
                new_path = os.path.join(preprocessed_root, filename)
                np.save(new_path, log_mel_spec)
                
        for index, sample in enumerate(self.samples):
            path, target = sample
            filename = os.path.basename(path).split('.')[0] + '.npy'
            new_path = os.path.join(preprocessed_root, filename)
            self.samples[index] = new_path, target

    def get_class_weight(self):
        classes = self.classes
        labels = [sample[1] for sample in self.samples]
        class_weight = compute_class_weight('balanced', range(len(classes)), labels)
        return class_weight
    
    def sample_length(self, log_mel_spec):
        return log_mel_spec.shape[-1]

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]

        log_mel_spec = np.load(path, allow_pickle=True)

        # normalize - instance based
        if self.norm_mean_std is not None:
            log_mel_spec = (log_mel_spec - self.norm_mean_std[0]) / self.norm_mean_std[1]

        # Padding if sample is shorter than expected - both head & tail are filled with 0s
        pad_size = self.unit_length - self.sample_length(log_mel_spec)
        if pad_size > 0:
            offset = pad_size // 2
            log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, 0), (offset, pad_size - offset)), 'constant')

        # Random crop
        crop_size = self.sample_length(log_mel_spec) - self.unit_length
        if crop_size > 0:
            start = np.random.randint(0, crop_size)
            log_mel_spec = log_mel_spec[..., start:start + self.unit_length]

        # Apply augmentations
        log_mel_spec = torch.Tensor(log_mel_spec)
        if self.transform is not None:
            log_mel_spec = self.transform(log_mel_spec)

        return log_mel_spec, target
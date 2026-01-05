import json
import os
import pathlib
import tempfile

import numpy as np
import torch
import scipy.io.wavfile as wavfile

from .audio_processing._bootstrap import ensure_project_root_on_sys_path

ensure_project_root_on_sys_path()

from utils.config_utils import read_full_config

from .audio_processing.features import load_audio_mono_resample, extract_mel_f0_segments
from .audio_processing.hifigan_infer import (
    build_model_and_mel_transform,
    synthesize_full,
    synthesize_segment_with_padding,
)
from .audio_processing.tension_fx import apply_tension_tilt_pd
from .audio_processing.vslib_engine import (
    VslibEngine,
    VslibError,
    VslibStatus,
    VslibUnavailableError,
)



class AudioProcessor:
    """Public entrypoint for model loading, feature extraction, and synthesis.

    Heavy-lifting is delegated into `hifi_shifter.audio_processing.*` submodules
    so each part can be debugged independently.
    """

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None
        self.config: dict = {}
        self.mel_transform = None
        self.synthesis_engine = 'hifigan'
        self._vslib_engine: VslibEngine | None = None

    def load_model(self, folder_path):
        """Load model and configuration from the specified folder."""
        folder_path = pathlib.Path(folder_path)

        # Check for config
        config_path = folder_path / 'config.yaml'
        if not config_path.exists():
            config_path = folder_path / 'config.json'

        if not config_path.exists():
            raise FileNotFoundError("目录中未找到 config.yaml 或 config.json。")

        # Load config
        if config_path.suffix == '.yaml':
            self.config = read_full_config(config_path)
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)

        # Patch config
        self._patch_config()

        # Check for checkpoint
        ckpt_path = folder_path / 'model.ckpt'
        if not ckpt_path.exists():
            ckpts = list(folder_path.glob('*.ckpt'))
            if ckpts:
                ckpt_path = ckpts[0]
            else:
                raise FileNotFoundError("目录中未找到 .ckpt 文件。")

        self.model, self.mel_transform = build_model_and_mel_transform(
            self.config,
            ckpt_path,
            self.device,
        )

        return self.config

    def _patch_config(self):
        if 'clip_grad_norm' not in self.config:
            self.config['clip_grad_norm'] = 1.0

        key_mapping = {
            'sampling_rate': 'audio_sample_rate',
            'num_mels': 'audio_num_mel_bins',
            'n_fft': 'fft_size',
        }
        for old_key, new_key in key_mapping.items():
            if old_key in self.config and new_key not in self.config:
                self.config[new_key] = self.config[old_key]

        if 'f0_min' not in self.config:
            self.config['f0_min'] = 40
        if 'f0_max' not in self.config:
            self.config['f0_max'] = 1600

        if 'model_args' not in self.config:
            model_arg_keys = [
                'mini_nsf',
                'upsample_rates',
                'upsample_kernel_sizes',
                'upsample_initial_channel',
                'resblock_kernel_sizes',
                'resblock_dilation_sizes',
                'resblock',
                'discriminator_periods',
            ]
            self.config['model_args'] = {}
            for key in model_arg_keys:
                if key in self.config:
                    self.config['model_args'][key] = self.config[key]

    def process_audio(self, file_path):
        """Load audio, resample, and extract features (Mel, F0).

        Returns:
            audio_np: numpy array of audio data (mono)
            sr: sample rate
            mel: Mel spectrogram tensor
            f0_midi: numpy array of F0 (MIDI, NaN for unvoiced)
            segments: list of (start, end) tuples in frames
        """
        if self.model is None or self.mel_transform is None:
            raise RuntimeError("请先加载模型以确保采样率正确。")

        target_sr = int(self.config['audio_sample_rate'])
        audio_t, sr = load_audio_mono_resample(file_path, target_sr)

        mel, f0_midi, segments = extract_mel_f0_segments(
            audio_t,
            config=self.config,
            mel_transform=self.mel_transform,
            key_shift=0.0,
        )

        return audio_t[0].numpy(), sr, mel, f0_midi, segments

    def synthesize_segment(self, mel, segment, f0_midi_segment):
        """Synthesize a specific segment with context padding to avoid artifacts."""
        if self.model is None:
            raise RuntimeError("模型未加载")

        hop_size = int(self.config['hop_size'])
        return synthesize_segment_with_padding(
            self.model,
            mel,
            segment,
            f0_midi_segment,
            device=self.device,
            hop_size=hop_size,
            pad_frames=64,
        )

    def synthesize(self, mel, f0_midi):
        """Synthesize full audio using the modified F0."""
        if self.model is None:
            raise RuntimeError("模型未加载")

        return synthesize_full(
            self.model,
            mel,
            f0_midi,
            device=self.device,
        )

    def _get_vslib_engine(self) -> VslibEngine:
        if self._vslib_engine is None:
            self._vslib_engine = VslibEngine()
        return self._vslib_engine

    def vslib_status(self) -> VslibStatus:
        """Return VSLIB availability and dll path without raising."""
        try:
            engine = self._get_vslib_engine()
            return engine.status
        except VslibUnavailableError as exc:
            return VslibStatus(False, None, exc.reason)

    def synthesize_full_vslib(
        self,
        audio: np.ndarray,
        sample_rate: int,
        f0_midi_original: np.ndarray,
        f0_midi_edited: np.ndarray,
    ) -> np.ndarray:
        """Synthesize audio with VSLIB using edited MIDI F0 while preserving consonants.

        We map user edits onto VSLIB control points. If a control point has no
        pitch change (original == edited within tolerance), we keep VSLIB's
        original pitch flags to avoid over-processing consonants.
        """
        if audio is None or f0_midi_edited is None:
            raise RuntimeError("缺少音频或音高数据，无法使用 VSLIB 合成")

        engine = self._get_vslib_engine()
        hop_size = int(self.config.get('hop_size', 512)) if self.config else 512

        audio_f = np.asarray(audio, dtype=np.float32)
        tmp_fd, tmp_path = tempfile.mkstemp(suffix='.wav', prefix='hifishifter_vslib_')
        os.close(tmp_fd)
        tmp_file = pathlib.Path(tmp_path)

        try:
            wav_int16 = (np.clip(audio_f, -1.0, 1.0) * 32767.0).astype(np.int16)
            wavfile.write(tmp_file, int(sample_rate), wav_int16)

            return engine.synthesize_from_pitch(
                tmp_file,
                f0_midi_original,
                f0_midi_edited,
                sample_rate=int(sample_rate),
                hop_size=hop_size,
            )
        except VslibError as exc:
            if exc.code == 6:
                # VSERR_FREQ: guide users to resample when VSLIB rejects current spec
                raise RuntimeError("VSLIB 不支持当前采样率或格式，请先转换为 44100Hz 16-bit WAV 再试。") from exc
            raise
        finally:
            try:
                tmp_file.unlink()
            except Exception:
                pass

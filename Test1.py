import json

import wave
from pathlib import Path
from subprocess import run

from typing import Optional


import IPython
import librosa
import librosa.core.spectrum
import librosa.display

import noisereduce
import numpy as np

# import scipy.signal

# import sounddevice
import soundfile

import vosk


def convert_to_wav(
    audio_file: Path,
    out_name: Optional[str] = None,
    out_dir: Optional[Path] = None,
) -> Path:
    """
    Convert audio to .wav format by ffmpeg

    :param audio_file: Path to audio
    :param out_name: Converted file name w/o suffix, defaults to {input file name}_out
    :param out_dir: Save directory for converted file , defaults to input directory
    :return: Path to converted file
    """

    if out_dir is None:

        out_dir = audio_file.parent

    if out_name is None:
        out_name = f'{audio_file.stem}_out.wav'

    out_file_name = out_dir / f'{out_name}.wav'

    command = [
        'ffmpeg',
        '-i',
        audio_file.as_posix(),
        '-c:v',
        'libx265',
        out_file_name.as_posix(),
    ]
    run(command, capture_output=True, check=True)
    return out_file_name


class Signal:
    "Class for audio wave"

    def __init__(self, wave: np.ndarray, sample_rate: int):
        self.wave = np.array(wave)
        self.sample_rate = int(sample_rate)

    @property
    def wave_dB(self):
        """Wave amplidudes in dB"""

        return librosa.amplitude_to_db(self.wave)

    @classmethod
    def open(cls, wav_path: Path) -> 'Signal':
        """
        Open .wav file

        :param wav_path: Path to .wav file
        :return: Signal object
        """
        wave, sample_rate = soundfile.read(wav_path.as_posix())
        return cls(wave=wave, sample_rate=sample_rate)

    def save(self, dir_path: Path, file_name: str) -> Path:
        """
        Save wave in .wav format

        :param dir_path: Save directory path
        :param name: File name w/o suffix
        :return: File path
        """
        save_path = dir_path / f'{file_name}.wav'
        soundfile.write(
            save_path.as_posix(),
            self.wave,
            self.sample_rate,
            format='wav',
            # subtype='PCM_24',
        )
        return save_path

    def play(self):
        """Play audio by Python.display.Audio"""

        return IPython.display.Audio(self.wave, rate=self.sample_rate)

    def resample(self, target_sample_rate=16_000) -> 'Signal':
        """
        Resample audio with target sample rate

        :param target_sample_rate: Target sample rate, defaults to 16_000
        :return: Signal object
        """
        resampled_wave = np.array(
            librosa.resample(
                y=self.wave,
                orig_sr=self.sample_rate,
                target_sr=target_sample_rate,
            )
        )
        return Signal(wave=resampled_wave, sample_rate=target_sample_rate)

    def show_wave(self):
        """Show amplitudes"""

        return librosa.display.waveshow(y=self.wave, sr=self.sample_rate)

    def show_spectrogramm(self, window_length: int = 2048):
        """
        Show spectrogramm with window length

        :param window_length: Window length to create spectrogramm, defaults to 2048
        """
        S = np.abs(librosa.stft(self.wave)) ** 2
        S_dB = librosa.amplitude_to_db(S)
        return librosa.display.specshow(
            data=S_dB, sr=self.sample_rate, x_axis='time', y_axis='linear', win_length=window_length
        )

    def show_mel_spectrogramm(self, n_mels: int = 128):
        """
        Show mel spectrogramm with window length

        :param n_mels: Number of mels on mel spectrogramm, defaults to 2048
        """
        S = librosa.feature.melspectrogram(y=self.wave, sr=self.sample_rate, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)

        return librosa.display.specshow(
            S_dB, x_axis='time', y_axis='mel', sr=self.sample_rate, fmax=8000
        )


def reduce_noise(signal: Signal, noise: Optional[Signal] = None) -> Signal:
    """
    Reduce noise by statistical methods with or w/o noise example

    :param signal: Input signal
    :param noise: Example noise signal, defaults to None
    :return: Noise reduced Signal object
    """
    static = False if noise is not None else True
    y_noise = noise.wave if noise is not None else None
    reduced_wave = noisereduce.reduce_noise(
        y=signal.wave,
        sr=signal.sample_rate,
        n_fft=512,
        y_noise=y_noise,
        stationary=static,
        use_tqdm=True,
        prop_decrease=0.95,
        freq_mask_smooth_hz=300,
        time_mask_smooth_ms=50,
    )
    return Signal(wave=np.array(reduced_wave), sample_rate=signal.sample_rate)


def filter_hight(signal: Signal):
    old_wave = signal.wave.copy()
    target = np.max(np.abs(old_wave))
    print(target)

    new_wave = np.where(np.abs(old_wave) > target * 0.2, 0, old_wave)
    return Signal(wave=new_wave, sample_rate=signal.sample_rate)


MODEL_PATH = r"D:\WORKS\TechTasks\vosk-model-ru-0.42"


class Recognizer:
    """Vosk recognizer class"""

    sample_rate = 8000
    """Model input sample rate"""

    def __init__(self, model_path: Path):
        self._model = vosk.Model(model_path.as_posix())
        self._recognizer = vosk.KaldiRecognizer(self._model, self.sample_rate)

    def __call__(self, file_path: Path):
        wav_file = wave.open(file_path.as_posix(), "rb")
        result = ''
        last_n = False
        while True:
            data = wav_file.readframes(self.sample_rate)
            if len(data) == 0:
                break

            if self._recognizer.AcceptWaveform(data):
                res = json.loads(self._recognizer.Result())
                if res['text'] != '':
                    result += f" {res['text']}"
                    last_n = False
                elif not last_n:
                    result += '\n'
                    last_n = True

        res = json.loads(self._recognizer.FinalResult())
        result += f"{res['text']}"
        return result


print('Prepare')
raw1 = Signal.open(Path('audio_out.wav'))
raw2 = Signal.open(Path('noise_out.wav'))

noise = Signal.open(Path('noise_out.wav'))
noise.wave = noise.wave[: 30 * noise.sample_rate]


reduced = filter_hight(reduce_noise(raw1, noise))
resampled_path = raw1.resample(8000).save(Path('.'), 'ForModel')

print(resampled_path)

print('Create model')
recognizer = Recognizer(Path(r'vosk-model-ru-0.42'))


print('Recognize')
text = recognizer(resampled_path)
print(text)

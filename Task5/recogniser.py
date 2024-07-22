import json
import wave
from pathlib import Path
from subprocess import run
from tempfile import TemporaryDirectory
from typing import Optional

import IPython
import librosa
import librosa.core.spectrum
import librosa.display
import noisereduce
import numpy as np
import pyaudio
import scipy
import scipy.signal
import soundfile
import vosk


def convert_to_wav(
    audio_file: Path,
    out_fname: Optional[str] = None,
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

    if out_fname is None:
        out_fname = f'{audio_file.stem}_converted.wav'

    out_fpath = out_dir / Path(out_fname)

    command = [
        'ffmpeg',
        '-i',
        audio_file.as_posix(),
        '-c:v',
        'libx265',
        out_fpath.as_posix(),
    ]
    run(command, capture_output=True, check=True)
    return out_fpath


class Signal:
    "Class for audio wave"

    def __init__(self, wave: np.ndarray, sample_rate: int):
        self.wave = np.array(wave)
        self.sample_rate = int(sample_rate)

    def __len__(self):
        return round(len(self.wave) / self.sample_rate * 1000)

    def __hash__(self):
        return hash((self.wave, self.sample_rate))

    def __repr__(self) -> str:
        return f'<Signal sr= {self.sample_rate}, len={len(self)}'

    @property
    def wave_dB(self):
        """Wave amplidudes in dB"""

        return librosa.amplitude_to_db(self.wave)

    @classmethod
    def load(cls, audio_path: Path) -> 'Signal':
        """
        Open audio file

        :param wav_path: Path to .wav file
        :return: Signal object
        """
        if audio_path.suffix != 'wav':
            with TemporaryDirectory() as temp_dir:
                audio_path = convert_to_wav(audio_path, out_dir=Path(temp_dir))

                wave, sample_rate = soundfile.read(audio_path.as_posix())
        return cls(wave=wave, sample_rate=sample_rate)

    def save(self, dir_path: Path, file_name: str) -> Path:
        """
        Save wave in .wav format

        :param dir_path: Save directory path
        :param name: File name w/o suffix
        :return: File path
        """
        save_path = dir_path / Path(f'{file_name}.wav')
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

    def show_mel_spectrogramm(self, n_mels: int = 64):
        """
        Show mel spectrogramm with window length

        :param n_mels: Number of mels on mel spectrogramm, defaults to 2048
        """
        S = librosa.feature.melspectrogram(y=self.wave, sr=self.sample_rate, n_mels=n_mels)
        S_dB = librosa.power_to_db(S, ref=np.max)

        return librosa.display.specshow(
            S_dB,
            x_axis='time',
            y_axis='mel',
            sr=self.sample_rate,
        )


class Filters:

    @staticmethod
    def hight_amplitudes(signal: Signal):
        """Reduce very hight noise in audio"""

        old_wave = signal.wave.copy()
        target = np.quantile(np.abs(signal.wave), 0.999)
        print(target)
        new_wave = np.where(np.abs(old_wave) > target, 0, old_wave)
        return Signal(wave=new_wave, sample_rate=signal.sample_rate)

    @staticmethod
    def hightpass(signal: Signal, threshold: int = 200):
        """Hightpass filter"""

        b, a = scipy.signal.butter(10, threshold / (signal.sample_rate) * 4, btype='highpass')
        yf = scipy.signal.lfilter(b, a, signal.wave)
        return Signal(yf, signal.sample_rate)

    @staticmethod
    def lowpass(signal: Signal, threshold: int = 4000):
        """Hightpass filter"""

        b, a = scipy.signal.butter(10, threshold / (signal.sample_rate) * 2, btype='lowpass')
        yf = scipy.signal.lfilter(b, a, signal.wave)
        return Signal(yf, signal.sample_rate)

    @staticmethod
    def reduce_noise(signal: Signal, noise: Optional[Signal] = None) -> Signal:
        """
        Reduce noise by statistical methods with or w/o noise example

        :param signal: Input signal
        :param noise: Example noise signal, defaults to None
        :return: Noise reduced Signal object
        """
        reduced_wave = noisereduce.reduce_noise(
            y=signal.wave,
            sr=signal.sample_rate,
            n_fft=1024,
            y_noise=noise.wave if noise is not None else None,
            stationary=False,
            use_tqdm=True,
            prop_decrease=0.9,
            freq_mask_smooth_hz=300,
            time_mask_smooth_ms=12,
        )
        return Signal(wave=np.array(reduced_wave), sample_rate=signal.sample_rate)


class Recognizer:
    """Vosk recognizer class"""

    def __init__(self, model: vosk.Model, model_sample_rate: int):
        self.model_sample_rate = model_sample_rate
        self._rec = vosk.KaldiRecognizer(model, model_sample_rate)

    def _recognize(self, data) -> str:
        answer = {'text': ''}
        if self._rec.AcceptWaveform(data):
            answer = json.loads(self._rec.Result())
        return answer['text']

    def listen(self, stream: pyaudio.PyAudio.Stream, sample_rate: Optional[int] = 4000):
        """
        Listen audio input

        :param stream: pyaudio.PyAudio.Stream channel
        :param sample_rate: Sample rate for listen stream, see documentation, defaults to 4000
        :return: Generator if say 'остановить поток' stream wil be stopped
        """
        if sample_rate is None:
            sample_rate = self.model_sample_rate // 4
        while True:
            data = stream.read(sample_rate, exception_on_overflow=False)
            res = self._recognize(data)
            if res:
                if 'остановить поток' in res:
                    stream.stop_stream()
                    return 0
                yield res

    def recognize_signal(self, signal: Signal) -> str:
        """Recognize signal to text"""

        with TemporaryDirectory() as temp_dir:
            signal_path = signal.resample(target_sample_rate=self.model_sample_rate).save(
                Path(temp_dir), 'temp_wav'
            )
            text = ''
            with wave.Wave_read(signal_path.as_posix()) as wav_file:
                while True:
                    data = wav_file.readframes(self.model_sample_rate)
                    if not len(data):
                        break

                    res = self._recognize(data)
                    text = text + '\n' + res if res else text
            res = json.loads(self._rec.FinalResult())['text']
        text = text + '\n' + res if res else text
        return text.strip()

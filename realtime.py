#!/usr/bin/env python3
"""
Локальная озвучка русского текста в реальном времени.

Движки:
  xtts   - Coqui XTTS-v2 на GPU (самый естественный звук, потоковая генерация)
  silero - Silero v5_ru на CPU (быстрый и надёжный запасной вариант)
  auto   - пробует XTTS, при нехватке видеопамяти/ошибке переходит на Silero

Установка (Windows, Python 3.10-3.12):
  py -3.11 -m venv venv
  venv\\Scripts\\activate
  pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
  pip install coqui-tts sounddevice numpy num2words

Запуск:
  python tts_realtime.py                       # интерактивный режим
  python tts_realtime.py --text "Привет, мир!"
  python tts_realtime.py --file book.txt
  type book.txt | python tts_realtime.py       # через stdin
  python tts_realtime.py --speaker-wav my_voice.wav   # свой голос (6-10 с, ru)

Как это работает: текст режется на предложения, XTTS отдаёт звук кусочками
по мере генерации, а воспроизведение начинается, как только накоплен
небольшой буфер. Поэтому звук идёт почти сразу, не дожидаясь конца текста.
"""
from __future__ import annotations

import argparse
import collections
import gc
import os
import re
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
import sounddevice as sd

# ----------------------------------------------------------------------------
# Подготовка текста
# ----------------------------------------------------------------------------
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def normalize(text: str) -> str:
    """Числа -> слова (только именительный падеж, без согласования)."""
    try:
        from num2words import num2words
    except ImportError:
        return text
    return re.sub(r"\d+", lambda m: num2words(int(m.group()), lang="ru"), text)


def split_text(text: str, max_len: int = 180, min_len: int = 40) -> list[str]:
    """Делит текст на предложения; слишком длинные режет по запятым/пробелам,
    слишком короткие склеивает (XTTS плохо работает на огрызках)."""
    text = re.sub(r"\s+", " ", normalize(text)).strip()
    parts: list[str] = []
    for s in SENTENCE_END.split(text):
        s = s.strip()
        while len(s) > max_len:
            cut = -1
            for sep in (", ", "; ", ": ", " — ", " "):
                cut = s.rfind(sep, min_len, max_len)
                if cut != -1:
                    break
            if cut == -1:
                cut = max_len - 1
            parts.append(s[: cut + 1].strip())
            s = s[cut + 1:].strip()
        if s:
            parts.append(s)

    merged: list[str] = []
    buf = ""
    for p in parts:
        if buf and len(buf) < min_len and len(buf) + 1 + len(p) <= max_len:
            buf += " " + p
        else:
            if buf:
                merged.append(buf)
            buf = p
    if buf:
        if merged and len(buf) < min_len and len(merged[-1]) + 1 + len(buf) <= max_len:
            merged[-1] += " " + buf
        else:
            merged.append(buf)
    return merged


# ----------------------------------------------------------------------------
# Движки синтеза
# ----------------------------------------------------------------------------
class XTTSEngine:
    sample_rate = 24000

    def __init__(self, speaker_wav=None, speaker_name="Claribel Dervla",
                 speed=1.0, fp16=False):
        os.environ.setdefault("COQUI_TOS_AGREED", "1")  # лицензия CPML (некоммерч.)
        import torch
        from TTS.api import TTS

        self.torch = torch
        self.speed = speed
        tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
        self.model = tts.synthesizer.tts_model
        self.model.to("cuda")
        self.model.eval()

        self.fp16 = False
        self.speaker_name = speaker_name
        self.set_voice(speaker_wav)
        if fp16:  # экспериментально: экономит память, но возможны артефакты
            self.model.half()
            self.lat, self.emb = self.lat.half(), self.emb.half()
            self.fp16 = True

        self.vram_mb = torch.cuda.memory_allocated() / 1024**2
        print(f"XTTS-v2 загружен, занято VRAM: {self.vram_mb:.0f} МБ")

    def set_voice(self, speaker_wav=None):
        """Сменить голос без перезагрузки модели (в режиме fp16 недоступно)."""
        if self.fp16:
            raise RuntimeError("В режиме fp16 голос меняется только перезагрузкой модели")
        if speaker_wav:
            lat, emb = self.model.get_conditioning_latents(audio_path=[speaker_wav])
        else:
            sp = self.model.speaker_manager.speakers[self.speaker_name]
            lat, emb = sp["gpt_cond_latent"], sp["speaker_embedding"]
        self.lat, self.emb = lat.to("cuda"), emb.to("cuda")
        self.voice_key = speaker_wav

    def synth(self, text: str):
        with self.torch.inference_mode():
            for chunk in self.model.inference_stream(
                text, "ru", self.lat, self.emb,
                stream_chunk_size=20,
                temperature=0.65,
                speed=self.speed,
                enable_text_splitting=False,
            ):
                yield chunk.squeeze().float().cpu().numpy()


class SileroEngine:
    sample_rate = 48000
    URL = "https://models.silero.ai/models/tts/ru/v5_ru.pt"

    def __init__(self, speaker="xenia"):
        import torch

        self.speaker = speaker
        self.torch = torch
        torch.set_num_threads(max(2, (os.cpu_count() or 4) // 2))
        path = Path.home() / ".cache" / "silero" / "v5_ru.pt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            print("Скачиваю модель Silero (~100 МБ)...")
            urllib.request.urlretrieve(self.URL, path)
        self.model = torch.package.PackageImporter(str(path)).load_pickle(
            "tts_models", "model")
        self.model.to("cpu")
        print(f"Silero v5_ru загружен (голос: {speaker}, CPU)")

    def synth(self, text: str):
        with self.torch.inference_mode():
            audio = self.model.apply_tts(
                text=text, speaker=self.speaker,
                sample_rate=self.sample_rate, put_accent=True, put_yo=True)
        yield audio.numpy().astype(np.float32)


def build_engine(args):
    if args.engine in ("auto", "xtts"):
        error = None
        try:
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA недоступна")
            print("Загружаю XTTS-v2 (при первом запуске скачается ~1.8 ГБ)...")
            return XTTSEngine(args.speaker_wav, speed=args.speed, fp16=args.fp16)
        except Exception as e:  # noqa: BLE001
            if args.engine == "xtts":
                raise
            error = f"{type(e).__name__}: {e}"
        print(f"[!] XTTS не запустился ({error}).\n    Перехожу на Silero (CPU).")
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
    return SileroEngine(args.silero_speaker)


# ----------------------------------------------------------------------------
# Воспроизведение без щелчков: общий буфер + callback звуковой карты
# ----------------------------------------------------------------------------
class Player:
    def __init__(self, sample_rate: int, prebuffer_sec: float):
        self.chunks: collections.deque[np.ndarray] = collections.deque()
        self.buffered = 0
        self.lock = threading.Lock()
        self.playing = False
        self.prebuffer = int(prebuffer_sec * sample_rate)
        self.stream = sd.OutputStream(
            samplerate=sample_rate, channels=1, dtype="float32",
            callback=self._callback)
        self.stream.start()

    def _callback(self, outdata, frames, time_info, status):
        out = outdata[:, 0]
        out[:] = 0.0
        if not self.playing:
            return
        n = 0
        with self.lock:
            while n < frames and self.chunks:
                c = self.chunks[0]
                take = min(len(c), frames - n)
                out[n:n + take] = c[:take]
                n += take
                if take == len(c):
                    self.chunks.popleft()
                else:
                    self.chunks[0] = c[take:]
            self.buffered -= n

    def feed(self, audio: np.ndarray):
        audio = np.ascontiguousarray(audio, dtype=np.float32)
        with self.lock:
            self.chunks.append(audio)
            self.buffered += len(audio)
            if not self.playing and self.buffered >= self.prebuffer:
                self.playing = True

    def finish(self):
        with self.lock:
            self.playing = True  # проиграть остаток, даже если буфер маленький
        while True:
            with self.lock:
                if self.buffered <= 0:
                    break
            time.sleep(0.03)
        time.sleep(float(self.stream.latency) + 0.15)

    def abort(self):
        self.stream.abort()

    def close(self):
        self.stream.close()


def speak(engine, text: str, prebuffer: float):
    sentences = split_text(text)
    if not sentences:
        return
    player = Player(engine.sample_rate, prebuffer)
    t0 = time.perf_counter()
    first = None
    audio_sec = 0.0
    try:
        for sentence in sentences:
            for chunk in engine.synth(sentence):
                if first is None:
                    first = time.perf_counter() - t0
                audio_sec += len(chunk) / engine.sample_rate
                player.feed(chunk)
        gen_time = time.perf_counter() - t0
        player.finish()
    except KeyboardInterrupt:
        player.abort()
        print("\n[остановлено]")
        return
    finally:
        player.close()

    if audio_sec > 0 and first is not None:
        rtf = gen_time / audio_sec
        print(f"  первый звук через {first:.2f} с | RTF {rtf:.2f}"
              + ("  (медленнее реального времени, будут паузы)" if rtf > 1 else ""))


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Локальная русская озвучка в реальном времени")
    ap.add_argument("--engine", choices=["auto", "xtts", "silero"], default="auto")
    ap.add_argument("--text", help="озвучить этот текст и выйти")
    ap.add_argument("--file", help="озвучить текстовый файл (UTF-8) и выйти")
    ap.add_argument("--speaker-wav", help="образец голоса для XTTS (6-10 с, русская речь)")
    ap.add_argument("--silero-speaker", default="xenia",
                    choices=["aidar", "baya", "kseniya", "xenia", "eugene"])
    ap.add_argument("--speed", type=float, default=1.0, help="скорость речи XTTS")
    ap.add_argument("--prebuffer", type=float, default=0.8,
                    help="сколько секунд звука накопить перед стартом (больше = меньше пауз)")
    ap.add_argument("--fp16", action="store_true", help="XTTS в fp16 (меньше VRAM, экспериментально)")
    args = ap.parse_args()

    for stream in (sys.stdout, sys.stdin):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass

    engine = build_engine(args)

    if args.text:
        speak(engine, args.text, args.prebuffer)
    elif args.file:
        speak(engine, Path(args.file).read_text(encoding="utf-8"), args.prebuffer)
    elif not sys.stdin.isatty():
        speak(engine, sys.stdin.read(), args.prebuffer)
    else:
        print("\nВведите текст и нажмите Enter. Ctrl+C во время речи - стоп, "
              "пустая строка - выход.")
        while True:
            try:
                line = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                break
            speak(engine, line, args.prebuffer)


if __name__ == "__main__":
    main()
    
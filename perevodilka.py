"""
Голосовой переводчик RU <-> EN. Работает полностью локально.

STT (речь -> текст):  Whisper (openai/whisper-medium, можно small)
Перевод:              Helsinki-NLP opus-mt-ru-en / opus-mt-en-ru
Озвучка (TTS):        Silero TTS (русский и английский)

Модели скачиваются автоматически при первом запуске и дальше лежат на диске.
"""
import threading
import tkinter as tk
from tkinter import ttk

import numpy as np
import sounddevice as sd
import torch
from transformers import MarianMTModel, MarianTokenizer, pipeline

# ---------------- настройки ----------------
WHISPER_MODEL = "openai/whisper-medium"   # для слабых ПК: "openai/whisper-small"
REC_SR = 16000                            # частота записи для Whisper
TTS_SR = 48000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

LANGS = {
    "ru": {"name": "russian", "tts_lang": "ru", "tts_model": "v4_ru", "speaker": "xenia"},
    "en": {"name": "english", "tts_lang": "en", "tts_model": "v3_en", "speaker": "en_0"},
}


class Engine:
    """Загружает все модели и делает STT / перевод / озвучку."""

    def __init__(self, log):
        self.log = log
        self.asr = None
        self.mt = {}
        self.tts = {}

    def load(self):
        self.log(f"Загрузка Whisper ({WHISPER_MODEL}) на {DEVICE}…")
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=DTYPE,
        )
        for pair in ("ru-en", "en-ru"):
            self.log(f"Загрузка переводчика {pair}…")
            name = f"Helsinki-NLP/opus-mt-{pair}"
            tok = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name).to(DEVICE, dtype=DTYPE).eval()
            self.mt[pair] = (tok, model)
        for code, cfg in LANGS.items():
            self.log(f"Загрузка голоса ({code})…")
            res = torch.hub.load(
                "snakers4/silero-models", "silero_tts",
                language=cfg["tts_lang"], speaker=cfg["tts_model"], trust_repo=True,
            )
            model = res[0] if isinstance(res, tuple) else res
            model.to("cpu")
            self.tts[code] = model
        self.log("Готово. Нажмите кнопку записи.")

    def transcribe(self, audio, lang):
        out = self.asr(
            {"raw": audio, "sampling_rate": REC_SR},
            generate_kwargs={"language": LANGS[lang]["name"], "task": "transcribe"},
        )
        return out["text"].strip()

    def translate(self, text, src, dst):
        tok, model = self.mt[f"{src}-{dst}"]
        inputs = tok(text, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            out = model.generate(**inputs, num_beams=4, max_new_tokens=256)
        return tok.decode(out[0], skip_special_tokens=True)

    def speak(self, text, lang):
        cfg = LANGS[lang]
        audio = self.tts[lang].apply_tts(
            text=text, speaker=cfg["speaker"], sample_rate=TTS_SR
        )
        sd.play(audio.numpy(), TTS_SR)
        sd.wait()


class App:
    def __init__(self, root):
        self.root = root
        root.title("Голосовой переводчик RU ↔ EN")
        root.geometry("900x420")
        root.columnconfigure((0, 1), weight=1)
        root.rowconfigure(0, weight=1)

        self.texts, self.buttons = {}, {}
        for col, (code, title, btn) in enumerate(
            [("ru", "Русский", "🎤 Запись (Русский)"), ("en", "English", "🎤 Record (English)")]
        ):
            frame = ttk.LabelFrame(root, text=title, padding=8)
            frame.grid(row=0, column=col, sticky="nsew", padx=8, pady=8)
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            t = tk.Text(frame, wrap="word", font=("Segoe UI", 13), height=10)
            t.grid(row=0, column=0, sticky="nsew")
            b = ttk.Button(frame, text=btn, command=lambda c=code: self.toggle(c))
            b.grid(row=1, column=0, sticky="ew", pady=(8, 0))
            b.state(["disabled"])
            self.texts[code], self.buttons[code] = t, b

        self.status = tk.StringVar(value="Запуск…")
        ttk.Label(root, textvariable=self.status).grid(
            row=1, column=0, columnspan=2, sticky="w", padx=10, pady=(0, 8)
        )

        self.labels = {"ru": "🎤 Запись (Русский)", "en": "🎤 Record (English)"}
        self.recording = None
        self.frames = []
        self.stream = None
        self.engine = Engine(self.set_status)
        threading.Thread(target=self._load, daemon=True).start()

    # ---------- вспомогательное ----------
    def ui(self, fn):
        self.root.after(0, fn)

    def set_status(self, msg):
        self.ui(lambda: self.status.set(msg))

    def set_text(self, code, text):
        def f():
            w = self.texts[code]
            w.delete("1.0", "end")
            w.insert("1.0", text)
        self.ui(f)

    def set_buttons(self, enabled):
        def f():
            for b in self.buttons.values():
                b.state(["!disabled"] if enabled else ["disabled"])
        self.ui(f)

    def _load(self):
        try:
            self.engine.load()
            self.set_buttons(True)
        except Exception as e:
            self.set_status(f"Ошибка загрузки моделей: {e}")

    # ---------- запись ----------
    def toggle(self, code):
        if self.recording is None:
            self.start(code)
        elif self.recording == code:
            self.stop()

    def start(self, code):
        self.recording = code
        self.frames = []
        other = "en" if code == "ru" else "ru"
        self.buttons[other].state(["disabled"])
        self.buttons[code].config(text="⏹ Стоп")
        self.status.set("Слушаю… нажмите «Стоп», когда закончите.")

        def cb(indata, frames, time, status):
            self.frames.append(indata.copy())

        self.stream = sd.InputStream(
            samplerate=REC_SR, channels=1, dtype="float32", callback=cb
        )
        self.stream.start()

    def stop(self):
        code = self.recording
        self.stream.stop()
        self.stream.close()
        self.recording = None
        self.buttons[code].config(text=self.labels[code])
        self.set_buttons(False)
        audio = (
            np.concatenate(self.frames).flatten()
            if self.frames else np.zeros(0, dtype="float32")
        )
        threading.Thread(target=self.pipeline, args=(code, audio), daemon=True).start()

    # ---------- основной конвейер ----------
    def pipeline(self, src, audio):
        dst = "en" if src == "ru" else "ru"
        try:
            if len(audio) < REC_SR * 0.3:
                self.set_status("Запись слишком короткая.")
                return
            self.set_status("Распознаю речь…")
            text = self.engine.transcribe(audio, src)
            if not text:
                self.set_status("Речь не распознана.")
                return
            self.set_text(src, text)

            self.set_status("Перевожу…")
            translated = self.engine.translate(text, src, dst)
            self.set_text(dst, translated)

            self.set_status("Озвучиваю перевод…")
            self.engine.speak(translated, dst)
            self.set_status("Готово.")
        except Exception as e:
            self.set_status(f"Ошибка: {e}")
        finally:
            self.set_buttons(True)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
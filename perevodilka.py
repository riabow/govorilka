"""
Голосовой переводчик RU <-> EN. Работает полностью локально.

STT (речь -> текст):  Whisper (openai/whisper-medium, можно small)
Перевод:              Helsinki-NLP opus-mt-ru-en / opus-mt-en-ru
Озвучка (TTS):        Silero TTS (русский и английский)

Модели скачиваются автоматически при первом запуске и дальше лежат на диске.
"""
import re
import threading
import unicodedata
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

SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

LANGS = {
    "ru": {"name": "russian", "tts_lang": "ru", "tts_model": "v4_ru", "speaker": "xenia"},
    "en": {"name": "english", "tts_lang": "en", "tts_model": "v3_en", "speaker": "en_0"},
}


# ---------------- очистка текста ----------------
_BAD_CATS = {"So", "Sk", "Cs", "Co", "Cn", "Cf", "Cc", "Me"}


def clean_text(text):
    """Убирает эмодзи, значки (🔊 ✓ ★ →…), невидимые символы и мусор, которые ломают модели."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("™", "").replace("®", "").replace("©", "")
    text = unicodedata.normalize("NFKC", text)
    out = []
    for ch in text:
        if ch == "\n":
            out.append("\n")
        elif ch == "\t":
            out.append(" ")
        elif (ord(ch) > 0xFFFF or 0xFE00 <= ord(ch) <= 0xFE0F or ch == "\u200d"
              or 0x2190 <= ord(ch) <= 0x2BFF):  # стрелки, тех. символы, фигуры, дингбаты
            continue
        elif unicodedata.category(ch) in _BAD_CATS:
            continue
        else:
            out.append(ch)
    text = "".join(out)
    text = re.sub(r"[ \u00a0]+", " ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))  # переносы и пустые строки сохраняются
    return text.strip()


def has_words(text):
    return any(ch.isalnum() for ch in text)


def split_chunks(text, limit):
    """Делит текст на куски по предложениям, чтобы модели не обрезали длинный текст."""
    parts = re.split(r"(?<=[.!?…])\s+|\n+", text)
    chunks, cur = [], ""
    for part in parts:
        part = part.strip()
        while len(part) > limit:
            cut = part.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(part[:cut].strip())
            part = part[cut:].strip()
        if not part:
            continue
        if cur and len(cur) + len(part) + 1 > limit:
            chunks.append(cur)
            cur = part
        else:
            cur = f"{cur} {part}".strip()
    if cur:
        chunks.append(cur)
    return [c for c in chunks if has_words(c)]


class Player:
    """Проигрыватель с паузой, остановкой и перемоткой (на базе sounddevice)."""

    def __init__(self):
        self.data = np.zeros(1, dtype="float32")
        self.pos = 0
        self.sr = TTS_SR
        self.paused = False
        self.stop_flag = False

    def load(self, audio, sr):
        self.data = np.asarray(audio, dtype="float32")
        self.sr = sr
        self.pos = 0
        self.paused = False
        self.stop_flag = False

    def run(self):
        """Блокирует поток, пока звук не доиграет или не будет остановлен."""
        done = threading.Event()

        def cb(outdata, frames, time_info, status):
            if self.stop_flag:
                outdata.fill(0)
                raise sd.CallbackStop
            if self.paused:
                outdata.fill(0)
                return
            end = min(self.pos + frames, len(self.data))
            n = end - self.pos
            if n > 0:
                outdata[:n, 0] = self.data[self.pos:end]
            if n < frames:
                outdata[max(n, 0):] = 0
            self.pos = end
            if end >= len(self.data):
                raise sd.CallbackStop

        with sd.OutputStream(samplerate=self.sr, channels=1, dtype="float32",
                             callback=cb, finished_callback=done.set):
            done.wait()


def fmt_time(sec):
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


class Engine:
    """Загружает все модели и делает STT / перевод / озвучку."""

    def __init__(self, log):
        self.log = log
        self.asr = None
        self.mt = {}
        self.tts = {}

    def load(self, stage):
        """stage(сообщение, процент_начала, процент_конца) — для индикатора загрузки."""
        stage(f"Загрузка распознавания речи Whisper ({WHISPER_MODEL}) на {DEVICE}…  (1/5)", 0, 55)
        self.asr = pipeline(
            "automatic-speech-recognition",
            model=WHISPER_MODEL,
            device=0 if DEVICE == "cuda" else -1,
            torch_dtype=DTYPE,
        )
        mt_stages = {"ru-en": (55, 68), "en-ru": (68, 81)}
        for i, pair in enumerate(("ru-en", "en-ru"), start=2):
            stage(f"Загрузка переводчика {pair}…  ({i}/5)", *mt_stages[pair])
            name = f"Helsinki-NLP/opus-mt-{pair}"
            tok = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name).to(DEVICE, dtype=DTYPE).eval()
            self.mt[pair] = (tok, model)
        tts_stages = {"ru": (81, 91), "en": (91, 100)}
        for i, (code, cfg) in enumerate(LANGS.items(), start=4):
            stage(f"Загрузка голоса ({code})…  ({i}/5)", *tts_stages[code])
            res = torch.hub.load(
                "snakers4/silero-models", "silero_tts",
                language=cfg["tts_lang"], speaker=cfg["tts_model"], trust_repo=True,
            )
            model = res[0] if isinstance(res, tuple) else res
            model.to("cpu")
            self.tts[code] = model

    def transcribe(self, audio, lang):
        out = self.asr(
            {"raw": audio, "sampling_rate": REC_SR},
            generate_kwargs={"language": LANGS[lang]["name"], "task": "transcribe"},
        )
        return out["text"].strip()

    def translate(self, text, src, dst):
        """Переводит построчно: переносы строк и пустые строки сохраняются."""
        text = clean_text(text)
        if not has_words(text):
            raise ValueError("В тексте нет слов для перевода.")
        tok, model = self.mt[f"{src}-{dst}"]
        out_lines = []
        for line in text.split("\n"):
            if not has_words(line):          # пустая строка (или только знаки, напр. «---») — оставляем как есть
                out_lines.append(line)
                continue
            parts = []
            for chunk in split_chunks(line, 400):
                inputs = tok(chunk, return_tensors="pt", truncation=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    out = model.generate(**inputs, num_beams=4, max_new_tokens=400)
                parts.append(tok.decode(out[0], skip_special_tokens=True))
            out_lines.append(" ".join(parts))
        return "\n".join(out_lines).strip()

    def _tts_chunk(self, chunk, lang):
        cfg = LANGS[lang]
        try:
            return self.tts[lang].apply_tts(text=chunk, speaker=cfg["speaker"], sample_rate=TTS_SR)
        except Exception:
            # запасной вариант: оставляем только буквы нужного языка, цифры и знаки препинания
            if lang == "ru":
                strict = re.sub(r"[^А-Яа-яЁё0-9\s.,!?:;\-—()%]", " ", chunk)
            else:
                strict = re.sub(r"[^A-Za-z0-9\s.,!?:;\-()%']", " ", chunk)
            strict = re.sub(r"\s+", " ", strict).strip()
            if not has_words(strict):
                return None
            return self.tts[lang].apply_tts(text=strict, speaker=cfg["speaker"], sample_rate=TTS_SR)

    def synthesize(self, text, lang, progress, cancel):
        """Генерирует звук по частям; progress(0..1) и cancel (Event) — для интерфейса."""
        chunks = split_chunks(clean_text(text), 250)
        if not chunks:
            raise ValueError("В тексте нет слов для озвучки.")
        pieces = []
        pause = torch.zeros(int(TTS_SR * 0.15))
        for i, chunk in enumerate(chunks, start=1):
            if cancel.is_set():
                return None
            audio = self._tts_chunk(chunk, lang)
            if audio is not None:
                pieces.extend([audio, pause])
            progress(i / len(chunks))
        if not pieces:
            raise ValueError("В тексте нет слов для озвучки.")
        return torch.cat(pieces).numpy()


class App:
    def __init__(self, root):
        self.root = root
        root.title("Голосовой переводчик RU ↔ EN")
        root.geometry("900x470")
        root.columnconfigure((0, 1), weight=1)
        root.rowconfigure(0, weight=1)

        self.texts, self.buttons, self.tbuttons, self.sbuttons = {}, {}, {}, {}
        for col, (code, title, btn) in enumerate(
            [("ru", "Русский", "🎤 Запись (Русский)"), ("en", "English", "🎤 Record (English)")]
        ):
            frame = ttk.LabelFrame(root, text=title, padding=8)
            frame.grid(row=0, column=col, sticky="nsew", padx=8, pady=8)
            frame.columnconfigure(0, weight=1)
            frame.rowconfigure(0, weight=1)
            t = tk.Text(frame, wrap="word", font=("Segoe UI", 13), height=10, undo=True)
            t.grid(row=0, column=0, sticky="nsew")
            self.enable_clipboard(t)
            row = ttk.Frame(frame)
            row.grid(row=1, column=0, sticky="ew", pady=(8, 0))
            row.columnconfigure((0, 1, 2), weight=1)
            b = ttk.Button(row, text=btn, command=lambda c=code: self.toggle(c))
            b.grid(row=0, column=0, sticky="ew", padx=(0, 4))
            tb = ttk.Button(row, text="Перевести", command=lambda c=code: self.translate_text(c))
            tb.grid(row=0, column=1, sticky="ew", padx=4)
            sb = ttk.Button(row, text="🔊 Озвучить", command=lambda c=code: self.speak_text(c))
            sb.grid(row=0, column=2, sticky="ew", padx=(4, 0))
            for x in (b, tb, sb):
                x.state(["disabled"])
            self.texts[code], self.buttons[code], self.tbuttons[code] = t, b, tb
            self.sbuttons[code] = sb

        self.status = tk.StringVar(value="Запуск…")
        self.pct_var = tk.StringVar(value="0%")
        bar = ttk.Frame(root)
        bar.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 4))
        bar.columnconfigure(1, weight=1)
        self.spin_lbl = ttk.Label(bar, text="", width=2, font=("Segoe UI Symbol", 16))
        self.spin_lbl.grid(row=0, column=0)
        ttk.Label(bar, textvariable=self.status).grid(row=0, column=1, sticky="w", padx=(4, 0))
        ttk.Label(bar, textvariable=self.pct_var).grid(row=0, column=2, sticky="e")
        self.progress = ttk.Progressbar(root, mode="determinate", maximum=100)
        self.progress.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))

        # панель проигрывателя (видна только во время озвучки)
        self.pframe = ttk.Frame(root)
        self.pframe.grid(row=3, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        self.pframe.columnconfigure(2, weight=1)
        self.pause_btn = ttk.Button(self.pframe, text="⏸ Пауза", width=14, command=self.toggle_pause)
        self.pause_btn.grid(row=0, column=0, padx=(0, 6))
        self.stop_btn = ttk.Button(self.pframe, text="⏹ Стоп", width=10, command=self.stop_speech)
        self.stop_btn.grid(row=0, column=1, padx=(0, 10))
        self.play_bar = ttk.Progressbar(self.pframe, mode="determinate", maximum=100)
        self.play_bar.grid(row=0, column=2, sticky="ew")
        self.play_bar.bind("<Button-1>", self.seek)   # клик по полосе — перемотка
        self.time_var = tk.StringVar(value="")
        ttk.Label(self.pframe, textvariable=self.time_var, width=22, anchor="e").grid(
            row=0, column=3, padx=(10, 0))
        self.pframe.grid_remove()
        self.player = Player()
        self.play_state = "idle"          # idle / synth / play
        self.synth_frac = 0.0
        self.cancel = threading.Event()

        # состояние загрузки / занятости
        self.loading = True
        self.busy = True
        self.spin_i = 0
        self.p_shown = 0.0
        self.p_next = 0
        for t in self.texts.values():       # пока модели грузятся, поля недоступны
            t.config(state="disabled")

        self.labels = {"ru": "🎤 Запись (Русский)", "en": "🎤 Record (English)"}
        self.recording = None
        self.frames = []
        self.stream = None
        self.engine = Engine(self.set_status)
        self.tick()
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
            self.busy = not enabled
            for b in (list(self.buttons.values()) + list(self.tbuttons.values())
                      + list(self.sbuttons.values())):
                b.state(["!disabled"] if enabled else ["disabled"])
        self.ui(f)

    def stage(self, msg, start, end):
        def f():
            self.status.set(msg)
            self.p_shown = max(self.p_shown, start)
            self.p_next = end
        self.ui(f)

    # ---------- озвучка с прогрессом, паузой и остановкой ----------
    def speak(self, text, lang):
        """Вызывается из рабочего потока: готовит звук, затем проигрывает."""
        self.cancel = threading.Event()
        self.synth_frac = 0.0
        self.ui(self._player_show)
        try:
            audio = self.engine.synthesize(
                text, lang, lambda f: setattr(self, "synth_frac", f), self.cancel
            )
            if audio is None or self.cancel.is_set():
                return
            self.player.load(audio, TTS_SR)
            self.ui(self._enter_play)
            self.player.run()
        finally:
            self.ui(self._player_hide)

    def _player_show(self):
        self.play_state = "synth"
        self.pause_btn.config(text="⏸ Пауза")
        self.pause_btn.state(["disabled"])
        self.stop_btn.state(["!disabled"])
        self.pframe.grid()

    def _enter_play(self):
        self.play_state = "play"
        self.pause_btn.config(text="⏸ Пауза")
        self.pause_btn.state(["!disabled"])

    def _player_hide(self):
        self.play_state = "idle"
        self.pframe.grid_remove()

    def toggle_pause(self):
        if self.play_state != "play":
            return
        self.player.paused = not self.player.paused
        self.pause_btn.config(text="▶ Продолжить" if self.player.paused else "⏸ Пауза")

    def stop_speech(self):
        self.cancel.set()
        self.player.stop_flag = True

    def seek(self, event):
        if self.play_state != "play":
            return
        width = max(self.play_bar.winfo_width(), 1)
        frac = min(max(event.x / width, 0.0), 1.0)
        self.player.pos = int(frac * (len(self.player.data) - 1))

    def tick(self):
        """Крутилка и плавное движение полосы загрузки."""
        self.spin_lbl.config(text=SPINNER[self.spin_i % len(SPINNER)] if self.busy else "")
        self.spin_i += 1
        if self.play_state == "synth":
            self.play_bar["value"] = self.synth_frac * 100
            self.time_var.set(f"Подготовка звука… {int(self.synth_frac * 100)}%")
        elif self.play_state == "play":
            total, pos = len(self.player.data), self.player.pos
            self.play_bar["value"] = pos / max(total, 1) * 100
            sr = self.player.sr
            self.time_var.set(f"{fmt_time(pos / sr)} / {fmt_time(total / sr)}")
        if self.loading:
            cap = self.p_next - 1 if self.p_next < 100 else 100
            if self.p_shown < cap:
                self.p_shown += (cap - self.p_shown) * 0.02
            self.progress["value"] = self.p_shown
            self.pct_var.set(f"{int(self.p_shown)}%")
        self.root.after(90, self.tick)

    def _load(self):
        try:
            self.engine.load(self.stage)
            self.ui(self._loaded)
        except Exception as e:
            def fail():
                self.loading = False
                self.busy = False
                self.pct_var.set("ошибка")
                self.status.set(f"Ошибка загрузки моделей: {e}")
            self.ui(fail)

    def _loaded(self):
        self.loading = False
        self.progress["value"] = 100
        self.progress.grid_remove()
        self.pct_var.set("")
        for t in self.texts.values():
            t.config(state="normal")
        self.status.set("Готово. Нажмите кнопку записи.")
        self.set_buttons(True)

    # ---------- буфер обмена (работает в любой раскладке) ----------
    def enable_clipboard(self, w):
        # Привязка по коду клавиши, а не по букве: Ctrl+C/V/X/A/Z работают и в русской раскладке
        def on_key(e):
            actions = {67: "<<Copy>>", 86: "<<Paste>>", 88: "<<Cut>>", 90: "<<Undo>>"}
            if e.keycode == 65:  # A
                w.tag_add("sel", "1.0", "end-1c")
                return "break"
            if e.keycode in actions:
                w.event_generate(actions[e.keycode])
                return "break"

        w.bind("<Control-KeyPress>", on_key)

        menu = tk.Menu(w, tearoff=0)
        menu.add_command(label="Вырезать", command=lambda: w.event_generate("<<Cut>>"))
        menu.add_command(label="Копировать", command=lambda: w.event_generate("<<Copy>>"))
        menu.add_command(label="Вставить", command=lambda: w.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Выделить всё", command=lambda: w.tag_add("sel", "1.0", "end-1c"))
        w.bind("<Button-3>", lambda e: (w.focus_set(), menu.tk_popup(e.x_root, e.y_root)))

    # ---------- перевод готового текста ----------
    def translate_text(self, src):
        text = self.texts[src].get("1.0", "end").strip()
        if not text:
            self.status.set("Поле пустое: вставьте или введите текст.")
            return
        self.set_buttons(False)
        threading.Thread(target=self.translate_pipeline, args=(src, text), daemon=True).start()

    def translate_pipeline(self, src, text):
        dst = "en" if src == "ru" else "ru"
        try:
            self.set_status("Перевожу…")
            translated = self.engine.translate(text, src, dst)
            self.set_text(dst, translated)
            self.set_status("Озвучиваю перевод…")
            self.speak(translated, dst)
            self.set_status("Готово.")
        except Exception as e:
            self.set_status(f"Ошибка: {e}")
        finally:
            self.set_buttons(True)

    # ---------- озвучка текста из поля ----------
    def speak_text(self, code):
        text = self.texts[code].get("1.0", "end").strip()
        if not text:
            self.status.set("Поле пустое: нечего озвучивать.")
            return
        self.set_buttons(False)
        threading.Thread(target=self.speak_pipeline, args=(code, text), daemon=True).start()

    def speak_pipeline(self, code, text):
        try:
            self.set_status("Озвучиваю…")
            self.speak(text, code)
            self.set_status("Готово.")
        except Exception as e:
            self.set_status(f"Ошибка: {e}")
        finally:
            self.set_buttons(True)

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
        for tb in list(self.tbuttons.values()) + list(self.sbuttons.values()):
            tb.state(["disabled"])
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
            self.speak(translated, dst)
            self.set_status("Готово.")
        except Exception as e:
            self.set_status(f"Ошибка: {e}")
        finally:
            self.set_buttons(True)


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
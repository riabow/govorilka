#!/usr/bin/env python3
"""
Говорилка - оконная версия локальной озвучки русского текста.

Запуск:  python tts_gui.py   (или двойной клик по start_govorilka.bat)
Нужен файл tts_realtime.py в той же папке.

Возможности:
  - окно для ввода текста (озвучивается выделенный фрагмент, если он есть)
  - выбор модели: XTTS-v2 (GPU) или Silero (CPU)
  - выбор готового образца голоса или запись с микрофона (10 с)
  - скорость речи, размер буфера, режим экономии видеопамяти (fp16)
  - сохранение озвучки в WAV-файл
  - настройки запоминаются в settings.json
Горячие клавиши: Ctrl+Enter - говорить, Esc - стоп.
"""
from __future__ import annotations

import datetime
import gc
import json
import os
import queue
import re
import sys
import threading
import time
import traceback
import wave
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

APP_DIR = Path(__file__).resolve().parent
SETTINGS_FILE = APP_DIR / "settings.json"
VOICES_DIR = APP_DIR / "voices"
LOG_FILE = APP_DIR / "govorilka.log"
DICT_FILE = APP_DIR / "ударения.txt"
DEFAULT_DICT = """# Словарь произношения. Одна строка - одно слово: слово = как его озвучить
# Совпадение по целому слову, регистр не важен. Формы слова (падежи) добавляйте отдельными строками.
#
# Silero: ставьте знак + перед ударной гласной.      замок = зам+ок
# XTTS:   знак + не поддерживается и удаляется, поэтому пишите слово так, как оно
#         должно звучать (подбор на слух), например дозвуком или другой записью.
#
# Примеры (уберите # в начале строки, чтобы включить):
# звонит = звон+ит
# дозвонишься = дозвон+ишься
"""

# При запуске через pythonw консоли нет - не даём библиотекам упасть на print/tqdm
if sys.stdout is None:
    sys.stdout = open(LOG_FILE, "a", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = sys.stdout

import numpy as np  # noqa: E402
import sounddevice as sd  # noqa: E402

from tts_realtime import Player, SileroEngine, XTTSEngine, split_text  # noqa: E402

ENGINE_LABELS = {
    "xtts": "XTTS-v2 - самый естественный (видеокарта)",
    "silero": "Silero - быстрый и лёгкий (процессор)",
}
SILERO_SPEAKERS = ["xenia", "kseniya", "baya", "aidar", "eugene"]
DEFAULT_TEXT = "Здравствуйте! Это проверка локальной озвучки текста."
NO_VOICE_LABEL = "встроенный голос (будет слышен акцент - лучше выбрать свой образец)"


def log_exc(where: str) -> str:
    tb = traceback.format_exc()
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {where}\n{tb}\n")
    except OSError:
        pass
    print(tb)
    return tb


def load_dictionary() -> list[tuple[str, str]]:
    if not DICT_FILE.exists():
        return []
    items = []
    for line in DICT_FILE.read_text(encoding="utf-8-sig", errors="replace").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = (x.strip() for x in line.split("=", 1))
        if key and val:
            items.append((key, val))
    items.sort(key=lambda kv: -len(kv[0]))  # длинные ключи первыми
    return items


def apply_dictionary(text: str, kind: str) -> str:
    """Подставляет произношение из ударения.txt (один проход, по целым словам)
    и убирает знак + для XTTS, который его не понимает."""
    items = load_dictionary()
    if items:
        table = {k.lower(): v for k, v in items}
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(k) for k, _ in items) + r")(?!\w)",
                             re.IGNORECASE)

        def fix(m: re.Match) -> str:
            val = table[m.group(0).lower()]
            if kind == "xtts":
                val = val.replace("+", "")
            if m.group(0)[:1].isupper():  # сохраняем заглавную букву
                for i, ch in enumerate(val):
                    if ch.isalpha():
                        return val[:i] + ch.upper() + val[i + 1:]
            return val

        text = pattern.sub(fix, text)
    if kind == "xtts":
        text = re.sub(r"\+(?=[аеёиоуыэюя])", "", text, flags=re.IGNORECASE)
    return text


class SileroStressEngine(SileroEngine):
    """Silero с полным набором флагов ударений, включая омографы (зАмок / замОк)."""

    def synth(self, text: str):
        with self.torch.inference_mode():
            kwargs = dict(text=text, speaker=self.speaker, sample_rate=self.sample_rate,
                          put_accent=True, put_yo=True)
            try:
                audio = self.model.apply_tts(**kwargs, put_stress_homo=True, put_yo_homo=True)
            except TypeError:  # старая версия модели без этих флагов
                audio = self.model.apply_tts(**kwargs)
        yield audio.numpy().astype(np.float32)


def build_plan(text: str, marker: str, unit: float, max_pause: float = 30.0) -> list[tuple[str, object]]:
    """Разбирает текст на шаги: ("s", предложение) и ("p", секунды паузы).
    Подряд идущие знаки паузы складываются: N знаков = N * unit секунд."""
    chunks: list[tuple[str, object]] = []
    if marker:
        pattern = re.compile(r"\s*(?:" + re.escape(marker) + r"\s*)+")
        pos = 0
        for m in pattern.finditer(text):
            chunks.append(("t", text[pos:m.start()]))
            chunks.append(("p", min(m.group().count(marker) * unit, max_pause)))
            pos = m.end()
        chunks.append(("t", text[pos:]))
    else:
        chunks.append(("t", text))
    plan: list[tuple[str, object]] = []
    for kind, val in chunks:
        if kind == "t":
            plan += [("s", sent) for sent in split_text(str(val))]
        else:
            plan.append(("p", val))
    return plan


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("Говорилка - локальная озвучка текста")
        root.geometry("900x720")
        root.minsize(720, 560)

        self.ui_q: queue.Queue = queue.Queue()
        self.stop_event = threading.Event()
        self.engines: dict[str, object] = {}
        self.player: Player | None = None
        self.busy = False

        cfg = self._load_settings()
        voice = cfg.get("voice_wav")
        self.voice_wav: str | None = voice if voice and Path(voice).exists() else None

        self.engine_var = tk.StringVar(value=ENGINE_LABELS.get(cfg.get("engine", "xtts"), ENGINE_LABELS["xtts"]))
        self.silero_var = tk.StringVar(value=cfg.get("silero_speaker", "xenia"))
        self.speed_var = tk.DoubleVar(value=float(cfg.get("speed", 1.0)))
        self.prebuf_var = tk.DoubleVar(value=float(cfg.get("prebuffer", 0.8)))
        self.fp16_var = tk.BooleanVar(value=bool(cfg.get("fp16", False)))
        self.pmark_var = tk.StringVar(value=cfg.get("pause_mark", "*"))
        self.punit_var = tk.DoubleVar(value=float(cfg.get("pause_unit", 0.4)))
        self.status_var = tk.StringVar(value="Готово. Введите текст и нажмите «Говорить».")

        self._build_ui(cfg.get("text", DEFAULT_TEXT))
        self._update_voice_label()
        self._update_controls()

        root.bind("<Control-Return>", lambda e: (self.on_speak(), "break")[1])
        root.bind("<Escape>", lambda e: self.on_stop())
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll()

    # ------------------------------------------------------------------ UI
    def _build_ui(self, initial_text: str):
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)

        box = ttk.LabelFrame(main, text="Настройки", padding=8)
        box.pack(fill="x")
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Модель:").grid(row=0, column=0, sticky="w", pady=2)
        self.engine_cb = ttk.Combobox(box, state="readonly", textvariable=self.engine_var,
                                      values=list(ENGINE_LABELS.values()))
        self.engine_cb.grid(row=0, column=1, sticky="ew", padx=6, pady=2)
        self.engine_cb.bind("<<ComboboxSelected>>", lambda e: self._update_controls())

        ttk.Label(box, text="Голос Silero:").grid(row=0, column=2, sticky="e", padx=(10, 0))
        self.silero_cb = ttk.Combobox(box, state="readonly", width=10,
                                      textvariable=self.silero_var, values=SILERO_SPEAKERS)
        self.silero_cb.grid(row=0, column=3, padx=6)

        ttk.Label(box, text="Образец голоса:").grid(row=1, column=0, sticky="w", pady=2)
        self.voice_lbl = ttk.Label(box, text="", foreground="#444")
        self.voice_lbl.grid(row=1, column=1, columnspan=3, sticky="w", padx=6)

        vbtns = ttk.Frame(box)
        vbtns.grid(row=2, column=1, columnspan=3, sticky="w", padx=6, pady=2)
        self.voice_pick_btn = ttk.Button(vbtns, text="Выбрать файл…", command=self.on_pick_voice)
        self.voice_rec_btn = ttk.Button(vbtns, text="● Записать с микрофона (10 с)", command=self.on_record_voice)
        self.voice_reset_btn = ttk.Button(vbtns, text="Сбросить", command=self.on_reset_voice)
        for b in (self.voice_pick_btn, self.voice_rec_btn, self.voice_reset_btn):
            b.pack(side="left", padx=(0, 6))

        ttk.Label(box, text="Скорость речи:").grid(row=3, column=0, sticky="w", pady=2)
        sp = ttk.Frame(box)
        sp.grid(row=3, column=1, columnspan=3, sticky="ew", padx=6)
        self.speed_lbl = ttk.Label(sp, width=6)
        self.speed_scale = ttk.Scale(sp, from_=0.7, to=1.4, variable=self.speed_var,
                                     command=lambda v: self.speed_lbl.config(text=f"{float(v):.2f}×"))
        self.speed_scale.pack(side="left", fill="x", expand=True)
        self.speed_lbl.pack(side="left", padx=6)
        self.speed_lbl.config(text=f"{self.speed_var.get():.2f}×")

        ttk.Label(box, text="Буфер перед стартом:").grid(row=4, column=0, sticky="w", pady=2)
        pb = ttk.Frame(box)
        pb.grid(row=4, column=1, columnspan=3, sticky="ew", padx=6)
        self.prebuf_lbl = ttk.Label(pb, width=6)
        self.prebuf_scale = ttk.Scale(pb, from_=0.3, to=4.0, variable=self.prebuf_var,
                                      command=lambda v: self.prebuf_lbl.config(text=f"{float(v):.1f} с"))
        self.prebuf_scale.pack(side="left", fill="x", expand=True)
        self.prebuf_lbl.pack(side="left", padx=6)
        self.prebuf_lbl.config(text=f"{self.prebuf_var.get():.1f} с")

        ttk.Label(box, text="Знак паузы:").grid(row=5, column=0, sticky="w", pady=2)
        pz = ttk.Frame(box)
        pz.grid(row=5, column=1, columnspan=3, sticky="ew", padx=6)
        self.pmark_entry = ttk.Entry(pz, width=5, textvariable=self.pmark_var, justify="center")
        self.pmark_entry.pack(side="left")
        ttk.Label(pz, text="  1 знак = ").pack(side="left")
        self.punit_lbl = ttk.Label(pz, width=6)
        self.punit_scale = ttk.Scale(pz, from_=0.1, to=1.5, variable=self.punit_var,
                                     command=lambda v: self.punit_lbl.config(text=f"{float(v):.1f} с"))
        self.punit_scale.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.punit_lbl.pack(side="left")
        self.punit_lbl.config(text=f"{self.punit_var.get():.1f} с")
        ttk.Label(box, foreground="#666",
                  text="Пример: «Привет *** мир» - пауза 3 знака подряд. Чем больше знаков, тем длиннее пауза.").grid(
            row=6, column=1, columnspan=3, sticky="w", padx=6)

        self.fp16_chk = ttk.Checkbutton(
            box, variable=self.fp16_var,
            text="Экономить видеопамять (fp16, экспериментально - включайте, если XTTS не помещается)")
        self.fp16_chk.grid(row=7, column=0, columnspan=4, sticky="w", pady=(4, 0))

        tf = ttk.LabelFrame(main, text="Текст", padding=6)
        tf.pack(fill="both", expand=True, pady=8)
        self.text = ScrolledText(tf, wrap="word", font=("Segoe UI", 12), undo=True, height=10)
        self.text.pack(fill="both", expand=True)
        self.text.insert("1.0", initial_text)
        self._setup_text_shortcuts()

        bar = ttk.Frame(main)
        bar.pack(fill="x")
        self.speak_btn = ttk.Button(bar, text="▶ Говорить (Ctrl+Enter)", command=self.on_speak)
        self.stop_btn = ttk.Button(bar, text="■ Стоп (Esc)", command=self.on_stop, state="disabled")
        self.save_btn = ttk.Button(bar, text="Сохранить в WAV…", command=lambda: self.on_speak(save=True))
        self.open_btn = ttk.Button(bar, text="Открыть текст из файла…", command=self.on_open_text)
        self.clear_btn = ttk.Button(bar, text="Очистить", command=lambda: self.text.delete("1.0", "end"))
        self.dict_btn = ttk.Button(bar, text="Словарь ударений…", command=self.on_open_dict)
        for b in (self.speak_btn, self.stop_btn, self.save_btn, self.open_btn, self.clear_btn, self.dict_btn):
            b.pack(side="left", padx=(0, 6))

        self.progress = ttk.Progressbar(main, mode="determinate")
        self.progress.pack(fill="x", pady=(8, 2))
        ttk.Label(main, textvariable=self.status_var, wraplength=860, justify="left").pack(fill="x")

        self._busy_controls = [self.engine_cb, self.silero_cb, self.voice_pick_btn, self.voice_rec_btn,
                               self.voice_reset_btn, self.speed_scale, self.fp16_chk, self.speak_btn,
                               self.pmark_entry, self.punit_scale,
                               self.save_btn, self.open_btn, self.clear_btn]

    def _setup_text_shortcuts(self):
        """Ctrl+C/V/X/A/Z/Y по коду клавиши - работает и в русской раскладке;
        плюс контекстное меню по правой кнопке мыши."""
        t = self.text

        def select_all():
            t.tag_add("sel", "1.0", "end-1c")
            t.mark_set("insert", "end-1c")

        actions = {
            86: lambda: t.event_generate("<<Paste>>"),   # V
            67: lambda: t.event_generate("<<Copy>>"),    # C
            88: lambda: t.event_generate("<<Cut>>"),     # X
            65: select_all,                              # A
            90: lambda: t.event_generate("<<Undo>>"),    # Z
            89: lambda: t.event_generate("<<Redo>>"),    # Y
        }

        def on_ctrl_key(e):
            action = actions.get(e.keycode)
            if action is None:
                return None
            action()
            return "break"

        t.bind("<Control-KeyPress>", on_ctrl_key)

        menu = tk.Menu(t, tearoff=0)
        menu.add_command(label="Вырезать", command=actions[88])
        menu.add_command(label="Копировать", command=actions[67])
        menu.add_command(label="Вставить", command=actions[86])
        menu.add_separator()
        menu.add_command(label="Выделить всё", command=select_all)

        def show_menu(e):
            t.focus_set()
            menu.tk_popup(e.x_root, e.y_root)
            return "break"

        t.bind("<Button-3>", show_menu)

    def _kind(self) -> str:
        for k, label in ENGINE_LABELS.items():
            if label == self.engine_var.get():
                return k
        return "xtts"

    def _update_controls(self):
        is_x = self._kind() == "xtts"
        for w in (self.speed_scale, self.fp16_chk, self.voice_pick_btn,
                  self.voice_rec_btn, self.voice_reset_btn):
            w.state(["!disabled"] if is_x else ["disabled"])
        self.silero_cb.state(["disabled"] if is_x else ["!disabled"])

    def _update_voice_label(self):
        self.voice_lbl.config(text=Path(self.voice_wav).name if self.voice_wav else NO_VOICE_LABEL)

    def _set_busy(self, busy: bool, keep_stop: bool = False):
        self.busy = busy
        if busy:
            for w in self._busy_controls:
                w.state(["disabled"])
            self.stop_btn.state(["!disabled"] if not keep_stop else ["disabled"])
            self.progress.configure(mode="indeterminate")
            self.progress.start(12)
        else:
            for w in self._busy_controls:
                w.state(["!disabled"])
            self.stop_btn.state(["disabled"])
            self.progress.stop()
            self.progress.configure(mode="determinate", value=0)
            self._update_controls()

    def _progress(self, value: int, maximum: int):
        self.progress.stop()
        self.progress.configure(mode="determinate", maximum=max(maximum, 1), value=value)

    # ----------------------------------------------- thread -> UI bridge
    def _post(self, fn):
        self.ui_q.put(fn)

    def _status(self, text: str):
        self._post(lambda: self.status_var.set(text))

    def _poll(self):
        try:
            while True:
                self.ui_q.get_nowait()()
        except queue.Empty:
            pass
        self.root.after(40, self._poll)

    # ------------------------------------------------------------ settings
    def _load_settings(self) -> dict:
        try:
            return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_settings(self):
        cfg = {
            "engine": self._kind(), "silero_speaker": self.silero_var.get(),
            "speed": round(self.speed_var.get(), 2), "prebuffer": round(self.prebuf_var.get(), 1),
            "fp16": self.fp16_var.get(), "voice_wav": self.voice_wav,
            "pause_mark": self.pmark_var.get().strip(), "pause_unit": round(self.punit_var.get(), 1),
            "text": self.text.get("1.0", "end").strip(),
        }
        try:
            SETTINGS_FILE.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass

    def on_close(self):
        self.stop_event.set()
        self._save_settings()
        self.root.destroy()

    # ------------------------------------------------------- voice sample
    def _set_voice(self, path: str | None):
        self.voice_wav = path
        self._update_voice_label()

    def on_pick_voice(self):
        path = filedialog.askopenfilename(
            title="Образец голоса (чистая русская речь, 6-10 секунд)",
            initialdir=str(VOICES_DIR if VOICES_DIR.exists() else APP_DIR),
            filetypes=[("Аудио", "*.wav *.mp3 *.flac *.ogg"), ("Все файлы", "*.*")])
        if not path:
            return
        if path.lower().endswith(".wav"):
            try:
                with wave.open(path, "rb") as w:
                    secs = w.getnframes() / w.getframerate()
                if secs < 3:
                    messagebox.showwarning("Короткий образец",
                                           f"В файле всего {secs:.1f} с. Для хорошего качества нужно 6-10 секунд.")
                elif secs > 30:
                    messagebox.showinfo("Длинный образец",
                                        f"В файле {secs:.0f} с. Модель использует только первые секунды, "
                                        "лучше взять фрагмент 6-10 секунд.")
            except (wave.Error, EOFError, OSError):
                pass
        self._set_voice(path)
        self.status_var.set("Образец голоса выбран. Он применится при следующем запуске озвучки.")

    def on_reset_voice(self):
        self._set_voice(None)

    def on_record_voice(self):
        if self.busy:
            return
        if not messagebox.askokcancel(
                "Запись образца",
                "После нажатия OK начнётся запись с микрофона на 10 секунд.\n\n"
                "Говорите по-русски, ровным голосом, в тихом помещении, без пауз и музыки."):
            return
        self._set_busy(True, keep_stop=True)
        threading.Thread(target=self._record_worker, daemon=True).start()

    def _record_worker(self):
        secs = 10
        try:
            rate = None
            for r in (24000, 44100, 48000):
                try:
                    sd.check_input_settings(samplerate=r, channels=1)
                    rate = r
                    break
                except Exception:  # noqa: BLE001
                    continue
            if rate is None:
                raise RuntimeError("Микрофон не найден или недоступен")
            data = sd.rec(int(secs * rate), samplerate=rate, channels=1, dtype="int16")
            for left in range(secs, 0, -1):
                self._status(f"● Идёт запись… осталось {left} с. Говорите по-русски, ровным голосом.")
                time.sleep(1)
            sd.wait()
            VOICES_DIR.mkdir(exist_ok=True)
            path = VOICES_DIR / f"record_{datetime.datetime.now():%Y%m%d_%H%M%S}.wav"
            with wave.open(str(path), "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(rate)
                w.writeframes(data.tobytes())
            self._post(lambda: self._set_voice(str(path)))
            self._status(f"Запись сохранена: {path.name}. Нажмите «Говорить», чтобы услышать результат.")
        except Exception as e:  # noqa: BLE001
            log_exc("record")
            msg = str(e)
            self._post(lambda: messagebox.showerror("Ошибка записи", msg))
            self._status("Запись не удалась.")
        finally:
            self._post(lambda: self._set_busy(False))

    # ---------------------------------------------------------- text file
    def on_open_text(self):
        path = filedialog.askopenfilename(title="Текстовый файл",
                                          filetypes=[("Текст", "*.txt *.md"), ("Все файлы", "*.*")])
        if not path:
            return
        raw = Path(path).read_bytes()
        for enc in ("utf-8-sig", "cp1251"):
            try:
                content = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            messagebox.showerror("Ошибка", "Не удалось определить кодировку файла.")
            return
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)

    def on_open_dict(self):
        try:
            if not DICT_FILE.exists():
                DICT_FILE.write_text(DEFAULT_DICT, encoding="utf-8")
            os.startfile(str(DICT_FILE))  # откроется в Блокноте (только Windows)
            self.status_var.set("Словарь открыт. Сохраните файл - изменения применятся при следующей озвучке.")
        except Exception as e:  # noqa: BLE001
            messagebox.showinfo("Словарь ударений", f"Откройте файл вручную:\n{DICT_FILE}\n\n({e})")

    # -------------------------------------------------------------- speak
    def _opts(self) -> dict:
        return {
            "kind": self._kind(), "silero": self.silero_var.get(),
            "speed": round(float(self.speed_var.get()), 2),
            "prebuf": round(float(self.prebuf_var.get()), 1),
            "fp16": bool(self.fp16_var.get()), "voice": self.voice_wav,
            "pmark": self.pmark_var.get().strip(), "punit": round(float(self.punit_var.get()), 2),
        }

    def on_speak(self, save: bool = False):
        if self.busy:
            return
        try:
            text = self.text.get("sel.first", "sel.last")
        except tk.TclError:
            text = self.text.get("1.0", "end")
        if not text.strip():
            messagebox.showinfo("Нет текста", "Введите текст для озвучки.")
            return
        save_path = None
        if save:
            save_path = filedialog.asksaveasfilename(
                defaultextension=".wav", filetypes=[("WAV", "*.wav")], initialfile="govorilka.wav")
            if not save_path:
                return
        self.stop_event.clear()
        self._set_busy(True)
        self._save_settings()
        threading.Thread(target=self._worker, args=(text, self._opts(), save_path), daemon=True).start()

    def on_stop(self):
        if not self.busy:
            return
        self.stop_event.set()
        p = self.player
        if p is not None:  # мгновенно заглушить уже накопленный звук
            with p.lock:
                p.chunks.clear()
                p.buffered = 0

    # ----- engines (вызывается из рабочего потока)
    def _drop_xtts(self):
        self.engines.pop("xtts", None)
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

    def _get_engine(self, o: dict):
        if o["kind"] == "xtts":
            eng = self.engines.get("xtts")
            if eng is not None and (eng.fp16 != o["fp16"] or (eng.fp16 and eng.voice_key != o["voice"])):
                self._drop_xtts()
                eng = None
            if eng is None:
                self._status("Загружаю XTTS-v2… При самом первом запуске скачивается ~1.8 ГБ - "
                             "это может занять несколько минут (прогресс виден в чёрном окне консоли).")
                try:
                    eng = XTTSEngine(o["voice"], fp16=o["fp16"])
                    self.engines["xtts"] = eng
                    self._status(f"XTTS-v2 загружен, занято видеопамяти: {eng.vram_mb:.0f} МБ.")
                except Exception as e:  # noqa: BLE001
                    log_exc("load xtts")
                    msg = f"{type(e).__name__}: {e}"
                    self._drop_xtts()
                    self._post(lambda: self._xtts_failed(msg))
                    o["kind"] = "silero"
            elif eng.voice_key != o["voice"]:
                self._status("Применяю новый образец голоса…")
                eng.set_voice(o["voice"])
            if o["kind"] == "xtts":
                eng.speed = o["speed"]
                return eng
        eng = self.engines.get("silero")
        if eng is None:
            self._status("Загружаю Silero (при первом запуске скачается ~100 МБ)…")
            eng = SileroStressEngine(o["silero"])
            self.engines["silero"] = eng
        eng.speaker = o["silero"]
        return eng

    def _xtts_failed(self, msg: str):
        self.engine_var.set(ENGINE_LABELS["silero"])  # элементы обновятся по завершении
        messagebox.showwarning(
            "XTTS не запустился",
            f"{msg}\n\nВероятно, не хватило видеопамяти или не установлена CUDA-версия PyTorch.\n"
            "Попробуйте закрыть другие программы, использующие видеокарту, или включить режим fp16.\n\n"
            "Пока переключаюсь на Silero. Подробности в govorilka.log.")

    # ----- рабочий поток
    def _worker(self, text: str, o: dict, save_path: str | None):
        player = None
        try:
            engine = self._get_engine(o)
            text = apply_dictionary(text, o["kind"])  # kind мог смениться на silero при сбое XTTS
            plan = build_plan(text, o["pmark"], o["punit"])
            n = sum(1 for kind, _ in plan if kind == "s")
            if n == 0:
                self._status("Нечего озвучивать.")
                return
            if not save_path:
                player = Player(engine.sample_rate, o["prebuf"])
                self.player = player

            parts: list[np.ndarray] = []
            t0 = time.perf_counter()
            first = None
            audio_sec = 0.0
            self._post(lambda: self._progress(0, n))
            i = 0
            for kind, value in plan:
                if self.stop_event.is_set():
                    break
                if kind == "p":  # пауза: тишина встаёт в очередь строго между фразами
                    silence = np.zeros(int(float(value) * engine.sample_rate), dtype=np.float32)
                    if player:
                        player.feed(silence)
                    else:
                        parts.append(silence)
                    continue
                i += 1
                self._status(f"Генерирую: предложение {i} из {n}")
                for chunk in engine.synth(str(value)):
                    if self.stop_event.is_set():
                        break
                    if first is None:
                        first = time.perf_counter() - t0
                    audio_sec += len(chunk) / engine.sample_rate
                    if player:
                        player.feed(chunk)
                    else:
                        parts.append(chunk)
                self._post(lambda i=i: self._progress(i, n))
            gen_time = time.perf_counter() - t0

            if self.stop_event.is_set():
                self._status("Остановлено.")
                return

            if player:
                with player.lock:
                    player.playing = True
                self._status("Дочитываю…")
                while not self.stop_event.is_set():
                    with player.lock:
                        if player.buffered <= 0:
                            break
                    time.sleep(0.03)
                if not self.stop_event.is_set():
                    time.sleep(float(player.stream.latency) + 0.15)
            else:
                audio = np.clip(np.concatenate(parts), -1.0, 1.0)
                pcm = (audio * 32767).astype(np.int16)
                with wave.open(save_path, "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(engine.sample_rate)
                    w.writeframes(pcm.tobytes())

            info = ""
            if audio_sec > 0 and first is not None:
                rtf = gen_time / audio_sec
                info = f" Первый звук через {first:.1f} с, скорость генерации RTF {rtf:.2f}"
                info += (" - медленнее речи: увеличьте буфер или выберите Silero." if rtf > 1
                         else " (меньше 1 - быстрее речи, пауз быть не должно).")
            self._status(("Сохранено: " + save_path + "." if save_path else "Готово.") + info)
        except Exception as e:  # noqa: BLE001
            log_exc("worker")
            msg = f"{type(e).__name__}: {e}"
            hint = ""
            if "out of memory" in msg.lower():
                hint = ("\n\nНе хватило видеопамяти. Закройте другие программы, включите fp16 "
                        "или выберите Silero.")
                self._drop_xtts()
            self._post(lambda: messagebox.showerror("Ошибка озвучки", msg + hint + "\n\nПодробности в govorilka.log"))
            self._status("Ошибка. Подробности в govorilka.log.")
        finally:
            if player is not None:
                try:
                    player.close()
                except Exception:  # noqa: BLE001
                    pass
            self.player = None
            self._post(lambda: self._set_busy(False))


def main():
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(1)  # чёткий текст на HiDPI
    except Exception:  # noqa: BLE001
        pass
    root = tk.Tk()
    style = ttk.Style()
    if "vista" in style.theme_names():
        style.theme_use("vista")
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
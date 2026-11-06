"""
Офлайн-переводчик EN <-> RU на локальных моделях (Helsinki-NLP opus-mt).
Работает на CPU, при наличии NVIDIA GPU автоматически использует CUDA.

Установка:
    pip install torch transformers sentencepiece sacremoses
(для видеокарты NVIDIA лучше ставить torch так:
    pip install torch --index-url https://download.pytorch.org/whl/cu121 )
"""

import re
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext

import torch
from transformers import MarianMTModel, MarianTokenizer

MODELS = {
    "en-ru": "Helsinki-NLP/opus-mt-en-ru",
    "ru-en": "Helsinki-NLP/opus-mt-ru-en",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_CHUNK_CHARS = 400  # длинный текст режем на куски, чтобы модель не теряла части


class Translator:
    def __init__(self):
        self._cache = {}  # direction -> (tokenizer, model)

    def _load(self, direction):
        if direction not in self._cache:
            name = MODELS[direction]
            tok = MarianTokenizer.from_pretrained(name)
            model = MarianMTModel.from_pretrained(name)
            if DEVICE == "cuda":
                model = model.half()
            model = model.to(DEVICE).eval()
            self._cache[direction] = (tok, model)
        return self._cache[direction]

    @staticmethod
    def _chunks(paragraph):
        """Режем абзац на предложения и склеиваем в куски до MAX_CHUNK_CHARS."""
        sentences = re.split(r"(?<=[.!?…])\s+", paragraph.strip())
        chunks, cur = [], ""
        for s in sentences:
            if cur and len(cur) + len(s) + 1 > MAX_CHUNK_CHARS:
                chunks.append(cur)
                cur = s
            else:
                cur = f"{cur} {s}".strip()
        if cur:
            chunks.append(cur)
        return chunks

    def translate(self, text, direction):
        tok, model = self._load(direction)
        result_paragraphs = []
        for para in text.split("\n"):
            if not para.strip():
                result_paragraphs.append("")
                continue
            chunks = self._chunks(para)
            out = []
            for i in range(0, len(chunks), 8):
                batch = chunks[i:i + 8]
                enc = tok(batch, return_tensors="pt", padding=True,
                          truncation=True, max_length=512).to(DEVICE)
                with torch.no_grad():
                    gen = model.generate(**enc, num_beams=4, max_length=512)
                out.extend(tok.batch_decode(gen, skip_special_tokens=True))
            result_paragraphs.append(" ".join(out))
        return "\n".join(result_paragraphs)


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Переводчик EN ⇄ RU (локальная модель)")
        self.geometry("1000x560")
        self.minsize(700, 400)

        self.translator = Translator()
        self.busy = False

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(1, weight=1)

        self.lbl_left = ttk.Label(self, text="Исходный текст", font=("Segoe UI", 10, "bold"))
        self.lbl_left.grid(row=0, column=0, sticky="w", padx=10, pady=(10, 0))
        self.lbl_right = ttk.Label(self, text="Перевод", font=("Segoe UI", 10, "bold"))
        self.lbl_right.grid(row=0, column=1, sticky="w", padx=10, pady=(10, 0))

        self.src = scrolledtext.ScrolledText(self, wrap="word", font=("Segoe UI", 11), undo=True)
        self.src.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        self.dst = scrolledtext.ScrolledText(self, wrap="word", font=("Segoe UI", 11))
        self.dst.grid(row=1, column=1, sticky="nsew", padx=10, pady=5)

        bar = ttk.Frame(self)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

        self.btn_en_ru = ttk.Button(bar, text="Перевести EN → RU",
                                    command=lambda: self.start("en-ru"))
        self.btn_en_ru.pack(side="left", padx=(0, 8))
        self.btn_ru_en = ttk.Button(bar, text="Перевести RU → EN",
                                    command=lambda: self.start("ru-en"))
        self.btn_ru_en.pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Очистить", command=self.clear).pack(side="left", padx=(0, 8))
        ttk.Button(bar, text="Копировать перевод", command=self.copy).pack(side="left")

        self.status = tk.StringVar(value=f"Готово. Устройство: {DEVICE.upper()}")
        ttk.Label(bar, textvariable=self.status).pack(side="right")

    def start(self, direction):
        if self.busy:
            return
        text = self.src.get("1.0", "end").strip()
        if not text:
            return
        self.busy = True
        self._set_buttons("disabled")
        first = direction not in self.translator._cache
        self.status.set("Загрузка модели (первый раз скачивается ~300 МБ)..." if first
                        else "Перевожу...")
        threading.Thread(target=self._work, args=(text, direction), daemon=True).start()

    def _work(self, text, direction):
        try:
            result = self.translator.translate(text, direction)
            self.after(0, self._done, result, None)
        except Exception as e:  # noqa
            self.after(0, self._done, None, e)

    def _done(self, result, error):
        self.busy = False
        self._set_buttons("normal")
        if error:
            self.status.set(f"Ошибка: {error}")
            return
        self.dst.delete("1.0", "end")
        self.dst.insert("1.0", result)
        self.status.set(f"Готово. Устройство: {DEVICE.upper()}")

    def _set_buttons(self, state):
        self.btn_en_ru.config(state=state)
        self.btn_ru_en.config(state=state)

    def clear(self):
        self.src.delete("1.0", "end")
        self.dst.delete("1.0", "end")

    def copy(self):
        self.clipboard_clear()
        self.clipboard_append(self.dst.get("1.0", "end").strip())


if __name__ == "__main__":
    App().mainloop()
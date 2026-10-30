Установка (Windows, Python 3.10-3.12)
py -3.11 -m venv venv
venv\Scripts\activate
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128

pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128 --default-timeout=120 --retries 15 --resume-retries 15

pip install coqui-tts sounddevice numpy num2words
Запуск
python tts_realtime.py                          # интерактивный режим: вводите текст, он озвучивается
python tts_realtime.py --file book.txt
python tts_realtime.py --speaker-wav voice.wav  # свой голос
python tts_realtime.py --engine silero          # принудительно запасной движок
Как это устроено
Старт звука: текст режется на предложения, XTTS отдаёт звук кусочками, и воспроизведение начинается после накопления около 0,8 с. Ждать конца текста не нужно.
Автопереключение: если XTTS не поместится в 4 ГБ или CUDA не заработает, программа сама перейдёт на Silero (CPU). Он слабее по естественности, зато работает всегда.
Диагностика: после каждой фразы выводится RTF. Если он меньше 1, генерация быстрее речи и пауз не будет. Если больше 1, будут паузы: увеличьте --prebuffer 2 или переключитесь на Silero.
На что обратить внимание
Голос. Встроенные голоса XTTS не русские, поэтому акцент будет слышен. Для лучшего результата запишите 6-10 секунд своей или чьей-то русской речи в wav (без шума) и передайте через --speaker-wav.
Память. Перед запуском закройте тяжёлые программы. На Windows при нехватке VRAM драйвер может начать использовать оперативную память, и тогда вместо ошибки вы получите сильное замедление. Следите за nvidia-smi. Если памяти не хватает, попробуйте --fp16, но это экспериментальный режим и возможны артефакты.
Ударения. XTTS их не расставляет и иногда ошибается в редких словах. Silero ставит ударения сам, поэтому для специфичных текстов он может звучать точнее.
Числа. Они превращаются в слова только в именительном падеже: «5 яблок» прозвучит как «пять яблок», но «о 5 яблоках» будет неточным.
Лицензия. Модель XTTS-v2 разрешена только для некоммерческого использования.

Если при установке появится ошибка про torchcodec или ffmpeg, пришлите текст ошибки, и я помогу её устранить.


После установки проверьте, что видна видеокарта:

python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

Должно вывести 2.8.0+cu128 True. Если выйдет False, напишите, разберёмся. После этого можно ставить остальное: pip install coqui-tts sounddevice numpy num2words.

сделать exe файл
pip install pyinstaller
pyinstaller --noconfirm --onedir --windowed --name Govorilka --collect-all TTS --collect-all trainer --collect-all gruut --collect-all num2words --collect-all sounddevice  tts_gui.py
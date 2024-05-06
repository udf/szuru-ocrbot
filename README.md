# Szuru OCRbot
A simple bot to run OCR (Optical Character Recognition) on images in a [szuru booru](https://github.com/rr-/szurubooru/).  
It runs on posts that have a special tag, and adds notes for any text that was found.

# Usage
Copy `config-example.py` to `config.py` and edit it to configure the bot:
- Add an access token, you can generate one from the user page (creating a user specifically for this purpose is recommended)
- Configure the API endpoint, by setting your booru's URL(s), and API and data endpoints can be configured separately (in the case of serving files from static.example.com, for example)
- `pip install -r requirements.txt` (recommended to use a venv)
- Tag a post that contains text on your booru with `pls_ocr_thx` (or whatever you set `TAG_PENDING` in the config to)
- Run `ocrbot.py`, you'd want to configure a systemd timer or cron job to run it periodically

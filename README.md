# Verifiable Indic Document OCR — Streamlit Portal

University of Engineering and Management, Kolkata · Dept. of CSE

Pages: Home · Document Inspector · Before/After Preprocessing · Live OCR Sandbox · Dataset Analytics (+ Download Hub) ·
Dataset Explorer · Data Quality · Dataset Splitting · Final Audit · About.

## 1. Run on your own computer (Windows)
1. Install Python 3.11 or 3.12 (tick "Add Python to PATH").
2. (Only for real Bengali/Hindi OCR on your PC) install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
   and tick the **Bengali** and **Hindi** language data in the installer, then add `C:\Program Files\Tesseract-OCR` to PATH.
3. Open a terminal in this folder and run:
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```
The site opens at http://localhost:8501. The Home page shows live which parts (OCR, translation, PDF) are working.

## 2. Push to GitHub
```
git init
git add .
git commit -m "Indic OCR portal"
git branch -M main
git remote add origin https://github.com/<your-username>/<repo-name>.git
git push -u origin main
```
(Create the empty repo on github.com first. Total size is about 65 MB, which GitHub accepts.)

## 3. Deploy on Streamlit Community Cloud
1. Go to https://share.streamlit.io → New app → pick your repo, branch `main`, main file `app.py`.
2. Deploy. `packages.txt` makes the server install Tesseract with English, Bengali and Hindi data automatically.

## 4. Plug in your teammates' models
* OCR: copy `modules/team_ocr_plugin_template.py` to `modules/team_ocr_plugin.py` and fill in `run_ocr()`.
* Translation: copy `modules/team_translation_plugin_template.py` to `modules/team_translation_plugin.py`.
Keep API keys in Streamlit "Secrets", never in code.

## 5. Data notes
* `data/` and `reports/` are copies of the project outputs; nothing was edited.
* Only the *mild* augmented images (`*_aug_mild.png`) are included (the *moderate* ones are 230 MB). Their catalog and
  annotations are included; Inspector lists only images that exist. To add them, copy the files into `data/augmented/images/`.
* `scripts/` holds the original pipeline scripts for reference; the website does not run them.

## 6. Reliable translation (free Gemini API key, no credit card)
The free Google web endpoint and MyMemory are often rate-limited or wrong for Bengali. For translation that works from
any phone or computer, add a free **Gemini API key**. The site then uses it first and falls back to the free services.

1. Open https://aistudio.google.com/apikey and sign in with any Google account.
2. Click **Create API key** (choose "Create API key in new project" if asked). Copy the key (it starts with `AIza`).
3. On your PC: in the `.streamlit` folder copy `secrets.toml.example` to `secrets.toml` and put the key in:
```
GEMINI_API_KEY = "paste-your-key-here"
MYMEMORY_EMAIL = "your.email@example.com"
```
4. Streamlit Cloud: your app -> **Settings -> Secrets**, paste the same lines, save.
5. Translate a document; the caption under the translation shows "Google Gemini API".

Notes: the free tier has daily/per-minute limits (plenty for a demo). Text you translate is sent to Google, so do not
upload private documents. Never commit the key to GitHub (`secrets.toml` is git-ignored). An Azure Translator key
(`AZURE_TRANSLATOR_KEY`, `AZURE_TRANSLATOR_REGION`) is also supported if you ever have one; it is tried first.

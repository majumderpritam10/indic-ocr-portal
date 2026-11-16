"""TEMPLATE — how your teammates plug their OCR model into the website.

1. Copy this file and rename the copy to:  modules/team_ocr_plugin.py
2. Replace the body of run_ocr() with a call to the real model (CRNN / TrOCR / Donut / ...).
3. Put the model weights in a folder (or load them from Hugging Face) and add any extra
   packages to requirements.txt. Restart the app — the sandbox will switch to your model automatically.

Contract (keep these names):
    MODEL_NAME : str                      shown on the website
    run_ocr(image, languages) -> dict     image = numpy array (gray or BGR), languages = e.g. ["hi", "en"]
        must return {"text": "..."}      and may also return "confidence" (0-100) and "word_count".

Never return placeholder text: if the model cannot read the page, return {"text": ""}.
"""
MODEL_NAME = "Team OCR model (replace me)"

_model = None


def _get_model():
    """Load the model once (the website calls run_ocr many times)."""
    global _model
    if _model is None:
        raise NotImplementedError("Load your real model here, e.g. _model = load_my_model('weights/model.pt')")
    return _model


def run_ocr(image, languages):
    model = _get_model()
    # text = model.predict(image, languages)
    raise NotImplementedError("Call your model here and return {'text': text, 'confidence': conf_or_None}")

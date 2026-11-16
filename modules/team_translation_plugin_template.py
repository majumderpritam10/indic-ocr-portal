"""TEMPLATE — how your teammates plug a translation model/API into the website.

Copy this file to  modules/team_translation_plugin.py  and fill in translate().
    MODEL_NAME : str
    translate(text, source, target) -> str
        source is 'bn' | 'hi' | 'en' | 'auto';  target is 'bn' | 'hi' | 'en'.

Keep API keys OUT of the code: on Streamlit Cloud put them in  App settings -> Secrets
and read them with  st.secrets["MY_KEY"]  (or os.environ).
"""
MODEL_NAME = "Team translation model (replace me)"


def translate(text, source, target):
    raise NotImplementedError("Call your real translation model/API here and return the translated string.")

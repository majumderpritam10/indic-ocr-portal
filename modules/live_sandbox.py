"""Screen 3 — Live upload, OCR and translation sandbox.

Everything is processed in memory (uploads are never written to disk) and every output shown on
the page comes from a step that actually ran. If a model/API is missing, the page says exactly what is missing.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from modules import language_detection as ld
from modules import ocr_engine, translation_engine, ui
from modules.image_preprocessing import quality_metrics, run_pipeline
from modules.paths import LANG_NAMES
from utils.file_validation import MAX_UPLOAD_MB, validate_upload
from utils.image_utils import before_after_slider_html, encode_png, human_size, load_upload, to_pil

LANG_CHOICES = ["en", "hi", "bn"]


def _reset():
    for k in ("sb_result", "sb_loaded", "sb_ocr_input", "sb_psm"):
        st.session_state.pop(k, None)
    st.session_state["sb_up_key"] = st.session_state.get("sb_up_key", 0) + 1  # new key => empty uploader


def _txt(text: str) -> bytes:
    return text.encode("utf-8-sig")


def _run_translation(res: dict, target: str):
    """Fills res['translation'] or res['translation_error']. Never raises."""
    res["translation"], res["translation_error"] = None, None
    try:
        src = res["source_code"]
        res["translation"] = translation_engine.translate(res["ocr"]["text"], src, target)
        res["translation"]["target"] = target
    except translation_engine.TranslationUnavailable as e:
        res["translation_error"] = str(e)
    except Exception as e:  # noqa: BLE001
        res["translation_error"] = f"Unexpected translation error: {type(e).__name__}: {e}"


def _variants(img, prep, mode):
    """Images handed to the OCR engine. 'Auto' tries several and keeps the best read."""
    pre = ("Preprocessed (binarised)", prep["final"])
    if mode == "Preprocessed image":
        return [pre]
    if mode == "Original image":
        return [("Original image", img)]
    out = [pre]
    clahe = next((st_["image"] for st_ in prep["stages"] if st_["name"] == "CLAHE" and st_["ran"]), None)
    if clahe is not None:
        out.append(("Contrast-enhanced grayscale (not binarised)", clahe))
    out.append(("Original image", img))
    return out


def _auto_detect(variants, psm):
    """Detect bn / hi / en. A first OCR pass reads with all three language packs and the script of the recognised
    letters is measured. If that pass was unreliable (low engine confidence, i.e. probably garbage), every language is
    tried on its own and the language whose OCR the engine is most sure about wins."""
    first = ocr_engine.run_ocr_best(variants, ["bn", "hi", "en"], psm=psm)
    a = ld.analyse_text(first["text"])
    conf = first.get("confidence")
    reliable = a["script"] is not None and a["letters"] >= 3 and (conf is None or conf >= 55)
    det = {"method": "auto", **a, "probes": None}
    if reliable:
        return det
    img = next((im for n, im in variants if n == first.get("variant")), variants[0][1])
    probes, best = [], None
    for code in ("bn", "hi", "en"):
        r = ocr_engine.run_ocr(img, [code], psm=psm)
        letters = sum(ch.isalpha() for ch in r["text"])
        probes.append({"code": code, "confidence": r.get("confidence"), "letters": letters})
        rk = ocr_engine.rank(r)
        if best is None or rk > best[0]:
            best = (rk, code, r)
    if best[2]["text"].strip():
        a2 = ld.analyse_text(best[2]["text"])
        det = {"method": "auto", **a2, "probes": probes}
        det["script"] = best[1]
        if best[1] not in det["present"]:
            det["present"] = [best[1]] + det["present"]
    else:
        det["probes"] = probes
    return det


def _process(loaded: dict, data_len: int, opts: dict, source_choice: str, ocr_on: str, psm: int, target: str,
             file_id: str, name: str):
    """The whole pipeline. Returns a result dict (never raises)."""
    res = {"file_id": file_id, "name": name, "loaded": loaded, "opts": opts, "ocr_input": ocr_on,
           "ocr": None, "ocr_error": None, "detect": None, "detect_error": None,
           "translation": None, "translation_error": None, "source_code": None}
    img = loaded["image"]
    with st.status("Processing your document…", expanded=True) as status:
        st.write("② Preprocessing image…")
        res["prep"] = run_pipeline(img, **opts)
        res["m_before"], res["m_after"] = quality_metrics(img), quality_metrics(res["prep"]["final"])
        res["m_before"]["size_bytes"] = data_len
        res["m_after"]["size_bytes"] = len(encode_png(res["prep"]["final"]))
        variants = _variants(img, res["prep"], ocr_on)

        st.write("③ Detecting script / language…")
        try:
            if source_choice != "auto":
                res["detect"] = {"method": "manual", "script": source_choice, "present": [source_choice],
                                 "shares": {}, "letters": 0}
            else:
                det = _auto_detect(variants, psm)
                if det["script"] is None:
                    res["detect_error"] = ("No readable letters were found, so the script could not be detected. "
                                           "Choose the document language manually.")
                else:
                    res["detect"] = det
        except ocr_engine.OcrUnavailable as e:
            res["detect_error"] = str(e)
        except Exception as e:  # noqa: BLE001
            res["detect_error"] = f"Language detection failed: {type(e).__name__}: {e}"

        if res["detect"]:
            res["source_code"] = res["detect"]["script"]
            st.write("④ Running OCR…")
            langs = ld.ocr_languages_for(res["detect"]["script"], res["detect"].get("present"))
            try:
                res["ocr"] = ocr_engine.run_ocr_best(variants, langs, psm=psm)
                res["ocr"]["analysis"] = ld.analyse_text(res["ocr"]["text"])
            except ocr_engine.OcrUnavailable as e:
                res["ocr_error"] = str(e)
            except Exception as e:  # noqa: BLE001
                res["ocr_error"] = f"OCR failed: {type(e).__name__}: {e}"
        if res["ocr"] and res["ocr"]["text"].strip():
            st.write("⑤ Translating…")
            _run_translation(res, target)
        status.update(label="Processing finished", state="complete", expanded=False)
    return res


def render():
    ui.page_header("📤", "Live OCR & Translation Sandbox",
                   "Upload a document, preprocess it, extract text with OCR and translate it to another language.")
    stepper_slot = st.empty()
    st.session_state.setdefault("sb_up_key", 0)

    # ───────────── upload + options ─────────────
    col_up, col_opt = st.columns([1.25, 1], gap="medium")
    with col_up.container(border=True):
        st.markdown("#### Upload Document")
        up = st.file_uploader("Drag & drop an image or click to upload", type=["png", "jpg", "jpeg", "tif", "tiff", "pdf"],
                              key=f"sb_up_{st.session_state['sb_up_key']}",
                              help=f"PNG, JPG, JPEG, TIFF or PDF, up to {MAX_UPLOAD_MB} MB. Files are processed in memory only.")
        loaded, data, file_id, pdf_page = None, b"", None, 0
        if up is not None:
            data = up.getvalue()
            ok, msg = validate_upload(up.name, data)
            if not ok:
                st.error(f"❌ {msg}")
            else:
                if Path(up.name).suffix.lower() == ".pdf":
                    pdf_page = int(st.number_input("PDF page to read", min_value=1, value=1, step=1)) - 1
                file_id = hashlib.sha256(data).hexdigest()[:16] + f"-{pdf_page}"
                cached = st.session_state.get("sb_loaded")
                if cached and cached[0] == file_id:
                    loaded = cached[1]
                else:
                    try:
                        loaded = load_upload(up.name, data, pdf_page)
                        st.session_state["sb_loaded"] = (file_id, loaded)
                    except ValueError as e:
                        st.error(f"❌ {e}")
                if loaded:
                    st.markdown(f'<div class="ok-box">✅ File uploaded: <b>{up.name}</b> ({human_size(len(data))})</div>',
                                unsafe_allow_html=True)
                    st.caption(f"Type: {loaded['kind']} · Dimensions: {loaded['width']} × {loaded['height']} px"
                               + (f" · Pages: {loaded['pages']}" if loaded["kind"] == "PDF" else ""))
                    if loaded["note"]:
                        st.info(loaded["note"])

    with col_opt.container(border=True):
        target = st.selectbox("Target language (for translation)", LANG_CHOICES, format_func=lambda c: LANG_NAMES[c], key="sb_target")
        source_choice = st.selectbox("Document language", ["auto", "bn", "hi", "en"], key="sb_source",
                                     format_func=lambda c: "Auto-detect" if c == "auto" else f"{LANG_NAMES[c]} (manual)")
        with st.expander("Advanced options"):
            opts = {
                "grayscale": st.checkbox("Grayscale", True), "clahe": st.checkbox("CLAHE contrast", True),
                "bilateral": st.checkbox("Bilateral filter", True), "otsu": st.checkbox("Otsu threshold", True),
                "deskew_on": st.checkbox("Deskew", True),
            }
            ocr_on = st.radio("Image given to the OCR engine",
                              ["Auto (try several, keep the best)", "Preprocessed image", "Original image"],
                              help="Auto tries the binarised image, a contrast-enhanced grayscale image and the original, "
                                   "and keeps the read with the best engine confidence. Otsu thresholding can ruin photos "
                                   "and handwriting, so Auto is safer.")
            psm = st.selectbox("Page layout mode (Tesseract)", [3, 4, 6, 7, 11], index=0,
                               help="3 = automatic, 4 = single column, 6 = single block, 7 = one line, 11 = sparse text. If a mode finds no text, the others are tried automatically.")
        b1, b2 = st.columns([2, 1])
        run = b1.button("🚀 Run OCR & Translate", type="primary", use_container_width=True, disabled=loaded is None)
        b2.button("🔄 Reset", use_container_width=True, on_click=_reset)

    if run and loaded is not None:
        st.session_state["sb_result"] = _process(loaded, len(data), opts, source_choice, ocr_on, psm, target, file_id, up.name)

    res = st.session_state.get("sb_result")
    if res and res["file_id"] != file_id:
        res = None  # result belongs to a previous file

    # ───────────── stepper ─────────────
    done = set()
    if loaded:
        done.add(1)
    if res:
        done.add(2)
        if res["detect"]:
            done.add(3)
        if res["ocr"] and res["ocr"]["text"].strip():
            done.add(4)
        if res["translation"]:
            done.add(5)
    with stepper_slot.container():
        ui.stepper(next((i for i in range(1, 6) if i not in done), 5), done)

    if loaded is None and res is None:
        st.info("Upload a PNG, JPG, TIFF or PDF document to begin. Nothing is shown until real processing has run.")
        return
    if res is None:
        with st.container(border=True):
            st.markdown("#### Original document preview")
            st.image(to_pil(loaded["image"]), use_container_width=True)
            st.caption("Click **Run OCR & Translate** to process it.")
        return

    # ───────────── results ─────────────
    left, right = st.columns([1.05, 1], gap="medium")
    with left.container(border=True):
        st.markdown("#### Image Preprocessing")
        prep = res["prep"]
        a, b = st.columns(2)
        a.markdown("**Original image**")
        a.image(to_pil(res["loaded"]["image"]), use_container_width=True)
        b.markdown("**Preprocessed image**")
        b.image(to_pil(prep["final"]), use_container_width=True)
        ran = ", ".join(prep["steps_run"]) or "none"
        st.success(f"Preprocessing finished. Steps that actually ran: {ran}.")
        for s in prep["stages"]:
            if s["name"] in ("Deskew",) and s["note"]:
                st.caption(f"Deskew: {s['note']}")
        with st.expander("See each stage, or compare with a slider"):
            tabs = st.tabs([s["name"] for s in prep["stages"]] + ["Slider"])
            for t, s in zip(tabs, prep["stages"]):
                with t:
                    st.image(to_pil(s["image"]), use_container_width=True)
                    st.caption(f"{s['name']}: {s['note']}")
            with tabs[-1]:
                components.html(before_after_slider_html(res["loaded"]["image"], prep["final"],
                                                         label_before="Original", label_after="Preprocessed"), height=720)
        st.markdown("**Image Quality Metrics**")
        mb, ma = res["m_before"], res["m_after"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Contrast", f"{ma['contrast']:.2f}", f"{ma['contrast'] - mb['contrast']:+.2f}")
        m2.metric("Sharpness", f"{ma['sharpness']:.1f}", f"{ma['sharpness'] - mb['sharpness']:+.1f}")
        m3.metric("Dimensions", f"{ma['width']}×{ma['height']}")
        m4.metric("File size", human_size(ma["size_bytes"]))
        st.caption(f"Contrast and sharpness are calculated from the actual pixels (change vs. the original shown). "
                   f"File size: {human_size(mb['size_bytes'])} uploaded → {human_size(ma['size_bytes'])} processed (PNG).")

    with right.container(border=True):
        st.markdown("#### Script / Language Detection")
        det = res["detect"]
        if det:
            name = ld.language_label(det["script"])
            if det["method"] == "manual":
                st.success(f"Language set manually: **{name}**")
            else:
                share = det["shares"].get(det["script"], 0) * 100
                st.success(f"Detected language: **{name}**  \nScript share: {share:.1f}% of recognised letters")
                st.caption("This is a character count (Unicode script analysis of a first OCR pass), not a model "
                           "probability, so no 'confidence %' is claimed.")
                if det.get("probes"):
                    pr = ", ".join(f"{ld.language_label(p['code'])}: {p['letters']} letters"
                                   + (f" at {p['confidence']}%" if p["confidence"] is not None else "")
                                   for p in det["probes"])
                    st.caption("The first read was unreliable (low engine confidence), so each language was tried on its "
                               f"own — {pr}. The language the engine read most reliably was kept.")
        else:
            st.error(res["detect_error"] or "Language detection did not run.")

        st.markdown("#### Extracted Text (OCR Output)")
        ocr = res["ocr"]
        if ocr_err := res["ocr_error"]:
            st.error(ocr_err)
        elif ocr is None:
            st.info("OCR did not run because the language could not be determined.")
        else:
            st.caption(f"Engine: {ocr['backend']} · languages: {ocr['languages']} · {ocr['seconds']} s"
                       + (f" · mean word confidence reported by the engine: {ocr['confidence']}%" if ocr["confidence"] is not None else ""))
            if len(ocr.get("attempts", [])) > 1 or (ocr.get("attempts") and ocr["variant"] != "Preprocessed (binarised)"):
                tried = "; ".join(f"{a['variant']}: {a['letters']} letters"
                                  + (f", confidence {a['confidence']}%" if a["confidence"] is not None else "")
                                  for a in ocr["attempts"])
                st.caption(f"Image variants tried → {tried}. Used: **{ocr['variant']}**.")
            if len(ocr.get("psm_tried", [])) > 1:
                st.caption(f"Page layout modes tried: {', '.join(map(str, ocr['psm_tried']))} — the last one was used.")
            if not ocr["text"].strip():
                st.warning("The OCR engine ran but found no text in this image.")
            else:
                st.code(ocr["text"], language=None, wrap_lines=True)  # built-in copy button (top-right)
                st.download_button("⬇ Download OCR Text (TXT)", _txt(ocr["text"]), f"{Path(res['name']).stem}_ocr.txt",
                                   "text/plain", key="dl_ocr")

        if ocr and ocr["text"].strip():
            tname = LANG_NAMES[target]
            st.markdown(f"#### Translated Text ({tname})")
            tr, err = res["translation"], res["translation_error"]
            if tr and tr.get("target") == target:
                if tr["note"]:
                    st.info(tr["note"])
                st.caption(f"Engine: {tr['backend']} · {tr['seconds']} s")
                st.code(tr["text"], language=None, wrap_lines=True)
                st.download_button("⬇ Download Translated Text (TXT)", _txt(tr["text"]),
                                   f"{Path(res['name']).stem}_{target}.txt", "text/plain", key="dl_tr")
            else:
                if tr:  # a translation exists, but for another language
                    st.info(f"The shown translation was made for {LANG_NAMES[tr['target']]}. "
                            f"Click Translate to get {tname}.")
                elif err:
                    st.error(err)
                if st.button(f"🌐 Translate to {tname}", key="sb_tr_btn"):
                    with st.spinner("Translating…"):
                        _run_translation(res, target)
                    st.session_state["sb_result"] = res
                    st.rerun()

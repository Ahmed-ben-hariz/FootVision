"""Streamlit dashboard. UI only: all logic lives in src/."""
from __future__ import annotations

import hashlib
import io
import time

import pandas as pd
import streamlit as st
from PIL import Image

from src.analysis import Analysis, analyze, render, run_detection
from src.config import ROOT, load_config
from src.device import get_device
from src.info import dataset_summary, read_json, read_text
from src.predict import load_predictor
from src.team_classifier import TEAM_A, TEAM_B, UNKNOWN, apply_overrides

PAGES = ["Player Detection", "Dataset Information", "Model Information"]


@st.cache_resource(show_spinner="Loading model...")
def get_model(checkpoint: str, device_pref: str):
    """Load the detector once per server process, not on every interaction."""
    device = get_device(device_pref)
    model, meta = load_predictor(checkpoint, device)
    return model, meta, str(device)


@st.cache_data(show_spinner=False)
def get_dataset_summary(dataset_dir: str, aliases: dict):
    return dataset_summary(dataset_dir, aliases)


def dataset_page(cfg: dict) -> None:
    st.title("Dataset Information")
    root = ROOT / cfg["data"]["dataset_dir"]
    summary = get_dataset_summary(str(root), cfg["data"]["class_aliases"])
    if summary is None:
        st.warning("The dataset has not been prepared yet. Download it as described "
                   "in docs/DATASET_GUIDE.md and place it under data/raw/.")
        return
    rows = [{"Split": s, "Images": v["images"], "Annotated players": v["player"],
             "Annotated balls": v["ball"]} for s, v in summary.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    st.caption("Source: Football-Player-Detection by Augmented Startups (Roboflow "
               "Universe), license CC BY 4.0. This is a preprocessed export (resized to "
               "1280x1280, augmented), so counts include augmented copies.")
    report = read_text(ROOT / "results" / "dataset_quality_report.md")
    with st.expander("Dataset quality report"):
        if report:
            st.markdown(report)
        else:
            st.info("Run `python -m src.data_quality --root <dataset folder>` to create it.")


def model_page(cfg: dict) -> None:
    st.title("Model Information")
    inf = cfg["inference"]
    ckpt = ROOT / inf["checkpoint"]
    if not ckpt.is_file():
        st.warning(f"No trained checkpoint found at {inf['checkpoint']}. Train a model first.")
        return
    _, meta, device_name = get_model(inf["checkpoint"], cfg["device"])
    props = {
        "Architecture": str(meta.get("architecture")),
        "Weights": "COCO-pretrained, fine-tuned on football data",
        "Classes": "3 (background, player, ball)",
        "Checkpoint": ckpt.name,
        "Epochs trained": str(meta.get("epoch")),
        "Trained at (UTC)": str(meta.get("trained_at")),
        "Input scale (min / max side)": f"{meta.get('min_size')} / {meta.get('max_size')}",
        "Anchor sizes": str(meta.get("anchor_sizes") or "default"),
        "Training images used": str(meta.get("n_train_images")),
        "Device": device_name,
    }
    st.dataframe(pd.DataFrame({"Property": list(props), "Value": list(props.values())}),
                 hide_index=True)
    metrics = read_json(ROOT / "results" / f"metrics_valid_{ckpt.stem}.json")
    st.subheader("Evaluation metrics")
    if metrics is None:
        st.info("No saved evaluation found. Run `python -m src.evaluate --split valid`.")
        return
    st.caption(f"Validation split, {metrics['n_images']} images, score threshold "
               f"{metrics['score_threshold']}. Not the test set.")
    cols = st.columns(3)
    cols[0].metric("mAP@0.5", f"{metrics['map_50']:.3f}")
    cols[1].metric("mAP@[.5:.95]", f"{metrics['map']:.3f}")
    cols[2].metric("mAR@100", f"{metrics['mar_100']:.3f}")
    rows = [{"Class": n, "Precision": round(c["precision"] or 0, 3),
             "Recall": round(c["recall"] or 0, 3),
             "Count MAE per image": round(c["count_mae"], 3)}
            for n, c in metrics["classes"].items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True)


def detection_page(cfg: dict) -> None:
    st.title("Player Detection")
    st.write("Upload a football match image. A PyTorch Faster R-CNN model fine-tuned on "
             "football images finds the visible players and the ball, and a color "
             "heuristic groups players into teams.")
    inf = cfg["inference"]
    if not (ROOT / inf["checkpoint"]).is_file():
        st.error(f"Checkpoint not found: {inf['checkpoint']}. Train a model first.")
        return
    model, _, _ = get_model(inf["checkpoint"], cfg["device"])
    device = get_device(cfg["device"])

    with st.sidebar:
        st.header("Settings")
        mode = st.selectbox("Inference mode", ["tiled", "single"],
                            index=0 if inf["mode"] == "tiled" else 1,
                            help="Tiled runs the model on 4 overlapping tiles: slower, "
                                 "better for small players and the ball.")
        threshold = st.slider("Confidence threshold", 0.05, 0.99,
                              float(inf["confidence_threshold"]), 0.01)

    upload = st.file_uploader("Upload a football match image", type=["jpg", "jpeg", "png"])
    if upload is None:
        st.info("Upload an image to begin.")
        return
    data = upload.getvalue()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        st.error(f"Could not read this image: {e}")
        return
    st.image(image, caption=f"Original ({image.width}x{image.height})")

    raw_key = f"{hashlib.md5(data).hexdigest()}|{mode}"
    if st.button("Analyze Image", type="primary"):
        t0 = time.time()
        with st.spinner("Detecting players and the ball..."):
            result = run_detection(model, image, device, inf, mode)
        st.session_state["raw"] = {"key": raw_key, "detections": result.detections,
                                   "seconds": time.time() - t0}
    raw = st.session_state.get("raw")
    if raw is None or raw["key"] != raw_key:
        st.info("Press **Analyze Image** to run detection with the current mode.")
        return

    base = analyze(image, raw["detections"], threshold, cfg["team"])
    sig = f"{raw_key}|{threshold:.2f}"
    result_box = st.container()          # filled below, after manual team edits
    labels = list(base.labels)
    st.subheader("Players")
    st.caption("Team grouping is a color heuristic. Correct any player below and the "
               "counts and image update.")
    if base.players:
        df = pd.DataFrame({"Player": range(1, len(base.players) + 1),
                           "Confidence": [round(d.score, 3) for d in base.players],
                           "Team": base.labels})
        edited = st.data_editor(
            df, key=f"editor|{sig}", hide_index=True, disabled=["Player", "Confidence"],
            column_config={"Team": st.column_config.SelectboxColumn(
                "Team", options=[TEAM_A, TEAM_B, UNKNOWN], required=True)})
        overrides = {i: t for i, (old, t) in enumerate(zip(base.labels, edited["Team"]))
                     if old != t}
        labels = apply_overrides(base.labels, overrides)
    final = Analysis(base.players, base.balls, base.teams, labels)
    annotated = render(image, final)

    with result_box:
        st.subheader("Result")
        st.image(annotated, caption="Orange = Team A, blue = Team B, grey = Unknown, "
                                    "red = ball")
        counts = final.team_counts()
        cols = st.columns(5)
        cols[0].metric("Players", len(final.players))
        cols[1].metric("Footballs", len(final.balls))
        cols[2].metric("Team A", counts[TEAM_A])
        cols[3].metric("Team B", counts[TEAM_B])
        cols[4].metric("Unknown", counts[UNKNOWN])
        if final.balls:
            st.write("Football confidence: " + ", ".join(f"{b.score:.2f}" for b in final.balls))
        if base.teams.note:
            st.warning(base.teams.note)
        if base.teams.colors:
            st.write("Dominant team colors found (heuristic)")
            for col, (team, rgb) in zip(st.columns(len(base.teams.colors)),
                                        base.teams.colors.items()):
                col.color_picker(team, "#%02x%02x%02x" % rgb, disabled=True,
                                 key=f"color|{team}|{sig}")
        buf = io.BytesIO()
        annotated.save(buf, format="PNG")
        st.download_button("Download annotated image", buf.getvalue(),
                           file_name="footvision_result.png", mime="image/png")
        st.caption(f"Detection took {raw['seconds']:.1f}s in {mode} mode. Counts are "
                   "visible detections only, not an assumption of 22 players.")


def main() -> None:
    st.set_page_config(page_title="FootVision", page_icon="⚽", layout="wide")
    cfg = load_config()
    with st.sidebar:
        st.title("FootVision")
        page = st.radio("Navigation", PAGES)
        st.divider()
    {"Player Detection": detection_page,
     "Dataset Information": dataset_page,
     "Model Information": model_page}[page](cfg)

"""
Local CLI for testing the 7 FLEURS region-specific wav2vec2 language ID
models — no FastAPI, no Docker, no GCP. Just: point it at a checkpoint
directory and an audio file, get a prediction back.

Usage:
    # Restrict to one region's model (faster)
    python predict.py --audio sample.wav --region western_europe

    # Omit --region to run all 7 loaded models and return the best match
    python predict.py --audio sample.wav

    # Point at a different checkpoint layout
    python predict.py --audio sample.wav --model-root /path/to/model --checkpoint-name ckpt_step5000.pt

Expects the same layout as the full service:
    model/
      western_europe/best.pt
      eastern_europe/best.pt
      central_asia_middle_east_north_africa/best.pt
      sub_saharan_africa/best.pt
      south_asia/best.pt
      south_east_asia/best.pt
      cjk/best.pt
"""
import argparse
import os

import torch
import torchaudio
from transformers import AutoConfig, AutoFeatureExtractor

from fleurs_config import FLEURS_GROUP_INFO
from model_arch import Wav2Vec2ForSpeechClassification

BASE_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE = 16000
REGIONS = list(FLEURS_GROUP_INFO.keys())

# ASSUMPTION: train.py's CLI defaults pooling_mode to "mean" and nothing in
# the checkpoint records which mode a given region actually used. Getting
# this wrong won't error, it'll silently misconstruct the pooling step.
# Override any region here if it was trained with --pooling_mode sum/max.
POOLING_MODE_OVERRIDES = {
    # "cjk": "max",
}
DEFAULT_POOLING_MODE = "mean"

_registry = {}
_feature_extractor = None


def get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL_NAME)
    return _feature_extractor


def load_model(region: str, model_root: str, checkpoint_name: str):
    if region not in REGIONS:
        raise ValueError(f"Unknown region '{region}'. Valid regions: {REGIONS}")

    cache_key = (region, model_root, checkpoint_name)
    if cache_key not in _registry:
        ckpt_path = os.path.join(model_root, region, checkpoint_name)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint found at '{ckpt_path}'. Check --model-root and "
                f"--checkpoint-name, and that this region's .pt file is actually there."
            )

        region_info = FLEURS_GROUP_INFO[region]
        configs = region_info["configs"]   # FLEURS short codes, e.g. "en_us" — training's label ordering
        names = region_info["names"]       # pretty names in the SAME order as configs
        num_labels = len(configs)
        id2label = {i: names[i] for i in range(num_labels)}
        label2id = {v: k for k, v in id2label.items()}

        config = AutoConfig.from_pretrained(
            BASE_MODEL_NAME,
            num_labels=num_labels,
            label2id=label2id,
            id2label=id2label,
            finetuning_task="wav2vec2_langid",
        )
        config.pooling_mode = POOLING_MODE_OVERRIDES.get(region, DEFAULT_POOLING_MODE)

        model = Wav2Vec2ForSpeechClassification(config)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
        model.load_state_dict(state_dict)
        model.eval()

        _registry[cache_key] = {"model": model, "id2label": id2label}

    entry = _registry[cache_key]
    return entry["model"], get_feature_extractor(), entry["id2label"]


def predict_one_region(waveform: torch.Tensor, region: str, model_root: str, checkpoint_name: str, k: int):
    model, feature_extractor, id2label = load_model(region, model_root, checkpoint_name)

    # attention_mask intentionally omitted -- see model_arch.py note: single-
    # utterance inference has no padding, so the unmasked mean-pool branch is
    # both correct and simpler, and sidesteps a raw-length-vs-downsampled-
    # length shape mismatch in merged_strategy's masked branch.
    inputs = feature_extractor(
        waveform.squeeze().numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )

    with torch.no_grad():
        # NOTE: forward() returns a raw logits tensor, not a `.logits` attr.
        logits = model(input_values=inputs["input_values"], attention_mask=None)
        probs = torch.softmax(logits, dim=-1).squeeze()

    k = min(k, probs.shape[-1])
    top_probs, top_indices = torch.topk(probs, k=k)

    return [
        (id2label[int(idx)], round(float(prob), 4), region)
        for prob, idx in zip(top_probs, top_indices)
    ]


def load_audio(path: str):
    waveform, sample_rate = torchaudio.load(path)
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:  # collapse stereo -> mono
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def main():
    parser = argparse.ArgumentParser(description="Local spoken language ID prediction")
    parser.add_argument("--audio", required=True, help="Path to a .wav/.flac/.mp3/.ogg file")
    parser.add_argument(
        "--region", choices=REGIONS, default=None,
        help="Restrict to one region's model. Omit to run all 7 and return the best match.",
    )
    parser.add_argument("--model-root", default="./model", help="Directory containing per-region checkpoint folders")
    parser.add_argument("--checkpoint-name", default="best.pt", help="Checkpoint filename inside each region folder")
    parser.add_argument("--top-k", type=int, default=3, help="Number of top predictions to show")
    args = parser.parse_args()

    if not os.path.isfile(args.audio):
        raise FileNotFoundError(f"Audio file not found: {args.audio}")

    waveform = load_audio(args.audio)

    regions_to_run = [args.region] if args.region else REGIONS
    all_predictions = []
    skipped = []
    for region in regions_to_run:
        try:
            all_predictions.extend(predict_one_region(waveform, region, args.model_root, args.checkpoint_name, args.top_k))
        except FileNotFoundError as e:
            skipped.append(region)
            print(f"[skip] {region}: {e}")

    if not all_predictions:
        raise SystemExit("No region models could be loaded -- check --model-root and checkpoint files.")

    all_predictions.sort(key=lambda p: p[1], reverse=True)
    top_k = all_predictions[: args.top_k]

    print()
    print(f"Audio: {args.audio}")
    print(f"Regions run: {[r for r in regions_to_run if r not in skipped]}")
    print(f"Top {len(top_k)} prediction(s):")
    for rank, (language, confidence, region) in enumerate(top_k, start=1):
        print(f"  {rank}. {language:20s} confidence={confidence:.4f}  (region: {region})")


if __name__ == "__main__":
    main()

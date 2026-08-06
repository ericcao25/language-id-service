import io
import os
from typing import List, Optional

import torch
import torchaudio
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from transformers import AutoConfig, AutoFeatureExtractor

from fleurs_config import FLEURS_GROUP_INFO
from model_arch import Wav2Vec2ForSpeechClassification

MODEL_ROOT = os.environ.get("MODEL_ROOT", "./models")
BASE_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"
SAMPLE_RATE = 16000
REGIONS = list(FLEURS_GROUP_INFO.keys())

POOLING_MODE_OVERRIDES = {}
DEFAULT_POOLING_MODE = "mean"

app = FastAPI(
    title="Spoken Language ID Service",
    description="Serves 7 FLEURS-region wav2vec2 models for spoken language identification.",
    version="1.0.0",
)

_registry = {}
_feature_extractor = None


def get_feature_extractor():
    global _feature_extractor
    if _feature_extractor is None:
        _feature_extractor = AutoFeatureExtractor.from_pretrained(BASE_MODEL_NAME)
    return _feature_extractor


def load_model(region: str):
    if region not in REGIONS:
        raise ValueError(f"Unknown region '{region}'. Valid regions: {REGIONS}")

    if region not in _registry:
        ckpt_path = os.path.join(MODEL_ROOT, region + ".pt")
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(
                f"No checkpoint found at '{ckpt_path}'. Check MODEL_ROOT and "
                f"that this region's .pt file is actually there."
            )

        region_info = FLEURS_GROUP_INFO[region]
        configs = region_info["configs"]
        names = region_info["names"]
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

        _registry[region] = {"model": model, "id2label": id2label}

    entry = _registry[region]
    return entry["model"], get_feature_extractor(), entry["id2label"]


def load_all_models():
    for region in REGIONS:
        try:
            load_model(region)
        except RuntimeError as e:
            print(f"[warning] {e}")


class LanguagePrediction(BaseModel):
    language: str
    confidence: float
    region: str


class PredictResponse(BaseModel):
    top_prediction: LanguagePrediction
    top_k: List[LanguagePrediction]
    regions_considered: List[str]


@app.on_event("startup")
def startup_event():
    load_all_models()


@app.get("/health")
def health():
    return {"status": "ok", "regions_loaded": sorted(_registry.keys())}


@app.get("/regions")
def list_regions():
    return {"available": REGIONS, "loaded": sorted(_registry.keys())}


def predict_one_region(waveform: torch.Tensor, region: str, k: int):
    model, feature_extractor, id2label = load_model(region)

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
        LanguagePrediction(language=id2label[int(idx)], confidence=round(float(prob), 4), region=region)
        for prob, idx in zip(top_probs, top_indices)
    ]


def predict_from_waveform(waveform: torch.Tensor, sample_rate: int, k: int = 3, region: Optional[str] = None) -> PredictResponse:
    if sample_rate != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)(waveform)
    if waveform.shape[0] > 1:  # collapse stereo -> mono
        waveform = waveform.mean(dim=0, keepdim=True)
 
    regions_to_run = [region] if region is not None else ([r for r in REGIONS if r in _registry] or REGIONS)
 
    all_predictions = []
    for r in regions_to_run:
        all_predictions.extend(predict_one_region(waveform, r, k))
 
    all_predictions.sort(key=lambda p: p.confidence, reverse=True)
    top_k = all_predictions[:k]
 
    return PredictResponse(top_prediction=top_k[0], top_k=top_k, regions_considered=regions_to_run)
 
 
@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...), top_k: int = 3, region: Optional[str] = Query(
        default=None,
        description=f"Restrict to one region's model. One of: {REGIONS}. Omit to run all 7 and return the best match.",
    ),
):
    if not file.filename.lower().endswith((".wav", ".flac", ".mp3", ".ogg")):
        raise HTTPException(status_code=400, detail="Unsupported audio format")
    if region is not None and region not in REGIONS:
        raise HTTPException(status_code=400, detail=f"Unknown region '{region}'. Valid: {REGIONS}")
 
    try:
        audio_bytes = await file.read()
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode audio: {e}")
 
    return predict_from_waveform(waveform, sample_rate, k=top_k, region=region)

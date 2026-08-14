import argparse
import os

import torch
from transformers import AutoConfig

from fleurs_config import FLEURS_GROUP_INFO
from model_arch import Wav2Vec2ForSpeechClassification


BASE_MODEL_NAME = "facebook/wav2vec2-xls-r-300m"


def _resolve_quant_engine() -> str:
    override = os.environ.get("QUANT_ENGINE")
    if override:
        return override
    supported = torch.backends.quantized.supported_engines
    for candidate in ("onednn", "qnnpack"):
        # linux/amd64 -> ondnn, arm64 -> qnnpack
        if candidate in supported:
            return candidate
    raise RuntimeError(f"No usable quantization backend found. Supported engines on this machine: {supported}")


def _extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict) and "model" in checkpoint_obj:
        return checkpoint_obj["model"]
    return checkpoint_obj


def _build_model(region: str) -> Wav2Vec2ForSpeechClassification:
    if region not in FLEURS_GROUP_INFO:
        raise ValueError(f"Unknown region '{region}'. Valid: {list(FLEURS_GROUP_INFO.keys())}")

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
    config.pooling_mode = "mean"
    return Wav2Vec2ForSpeechClassification(config)


def strip_checkpoint(input_path: str, output_path: str, int8: bool = False, region: str = None) -> None:
    if int8 and not region:
        raise ValueError("--int8 requires --region (needed to rebuild the quantized module structure).")

    full_checkpoint = torch.load(input_path, map_location="cpu")
    model_state = _extract_state_dict(full_checkpoint)

    if int8:
        engine = _resolve_quant_engine()
        torch.backends.quantized.engine = engine
        model = _build_model(region)
        model.load_state_dict(model_state)
        model.eval()
        quantized_model = torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8
        )
        torch.save(quantized_model.state_dict(), output_path)
    else:
        torch.save(model_state, output_path)

    input_size = os.path.getsize(input_path)
    output_size = os.path.getsize(output_path)
    reduction = 100 * (1 - output_size / input_size)
    print(f"{input_path}: {input_size/1e9:.2f} GB -> {output_path}: {output_size/1e9:.2f} GB "
          f"({reduction:.0f}% smaller)")


def main():
    parser = argparse.ArgumentParser(description="Strip optimizer state from a training checkpoint, optionally compress further")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--region", default=None)
    args = parser.parse_args()
    strip_checkpoint(args.input, args.output, int8=args.int8, region=args.region)


if __name__ == "__main__":
    main()
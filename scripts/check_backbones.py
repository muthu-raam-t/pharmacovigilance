"""Finds which Hugging Face id is available (in the local cache) for each of the five backbones
and writes results/backbones.json, which run_all_experiments.sh reads.

Run it before an overnight training run so a missing model is caught in seconds, not hours.
"""
import json
import sys
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402

if __name__ == "__main__":
    found, missing = {}, []
    for key, (display, candidates) in cfg.BACKBONE_CANDIDATES.items():
        for hf_id in candidates:
            try:
                AutoConfig.from_pretrained(hf_id)
                AutoTokenizer.from_pretrained(hf_id)
                found[key] = {"name": display, "hf_id": hf_id}
                print(f"[OK]      {display:<11} -> {hf_id}")
                break
            except Exception:
                continue
        else:
            missing.append(display)
            print(f"[MISSING] {display:<11} tried: {', '.join(candidates)}")

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = cfg.RESULTS_DIR / "backbones.json"
    with open(out, "w") as f:
        json.dump(found, f, indent=2)
    print(f"\nSaved {out}")
    if missing:
        print("\nSome backbones were not found in the local Hugging Face cache.")
        print("If the container has internet access, run once with downloads allowed:")
        print("    HF_HUB_OFFLINE=0 python check_backbones.py")
        sys.exit(1)

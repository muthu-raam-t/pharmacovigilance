"""Runs a 1-epoch training + evaluation on a tiny subset to check the full loop works
(takes a minute; run it before starting the long training run)."""
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "models"))
import pv_config as cfg  # noqa: E402

BASE = cfg.PROCESSED_DIR
SMOKE_DIR = cfg.CHECKPOINT_DIR / "smoke_test"

if __name__ == "__main__":
    backbone = sys.argv[1] if len(sys.argv) > 1 else None
    if backbone is None:
        bb_file = cfg.RESULTS_DIR / "backbones.json"
        backbone = json.load(open(bb_file))["bert_base"]["hf_id"] if bb_file.exists() else "bert-base-uncased"

    for split, n in [("train", 24), ("val", 12), ("test", 12)]:
        with open(BASE / f"ner_re_{split}.jsonl") as f:
            lines = f.readlines()[:n]
        with open(BASE / f"smoke_{split}.jsonl", "w") as f:
            f.writelines(lines)

    # keep smoke-test outputs out of the real results folder
    env = dict(os.environ, RESULTS_DIR=str(SMOKE_DIR / "results"))
    print(f"Smoke subsets created. Running 1-epoch training with {backbone} ...")
    train = subprocess.run([sys.executable, str(REPO / "scripts/train_joint_model.py"),
                            "--backbone", backbone, "--model_name", "smoke",
                            "--train_path", str(BASE / "smoke_train.jsonl"),
                            "--val_path", str(BASE / "smoke_val.jsonl"),
                            "--output_dir", str(SMOKE_DIR), "--epochs", "1", "--batch_size", "4",
                            "--num_workers", "0"], env=env)
    if train.returncode != 0:
        print("\nSmoke test FAILED during training - see the traceback above.")
        sys.exit(1)

    evaluate = subprocess.run([sys.executable, str(REPO / "scripts/evaluate_model.py"),
                               "--backbone", backbone, "--model_name", "smoke", "--model_key", "smoke_test",
                               "--checkpoint", str(SMOKE_DIR / "model.pt"),
                               "--test_path", str(BASE / "smoke_test.jsonl"),
                               "--output_csv", str(SMOKE_DIR / "results" / "smoke_comparison.csv")], env=env)
    if evaluate.returncode != 0:
        print("\nSmoke test FAILED during evaluation - see the traceback above.")
        sys.exit(1)
    print("\nSmoke test PASSED - training and evaluation ran end to end.")

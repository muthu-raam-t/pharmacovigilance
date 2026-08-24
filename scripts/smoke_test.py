import json
import subprocess
import sys

BASE = "/workspace/data/processed"

with open(f"{BASE}/ner_re_train.jsonl") as f:
    lines = f.readlines()[:20]
with open(f"{BASE}/smoke_train.jsonl", "w") as f:
    f.writelines(lines)

with open(f"{BASE}/ner_re_val.jsonl") as f:
    lines = f.readlines()[:10]
with open(f"{BASE}/smoke_val.jsonl", "w") as f:
    f.writelines(lines)

print("Smoke subset files created. Running 1-epoch training test...")

result = subprocess.run([
    sys.executable, "/workspace/scripts/train_joint_model.py",
    "--backbone", "bert-base-uncased",
    "--train_path", f"{BASE}/smoke_train.jsonl",
    "--val_path", f"{BASE}/smoke_val.jsonl",
    "--output_dir", "/workspace/models/checkpoints/smoke_test",
    "--epochs", "1",
    "--batch_size", "2"
])

if result.returncode == 0:
    print("\nSmoke test PASSED — training loop ran end to end without errors.")
else:
    print("\nSmoke test FAILED — see traceback above.")

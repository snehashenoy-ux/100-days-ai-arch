"""Split the fp16 model into <25 MB parts (Cloudflare Pages per-file limit)."""
from pathlib import Path
import json

src = Path(__file__).parent / "build" / "model_fp16.onnx"
dst = Path(__file__).parents[1] / "public" / "model"
dst.mkdir(parents=True, exist_ok=True)
for old in dst.glob("part*"):
    old.unlink()
data, size, parts = src.read_bytes(), 20_000_000, []
for i in range(0, len(data), size):
    name = f"part{i // size:02d}.bin"
    (dst / name).write_bytes(data[i:i + size])
    parts.append(name)
(dst / "manifest.json").write_text(json.dumps({"total": len(data), "parts": parts}, indent=1))
print(len(data), parts)

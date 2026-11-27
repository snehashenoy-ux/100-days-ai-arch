"""Check the ONNX export against PyTorch and make smaller fp16 / int8 copies."""
from pathlib import Path
import numpy as np, onnx, onnxruntime as ort, torch
from PIL import Image
from transformers import AutoModelForDepthEstimation, AutoImageProcessor
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxruntime.quantization import quantize_dynamic, QuantType

B = Path(__file__).parent / "build"
MID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
proc = AutoImageProcessor.from_pretrained(MID)
model = AutoModelForDepthEstimation.from_pretrained(MID).eval()
img = Image.open(Path(__file__).parents[2] / "test_peix_barcelona.jpg").convert("RGB")
x = proc(images=img, return_tensors="pt")["pixel_values"]
print("input shape", tuple(x.shape))
with torch.no_grad():
    ref = model(pixel_values=x).predicted_depth.numpy()

fp16 = convert_float_to_float16(onnx.load(B / "model_fp32.onnx"), keep_io_types=True)
onnx.save(fp16, B / "model_fp16.onnx")
quantize_dynamic(B / "model_fp32.onnx", B / "model_int8.onnx", weight_type=QuantType.QUInt8)

for name in ["model_fp32", "model_fp16", "model_int8"]:
    p = B / f"{name}.onnx"
    try:
        out = ort.InferenceSession(str(p)).run(None, {"pixel_values": x.numpy()})[0]
        rel = np.abs(out - ref).mean() / ref.mean()
        print(f"{name}: {p.stat().st_size/1e6:.1f} MB  shape {out.shape}  mean rel err {rel*100:.2f}%  depth range {out.min():.1f}-{out.max():.1f} m")
    except Exception as e:
        print(name, "FAILED", e)

# Different aspect ratios must also work (dynamic shapes)
sess = ort.InferenceSession(str(B / "model_fp16.onnx"))
for h, w in [(518, 686), (700, 518), (518, 518), (392, 518)]:
    t = torch.randn(1, 3, h, w) * 0.5
    with torch.no_grad():
        r = model(pixel_values=t).predicted_depth.numpy()
    o = sess.run(None, {"pixel_values": t.numpy()})[0]
    print(f"fp16 {h}x{w}: out {o.shape}  rel err {np.abs(o-r).mean()/np.abs(r).mean()*100:.2f}%")

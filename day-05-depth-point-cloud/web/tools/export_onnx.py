"""Export Depth Anything V2 Metric-Outdoor-Small to ONNX for the browser.

Run from day-05-depth-point-cloud/:
    .venv/bin/python web/tools/export_onnx.py
Writes web/tools/build/model_fp32.onnx (and quantized variants).
"""
from pathlib import Path
import torch
from transformers import AutoModelForDepthEstimation

MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"
OUT = Path(__file__).parent / "build"
OUT.mkdir(exist_ok=True)


class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).predicted_depth


# The stock head wraps the output size in int(), which freezes it to the
# example image's size during export. Same maths, minus the int().
from transformers.models.depth_anything import modeling_depth_anything as mda


def head_forward(self, hidden_states, patch_height, patch_width):
    x = self.conv1(hidden_states[self.head_in_index])
    x = torch.nn.functional.interpolate(
        x, (patch_height * self.patch_size, patch_width * self.patch_size),
        mode="bilinear", align_corners=True)
    x = self.conv3(self.activation1(self.conv2(x)))
    return (self.activation2(x) * self.max_depth).squeeze(dim=1)


mda.DepthAnythingDepthEstimationHead.forward = head_forward

model = Wrapper(AutoModelForDepthEstimation.from_pretrained(MODEL_ID)).eval()
dummy = torch.randn(1, 3, 518, 686)
# Image sides must be multiples of the 14 px patch size; keep them symbolic
# so any aspect ratio works in the browser.
ph = torch.export.Dim("ph", min=8, max=80)
pw = torch.export.Dim("pw", min=8, max=80)
onnx_program = torch.onnx.export(
    model, (dummy,),
    input_names=["pixel_values"], output_names=["predicted_depth"],
    dynamic_shapes={"pixel_values": {2: 14 * ph, 3: 14 * pw}},
    opset_version=18, dynamo=True, external_data=False,
)
onnx_program.save(OUT / "model_fp32.onnx")
print("saved", OUT / "model_fp32.onnx")

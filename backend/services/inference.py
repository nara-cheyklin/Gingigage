import base64
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

from backend.config.settings import MODEL_PATH, KGW_THRESHOLD_MM, CAMERA_INTRINSICS, DEPTH_UNIT_SCALE


# -------------------------------
# Dummy model
# Replace later with your real model
# -------------------------------
class DummyModel:
    def eval(self):
        pass

    def __call__(self, x):
        # Fake mask for testing
        return torch.rand((1, 1, 256, 256))


model = DummyModel()

try:
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
except Exception:
    model = DummyModel()


# -------------------------------
# Preprocess RGB image
# -------------------------------
def preprocess(image_bytes):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    original_size = image.size  # (width, height)

    image = image.resize((256, 256))
    image_np = np.array(image) / 255.0
    image_np = np.transpose(image_np, (2, 0, 1))
    tensor = torch.tensor(image_np, dtype=torch.float32).unsqueeze(0)

    return tensor, original_size


# -------------------------------
# Resize mask back to original image size
# -------------------------------
def resize_mask_to_original(mask, original_size):
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_img = mask_img.resize(original_size, Image.NEAREST)
    return (np.array(mask_img) > 127).astype(np.uint8)


# -------------------------------
# Find two measurement points
# Placeholder logic:
# choose topmost and bottommost mask pixels in the center column
# Replace later with your actual MGJ/KG boundary logic
# -------------------------------
def find_measurement_points(binary_mask):
    h, w = binary_mask.shape
    center_x = w // 2

    ys = np.where(binary_mask[:, center_x] > 0)[0]

    if len(ys) < 2:
        # fallback: search globally
        coords = np.column_stack(np.where(binary_mask > 0))
        if len(coords) < 2:
            raise RuntimeError("Could not find enough mask points for measurement")

        top = coords[np.argmin(coords[:, 0])]
        bottom = coords[np.argmax(coords[:, 0])]
        p1 = (int(top[1]), int(top[0]))      # (u, v)
        p2 = (int(bottom[1]), int(bottom[0]))
        return p1, p2

    p1 = (center_x, int(ys.min()))
    p2 = (center_x, int(ys.max()))
    return p1, p2


# -------------------------------
# Convert pixel + depth to 3D
# -------------------------------
def pixel_to_3d(u, v, depth_value, intrinsics):
    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    X = (u - cx) * depth_value / fx
    Y = (v - cy) * depth_value / fy
    Z = depth_value

    return np.array([X, Y, Z], dtype=np.float32)


# -------------------------------
# Compute real-world distance in mm
# -------------------------------
def calculate_kgw_from_depth(binary_mask, depth_map, intrinsics, depth_scale=1.0):
    p1, p2 = find_measurement_points(binary_mask)

    u1, v1 = p1
    u2, v2 = p2

    if not (0 <= v1 < depth_map.shape[0] and 0 <= u1 < depth_map.shape[1]):
        raise RuntimeError("Point 1 is outside depth map bounds")

    if not (0 <= v2 < depth_map.shape[0] and 0 <= u2 < depth_map.shape[1]):
        raise RuntimeError("Point 2 is outside depth map bounds")

    z1 = float(depth_map[v1, u1]) * depth_scale
    z2 = float(depth_map[v2, u2]) * depth_scale

    if z1 <= 0 or z2 <= 0:
        raise RuntimeError("Invalid depth values at measurement points")

    p3d_1 = pixel_to_3d(u1, v1, z1, intrinsics)
    p3d_2 = pixel_to_3d(u2, v2, z2, intrinsics)

    distance_mm = np.linalg.norm(p3d_2 - p3d_1)

    return round(float(distance_mm), 2), p1, p2


# -------------------------------
# Confidence placeholder
# -------------------------------
def confidence_logic(mask):
    return round(float(np.mean(mask)), 2)


# -------------------------------
# Draw measurement annotations on the original image and return as base64 JPEG
# -------------------------------
def annotate_image(image_bytes, point_1, point_2, kgw_mm):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(image)

    r = max(4, image.width // 80)
    dot_color = (0, 255, 0)
    line_color = (0, 200, 255)

    draw.line([point_1, point_2], fill=line_color, width=max(2, r // 2))

    for pt in (point_1, point_2):
        x, y = pt
        draw.ellipse([x - r, y - r, x + r, y + r], fill=dot_color, outline=(0, 0, 0), width=1)

    label = f"KGW: {kgw_mm} mm"
    margin = 8
    try:
        font = ImageFont.truetype("arial.ttf", size=max(14, image.width // 30))
    except Exception:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    tx = max(margin, min(point_1[0], image.width - text_w - margin))
    ty = max(margin, point_1[1] - text_h - margin * 2)
    draw.rectangle([tx - margin, ty - margin, tx + text_w + margin, ty + text_h + margin], fill=(0, 0, 0, 180))
    draw.text((tx, ty), label, fill=(255, 255, 255), font=font)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


# -------------------------------
# Main inference
# depth_map must already be aligned to RGB
# -------------------------------
def run_inference(image_bytes, depth_map):
    input_tensor, original_size = preprocess(image_bytes)

    with torch.no_grad():
        output = model(input_tensor)

    mask = output.squeeze().cpu().numpy()
    binary_mask_small = (mask > 0.5).astype(np.uint8)

    # Resize segmentation result to original RGB size
    binary_mask = resize_mask_to_original(binary_mask_small, original_size)

    # depth_map should match original RGB size
    if depth_map.shape[:2] != (original_size[1], original_size[0]):
        raise RuntimeError(
            f"Depth map shape {depth_map.shape} does not match original RGB size {original_size[::-1]}"
        )

    kgw_mm, point_1, point_2 = calculate_kgw_from_depth(
        binary_mask=binary_mask,
        depth_map=depth_map,
        intrinsics=CAMERA_INTRINSICS,
        depth_scale=DEPTH_UNIT_SCALE
    )

    confidence = confidence_logic(mask)

    interpretation = (
        "Adequate keratinized gingiva width"
        if kgw_mm >= KGW_THRESHOLD_MM
        else "Inadequate keratinized gingiva width"
    )

    image_base64 = annotate_image(image_bytes, point_1, point_2, kgw_mm)

    return {
        "kgw_mm": kgw_mm,
        "confidence": confidence,
        "interpretation": interpretation,
        "measurement_points": {
            "point_1": {"u": point_1[0], "v": point_1[1]},
            "point_2": {"u": point_2[0], "v": point_2[1]}
        },
        "image_base64": image_base64
    }
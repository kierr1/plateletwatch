# inference_server.py — ONNX Runtime edition
# Run with: python inference_server.py
# Requires: pip install fastapi uvicorn onnxruntime pillow python-multipart numpy
#
# Drop-in replacement for the Ultralytics/PyTorch version. Same request and
# response shape, same endpoints, same zoom-calibration and severity logic —
# only the model backend changed, so the frontend needs zero changes.

import io, base64, os
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import onnxruntime as ort

app = FastAPI(title="PlateletWatch Inference Server (ONNX)")

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Two separate ONNX models ───────────────────────────────────────────────
PLATELET_MODEL_PATH  = os.environ.get("PLATELET_MODEL_PATH",  "./plateletwatch.onnx")
BLOODCELL_MODEL_PATH = os.environ.get("BLOODCELL_MODEL_PATH", "./bloodcellwatch.onnx")

# Both models were exported by Ultralytics at 640x640 (verified via model
# metadata). This is DIFFERENT from the 1280 used by the old PyTorch server —
# do not change unless you re-export the .onnx files at a different size.
IMGSZ = int(os.environ.get("MODEL_IMGSZ", 640))

ORT_PROVIDERS = ["CPUExecutionProvider"]


class OnnxYoloModel:
    """Wraps an ONNX Runtime session with the pieces this server needs:
    load a model, know its class names, run inference, return boxes."""

    def __init__(self, path: str, names: dict):
        self.session = ort.InferenceSession(path, providers=ORT_PROVIDERS)
        self.input_name = self.session.get_inputs()[0].name
        self.names = names  # e.g. {0: "Platelet"} or {0: "WBC", 1: "RBC"}

    def predict(self, padded_arr: np.ndarray, conf: float, iou: float = 0.45):
        """padded_arr: HxWx3 uint8 array, already letterboxed to IMGSZ x IMGSZ.
        Returns list of (xyxy_in_padded_coords, class_id, confidence)."""
        img = padded_arr.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))[None, ...]  # 1,3,H,W
        outputs = self.session.run(None, {self.input_name: img})[0]  # (1, 4+nc, N)

        preds = outputs[0].T  # (N, 4+nc)  -> cx,cy,w,h, class_scores...
        boxes_cxcywh = preds[:, :4]
        class_scores = preds[:, 4:]

        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        mask = confidences > conf
        boxes_cxcywh = boxes_cxcywh[mask]
        class_ids    = class_ids[mask]
        confidences  = confidences[mask]

        if len(boxes_cxcywh) == 0:
            return []

        xyxy = np.zeros_like(boxes_cxcywh)
        xyxy[:, 0] = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        xyxy[:, 1] = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        xyxy[:, 2] = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        xyxy[:, 3] = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        # Per-class NMS (matches Ultralytics' default class-aware behaviour)
        keep_idx = []
        for cid in np.unique(class_ids):
            cls_mask = class_ids == cid
            cls_boxes = xyxy[cls_mask]
            cls_scores = confidences[cls_mask]
            cls_indices = np.where(cls_mask)[0]
            kept = _nms(cls_boxes, cls_scores, iou)
            keep_idx.extend(cls_indices[kept].tolist())

        return [(xyxy[i], int(class_ids[i]), float(confidences[i])) for i in keep_idx]


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float):
    """Plain-numpy NMS. Returns indices (into boxes/scores) to keep."""
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        union = areas[i] + areas[order[1:]] - inter + 1e-9
        iou = inter / union
        order = order[1:][iou <= iou_thresh]
    return np.array(keep, dtype=int)


print(f"[startup] Loading platelet model:  {PLATELET_MODEL_PATH}")
platelet_model = OnnxYoloModel(PLATELET_MODEL_PATH, {0: "Platelet"})
print(f"[startup]   -> classes: {platelet_model.names}")

print(f"[startup] Loading bloodcell model: {BLOODCELL_MODEL_PATH}")
bloodcell_model = OnnxYoloModel(BLOODCELL_MODEL_PATH, {0: "WBC", 1: "RBC"})
print(f"[startup]   -> classes: {bloodcell_model.names}")

PLATELET_CLASSES  = ["Platelet"]
BLOODCELL_CLASSES = ["WBC", "RBC"]


# ── Zoom / Objective Calibration Table ───────────────────────────────────────
# est_per_ul = platelet_count_in_field * correction_factor  (1 image = 1 HPF)
ZOOM_CALIBRATION = {
    "10x":  {
        "label":  "10x Objective (~2200 um FOV)",
        "factor": 1_000,
        "note":   "Low-power scan. Use 40x or 100x for accurate platelet counts.",
    },
    "40x":  {
        "label":  "40x Objective (~550 um FOV) - Standard",
        "factor": 15_000,
        "note":   "Standard high-power field (HPF). Most clinical protocols use 40x.",
    },
    "100x": {
        "label":  "100x Oil Immersion (~220 um FOV)",
        "factor": 100_000,
        "note":   "Oil-immersion HPF. Highest accuracy for individual cell morphology.",
    },
}

DEFAULT_ZOOM = os.environ.get("DEFAULT_ZOOM", "40x")


# ── Helpers ───────────────────────────────────────────────────────────────────

def letterbox_image(img: Image.Image, target: int = IMGSZ):
    w, h = img.size
    scale = min(target / w, target / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = img.resize((new_w, new_h), Image.BILINEAR)
    padded = Image.new("RGB", (target, target), (114, 114, 114))
    pad_left = (target - new_w) // 2
    pad_top  = (target - new_h) // 2
    padded.paste(resized, (pad_left, pad_top))
    return padded, scale, pad_left, pad_top


def unpad_bbox(bbox, scale, pad_left, pad_top):
    x1, y1, x2, y2 = bbox
    return [
        round((x1 - pad_left) / scale, 1),
        round((y1 - pad_top)  / scale, 1),
        round((x2 - pad_left) / scale, 1),
        round((y2 - pad_top)  / scale, 1),
    ]


def run_model(model: OnnxYoloModel, class_names, padded_arr, scale, pad_left, pad_top, conf, label):
    raw = model.predict(padded_arr, conf=conf, iou=0.45)
    print(f"[{label}] conf={conf:.2f} -> detections after NMS: {len(raw)}")

    detections = []
    for xyxy, cls_id, conf_v in raw:
        name = model.names.get(cls_id) or (class_names[cls_id] if cls_id < len(class_names) else "Unknown")
        orig_bbox = unpad_bbox(xyxy.tolist(), scale, pad_left, pad_top)
        print(f"  [{label}] cls={cls_id} ({name}) conf={conf_v:.3f} bbox={orig_bbox}")
        detections.append({
            "class_name": name,
            "confidence": round(conf_v, 4),
            "bbox": orig_bbox,
        })
    return detections


def classify_severity(est_per_ul: int, platelet_count: int) -> dict:
    if platelet_count == 0:
        return {
            "severity": "UNKNOWN",
            "severity_label": "No Platelets Detected",
            "severity_color": "#6b7280",
            "clinical_note": "No platelets were detected in this field. "
                             "Try adjusting confidence threshold, ensure correct magnification, "
                             "or check that the smear quality is adequate.",
        }
    elif est_per_ul < 20_000:
        return {
            "severity": "CRITICAL",
            "severity_label": "Critical - Severe Thrombocytopenia",
            "severity_color": "#dc2626",
            "clinical_note": f"Estimated {est_per_ul:,}/uL. "
                             "Platelet count < 20,000/uL - risk of spontaneous hemorrhage. "
                             "Immediate hospitalization and transfusion evaluation required.",
        }
    elif est_per_ul < 50_000:
        return {
            "severity": "DANGER",
            "severity_label": "Danger - Very Low Platelets",
            "severity_color": "#f97316",
            "clinical_note": f"Estimated {est_per_ul:,}/uL. "
                             "Platelet count 20,000-49,999/uL - very low. "
                             "Close clinical monitoring and possible hospitalization advised.",
        }
    elif est_per_ul < 150_000:
        return {
            "severity": "LOW",
            "severity_label": "Low - Mild Thrombocytopenia",
            "severity_color": "#eab308",
            "clinical_note": f"Estimated {est_per_ul:,}/uL. "
                             "Platelet count 50,000-149,999/uL - below normal. "
                             "Daily CBC monitoring recommended.",
        }
    elif est_per_ul < 400_000:
        return {
            "severity": "NORMAL",
            "severity_label": "Normal Range",
            "severity_color": "#16a34a",
            "clinical_note": f"Estimated {est_per_ul:,}/uL. "
                             "Platelet count within normal adult range (150,000-399,999/uL).",
        }
    else:
        return {
            "severity": "HIGH",
            "severity_label": "High - Thrombocytosis",
            "severity_color": "#7c3aed",
            "clinical_note": f"Estimated {est_per_ul:,}/uL. "
                             "Platelet count >= 400,000/uL - elevated. "
                             "May indicate reactive thrombocytosis (infection, inflammation, iron deficiency) "
                             "or essential thrombocythemia. Follow-up recommended.",
        }


# ── Schemas ───────────────────────────────────────────────────────────────────

class InferenceRequest(BaseModel):
    image:       str
    mediaType:   str   = "image/jpeg"
    confidence:  float = 0.15
    zoom:        str   = DEFAULT_ZOOM
    calib_factor: float | None = None


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/api/analyze-image")
async def analyze_image(req: InferenceRequest):
    try:
        img_bytes = base64.b64decode(req.image)
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image data")

    orig_w, orig_h = img.size
    conf_thresh = min(max(req.confidence, 0.05), 0.50)

    zoom_key = req.zoom.strip().lower().replace(" ", "")
    if not zoom_key.endswith("x"):
        zoom_key = zoom_key + "x"
    if zoom_key.startswith("x"):
        zoom_key = zoom_key[1:] + "x"

    if req.calib_factor is not None:
        calib_factor = req.calib_factor
        zoom_label   = f"Custom (factor={calib_factor:,})"
        zoom_note    = "Manually specified calibration factor."
    elif zoom_key in ZOOM_CALIBRATION:
        cal          = ZOOM_CALIBRATION[zoom_key]
        calib_factor = cal["factor"]
        zoom_label   = cal["label"]
        zoom_note    = cal["note"]
    else:
        cal          = ZOOM_CALIBRATION["40x"]
        calib_factor = cal["factor"]
        zoom_label   = cal["label"] + " [fallback - unrecognised zoom value]"
        zoom_note    = f"Zoom '{req.zoom}' not recognised. Fell back to 40x. Valid options: 10x, 40x, 100x."
        print(f"[warn] Unrecognised zoom '{req.zoom}', falling back to 40x")

    print(f"\n[request] image={orig_w}x{orig_h}  conf={conf_thresh}  zoom={zoom_key}  calib_factor={calib_factor}")

    padded, scale, pad_left, pad_top = letterbox_image(img, IMGSZ)
    padded_arr = np.array(padded)

    platelet_dets = run_model(
        platelet_model, PLATELET_CLASSES,
        padded_arr, scale, pad_left, pad_top,
        conf=conf_thresh, label="platelet",
    )

    bloodcell_dets = run_model(
        bloodcell_model, BLOODCELL_CLASSES,
        padded_arr, scale, pad_left, pad_top,
        conf=min(conf_thresh + 0.05, 0.50), label="bloodcell",
    )

    platelet_count = sum(1 for d in platelet_dets  if d["class_name"] == "Platelet")
    rbc_count      = sum(1 for d in bloodcell_dets if d["class_name"] == "RBC")
    wbc_count      = sum(1 for d in bloodcell_dets if d["class_name"] == "WBC")

    print(f"[result] Platelet={platelet_count}  RBC={rbc_count}  WBC={wbc_count}")

    if platelet_count == 0 and rbc_count == 0 and wbc_count == 0:
        print("[debug] All zero - retrying at conf=0.01 to show raw scores:")
        for label, model in [("platelet", platelet_model), ("bloodcell", bloodcell_model)]:
            debug = model.predict(padded_arr, conf=0.01, iou=0.45)
            if debug:
                scores = sorted([c for _, _, c in debug], reverse=True)[:10]
                print(f"  [{label}] top scores at conf=0.01: {[round(s,3) for s in scores]}")
            else:
                print(f"  [{label}] still zero detections even at conf=0.01 - check model path / image")

    est_per_ul = platelet_count * calib_factor
    sev = classify_severity(est_per_ul, platelet_count)

    print(f"[result] est_per_ul={est_per_ul:,}  severity={sev['severity']}  zoom={zoom_label}")

    return {
        "platelets":      platelet_count,
        "rbc":            rbc_count,
        "wbc":            wbc_count,
        "est_per_ul":     est_per_ul,
        "calib_factor":   calib_factor,
        "zoom":           zoom_label,
        "zoom_note":      zoom_note,
        "severity":       sev["severity"],
        "severity_label": sev["severity_label"],
        "severity_color": sev["severity_color"],
        "clinical_note":  sev["clinical_note"],
        "note":           f"{platelet_count} platelets - {rbc_count} RBC - {wbc_count} WBC detected",
        "detections":     platelet_dets + bloodcell_dets,
        "total_objects":  len(platelet_dets) + len(bloodcell_dets),
        "image_size":     [orig_w, orig_h],
    }


@app.post("/analyze")
async def analyze_legacy(req: InferenceRequest):
    return await analyze_image(req)


@app.get("/health")
async def health():
    return {
        "status":            "ok",
        "backend":           "onnxruntime",
        "platelet_model":    PLATELET_MODEL_PATH,
        "bloodcell_model":   BLOODCELL_MODEL_PATH,
        "platelet_classes":  platelet_model.names,
        "bloodcell_classes": bloodcell_model.names,
        "zoom_options":      {k: v["label"] for k, v in ZOOM_CALIBRATION.items()},
        "default_zoom":      DEFAULT_ZOOM,
    }


@app.get("/calibration")
async def calibration_info():
    return {
        "options":      ZOOM_CALIBRATION,
        "default":      DEFAULT_ZOOM,
        "formula":      "est_per_ul = platelet_count_in_field x correction_factor",
        "fields_note":  "One image = one high-power field (HPF). "
                        "For multi-field averaging, sum platelets across images "
                        "and divide by number of images before multiplying by factor.",
    }


if __name__ == "__main__":
    import uvicorn, socket
    try:
        lan_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        lan_ip = "unknown"
    print(f"\n[server] Listening on:")
    print(f"  Laptop -> http://localhost:8000")
    print(f"  Phone  -> http://{lan_ip}:8000")
    print(f"[server] Backend: onnxruntime  |  IMGSZ={IMGSZ}")
    print(f"[server] Zoom calibration options: {list(ZOOM_CALIBRATION.keys())}")
    print(f"[server] Default zoom: {DEFAULT_ZOOM}\n")
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

"""Clean Diffusers extraction tester (no AI-Toolkit). bf16 + per-block torch.compile (after LoRA) + warmup."""
from __future__ import annotations
import io, os, time, uuid, threading, logging
from pathlib import Path
import torch
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from safetensors.torch import load_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("diffusers-extract")
MODEL = os.environ.get("QWEN_IMAGE_EDIT_MODEL_PATH", "/mnt/models/qwen-image-edit-2511")
LORA_DIR = Path("/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1")
WORK = Path("/mnt/tryon-data/webapp_outputs/diffusers_app_outputs"); WORK.mkdir(parents=True, exist_ok=True)
SAMPLE_SRC = "/mnt/tryon-data/webapp_outputs/outputs/1780590015_extract_bottom_30000_seed7777_69f80bef180f/source.jpg"
PROMPTS = {"top": "GlamTopExt. Extract top wear as a standalone product.",
           "bottom": "GlamBtmExt. Extract bottom wear as a standalone product.",
           "dress": "GlamDressExt. Extract dress as a standalone product."}
DEV, DT = "cuda", torch.bfloat16
COMPILE = os.environ.get("COMPILE", "1") not in {"0", "false", "no"}
DEF_STEPS = int(os.environ.get("DEFAULT_STEPS", "15"))
MAX_INPUT_SIDE = int(os.environ.get("MAX_INPUT_SIDE", "1024"))
TRYON_PERSON_MAX_SIDE = int(os.environ.get("TRYON_PERSON_MAX_SIDE", "1248"))
TRYON_GARMENT_MAX_SIDE = int(os.environ.get("TRYON_GARMENT_MAX_SIDE", "768"))

EXTRACTION_LORAS = [{'key': 'top_27000', 'category': 'top', 'step': '27000', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000027000.safetensors'}, {'key': 'top_26500', 'category': 'top', 'step': '26500', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000026500.safetensors'}, {'key': 'top_26000', 'category': 'top', 'step': '26000', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000026000.safetensors'}, {'key': 'top_25500', 'category': 'top', 'step': '25500', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000025500.safetensors'}, {'key': 'top_25000', 'category': 'top', 'step': '25000', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000025000.safetensors'}, {'key': 'top_24500', 'category': 'top', 'step': '24500', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000024500.safetensors'}, {'key': 'top_24000', 'category': 'top', 'step': '24000', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000024000.safetensors'}, {'key': 'top_23500', 'category': 'top', 'step': '23500', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000023500.safetensors'}, {'key': 'top_23000', 'category': 'top', 'step': '23000', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000023000.safetensors'}, {'key': 'top_22500', 'category': 'top', 'step': '22500', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000022500.safetensors'}, {'key': 'top_22000', 'category': 'top', 'step': '22000', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000022000.safetensors'}, {'key': 'top_21500', 'category': 'top', 'step': '21500', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000021500.safetensors'}, {'key': 'top_21000', 'category': 'top', 'step': '21000', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000021000.safetensors'}, {'key': 'top_20500', 'category': 'top', 'step': '20500', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000020500.safetensors'}, {'key': 'top_20000', 'category': 'top', 'step': '20000', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000020000.safetensors'}, {'key': 'top_19500', 'category': 'top', 'step': '19500', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1/qwen_garment_extract_top126_glamtopext_rank16_b1_continue_12k_to_27k_lr8e5_v1_000019500.safetensors'}, {'key': 'bottom_30000', 'category': 'bottom', 'step': '30000', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_bottom119_glambtmext_rank16_b1_continue_12k_to_22k_lr815e5_v1/qwen_garment_extract_bottom119_glambtmext_rank16_b1_continue_12k_to_22k_lr815e5_v1/qwen_garment_extract_bottom119_glambtmext_rank16_b1_continue_12k_to_22k_lr815e5_v1.safetensors'}, {'key': 'dress_27000', 'category': 'dress', 'step': '27000', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_dress125_glamdressext_rank16_b1_continue_15k_to_30k_lr125e5_v1/qwen_garment_extract_dress125_glamdressext_rank16_b1_continue_15k_to_30k_lr125e5_v1/qwen_garment_extract_dress125_glamdressext_rank16_b1_continue_15k_to_30k_lr125e5_v1_000027000.safetensors'}, {'key': 'dress_30000', 'category': 'dress', 'step': '30000', 'path': '/mnt/qwen-garment-extract/outputs/qwen_garment_extract_dress125_glamdressext_rank16_b1_continue_15k_to_30k_lr125e5_v1/qwen_garment_extract_dress125_glamdressext_rank16_b1_continue_15k_to_30k_lr125e5_v1/qwen_garment_extract_dress125_glamdressext_rank16_b1_continue_15k_to_30k_lr125e5_v1.safetensors'}]

def discover():
    out = []
    for item in EXTRACTION_LORAS:
        path = Path(item["path"])
        if path.is_file():
            row = dict(item)
            row["path"] = str(path)
            out.append(row)
        else:
            log.warning("missing configured extraction LoRA %s at %s", item["key"], path)
    return out

def resize_input_for_model(img, max_side=MAX_INPUT_SIDE):
    original = img.size
    w, h = original
    if max(w, h) <= max_side:
        return img, {
            "original_width": w, "original_height": h,
            "processed_width": w, "processed_height": h,
            "resize_applied": False,
            "resize_rule": f"unchanged; max side {max(w, h)} <= {max_side}",
        }
    if w >= h:
        nw = max_side
        nh = max(16, round((h * max_side / w) / 16) * 16)
    else:
        nh = max_side
        nw = max(16, round((w * max_side / h) / 16) * 16)
    out = img.resize((nw, nh), Image.Resampling.LANCZOS)
    return out, {
        "original_width": w, "original_height": h,
        "processed_width": nw, "processed_height": nh,
        "resize_applied": True,
        "resize_rule": f"max side resized to {max_side}; other side rounded to nearest multiple of 16",
    }

def fit_max_side(img, max_side):
    """Cap longest side to max_side (downscale only), round both dims to /16 for the model."""
    w, h = img.size
    if max(w, h) > max_side:
        if w >= h: nw, nh = max_side, round(h * max_side / w)
        else: nh, nw = max_side, round(w * max_side / h)
    else:
        nw, nh = w, h
    nw = max(16, round(nw / 16) * 16); nh = max(16, round(nh / 16) * 16)
    out = img if (nw, nh) == (w, h) else img.resize((nw, nh), Image.Resampling.LANCZOS)
    return out, (nw, nh)

TRYON_LORA_DIR = Path(os.environ.get("TRYON_LORA_DIR", "/mnt/tryon-data/inference_apps/qwen_tryon_webapp/loras_tryon_last5"))
TRYON_PROMPTS = {
    "top": "Apply GlamifyTopTryon on this person",
    "bottom": "Apply GlamifyBottomTryon on this person",
    "dress": "Apply GlamifyDressTryon on this person",
    "multi": "Apply GlamifyMultiTryon on this person",
}
def discover_tryon():
    out=[]
    for p in sorted(TRYON_LORA_DIR.glob("glamify_*.safetensors")):
        parts=p.stem.split("_")
        if len(parts)<3: continue
        out.append({"key":f"{parts[1]}_{parts[2]}","category":parts[1],"step":parts[2],"path":str(p)})
    return out

class Engine:
    def __init__(self):
        self.pipe=None; self.loaded=set(); self.lock=threading.Lock(); self.loras=discover(); self.tryon_loras=discover_tryon()
        self.ready=False; self.warming=False; self.compiled=False
    def _ensure(self):
        if self.pipe is None:
            from diffusers import QwenImageEditPlusPipeline
            log.info("loading pipeline ..."); self.pipe=QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=DT).to(DEV)
            log.info("pipeline ready, VRAM %.1f GB", torch.cuda.memory_allocated()/1e9)
    def _ensure_lora(self, key, path):
        if key in self.loaded: return
        sd=load_file(path)
        conv={(("transformer."+k[len("diffusion_model."):]) if k.startswith("diffusion_model.") else k):v for k,v in sd.items()}
        self.pipe.load_lora_weights(conv, adapter_name=key); self.loaded.add(key); log.info("loaded LoRA %s", key)
    def _compile(self):
        if COMPILE and not self.compiled and self.pipe is not None:
            for i,b in enumerate(self.pipe.transformer.transformer_blocks):
                self.pipe.transformer.transformer_blocks[i]=torch.compile(b, dynamic=False)
            self.compiled=True; log.info("per-block torch.compile enabled (after LoRA load; RoPE stays eager)")
    def _gen(self, key, images, prompt, steps, seed, w, h, scale):
        self.pipe.set_adapters([key],[float(scale)])
        g=torch.Generator(device=DEV).manual_seed(int(seed))
        with torch.inference_mode():
            return self.pipe(image=images, prompt=prompt, true_cfg_scale=1.0, num_inference_steps=int(steps),
                             height=int(h), width=int(w), generator=g).images[0]
    def warmup_all(self):
        self.warming=True
        try:
            with self.lock:
                self._ensure()
                # lazy-load mode: do NOT load/warm all LoRAs. One warm pass to settle CUDA kernels.
                if self.loras and Path(SAMPLE_SRC).exists():
                    first=self.loras[0]; self._ensure_lora(first["key"], first["path"])
                    src, _ = resize_input_for_model(Image.open(SAMPLE_SRC).convert("RGB"))
                    t=time.perf_counter()
                    self._gen(first["key"], src, PROMPTS.get(first["category"],"extract garment"), DEF_STEPS, 0, 832, 1248, 1.0)
                    log.info("warm pass %.1fs", time.perf_counter()-t)
                self.ready=True; log.info("READY (lazy-load: LoRAs load on first request)")
        except Exception:
            log.exception("warmup failed")
        finally:
            self.warming=False
    def generate(self, key, path, src, prompt, steps, seed, w, h, scale):
        with self.lock:
            self._ensure(); self._ensure_lora(key, path)
            t=time.perf_counter(); img=self._gen(key, src, prompt, steps, seed, w, h, scale)
            return img, round(time.perf_counter()-t, 2)
    def generate_tryon(self, key, path, person, garment, prompt, steps, seed, w, h, scale):
        with self.lock:
            self._ensure(); self._ensure_lora(key, path)
            t=time.perf_counter(); img=self._gen(key, [person, garment], prompt, steps, seed, w, h, scale)
            return img, round(time.perf_counter()-t, 2)

ENG=Engine()
app=FastAPI(title="Diffusers Extraction Tester")

@app.on_event("startup")
def _startup():
    threading.Thread(target=ENG.warmup_all, daemon=True).start()

@app.get("/api/loras")
def loras(mode: str="extract"):
    src = ENG.tryon_loras if mode=="tryon" else ENG.loras
    return JSONResponse([{"key":l["key"],"category":l["category"],"step":l["step"]} for l in src])

@app.get("/api/status")
def status():
    return JSONResponse({"engine":"diffusers","mode":"extract+tryon","lazy_load":True,"ready":ENG.ready,"warming":ENG.warming,
                         "default_steps":DEF_STEPS,"max_input_side":MAX_INPUT_SIDE,"loaded_loras":sorted(ENG.loaded),
                         "extract_loras":len(ENG.loras),"tryon_loras":len(ENG.tryon_loras)})

@app.post("/api/extract")
async def extract(source: UploadFile=File(...), lora_key: str=Form(...), steps: int=Form(DEF_STEPS),
                  seed: int=Form(7777), width: int=Form(832), height: int=Form(1248),
                  lora_scale: float=Form(1.0), prompt: str=Form("")):
    lk={l["key"]:l for l in ENG.loras}
    if lora_key not in lk: raise HTTPException(400, f"unknown lora_key {lora_key}")
    e=lk[lora_key]; p=(prompt.strip() or PROMPTS.get(e["category"],"Extract the garment as a standalone product."))
    job=WORK/f"{int(time.time())}_{lora_key}_{uuid.uuid4().hex[:8]}"; job.mkdir(parents=True, exist_ok=True)
    source_original = Image.open(io.BytesIO(await source.read())).convert("RGB")
    src, input_metrics = resize_input_for_model(source_original)
    src.save(job/"source.jpg", quality=95)
    try: img, secs = ENG.generate(lora_key, e["path"], src, p, steps, seed, width, height, lora_scale)
    except Exception as ex: log.exception("gen failed"); raise HTTPException(500, f"generation failed: {ex}")
    img.save(job/"output.jpg", quality=95); rel=job.relative_to(WORK).as_posix()
    ow, oh = img.size
    return JSONResponse({
        "seconds": secs,
        "prompt": p,
        "steps": steps,
        "lora": lora_key,
        "compiled": ENG.compiled,
        "requested_output_width": int(width),
        "requested_output_height": int(height),
        "actual_output_width": ow,
        "actual_output_height": oh,
        "input_metrics": input_metrics,
        "source_url": f"/files/{rel}/source.jpg",
        "output_url": f"/files/{rel}/output.jpg",
    })

@app.post("/api/tryon")
async def tryon(person: UploadFile=File(...), garment: UploadFile=File(...), lora_key: str=Form(...),
                steps: int=Form(DEF_STEPS), seed: int=Form(7777), width: int=Form(0), height: int=Form(0),
                lora_scale: float=Form(1.0), prompt: str=Form("")):
    lk={l["key"]:l for l in ENG.tryon_loras}
    if lora_key not in lk: raise HTTPException(400, f"unknown tryon lora_key {lora_key}")
    e=lk[lora_key]; p=(prompt.strip() or TRYON_PROMPTS.get(e["category"],"Apply the garment on this person"))
    job=WORK/f"{int(time.time())}_tryon_{lora_key}_{uuid.uuid4().hex[:8]}"; job.mkdir(parents=True, exist_ok=True)
    person_img, (pw, ph) = fit_max_side(Image.open(io.BytesIO(await person.read())).convert("RGB"), TRYON_PERSON_MAX_SIDE)
    garment_img, (gw, gh) = fit_max_side(Image.open(io.BytesIO(await garment.read())).convert("RGB"), TRYON_GARMENT_MAX_SIDE)
    person_img.save(job/"person.jpg", quality=95); garment_img.save(job/"garment.jpg", quality=95)
    ow, oh = pw, ph   # output ALWAYS == processed person dims (try-on must preserve person size)
    try: img, secs = ENG.generate_tryon(lora_key, e["path"], person_img, garment_img, p, steps, seed, ow, oh, lora_scale)
    except Exception as ex: log.exception("tryon failed"); raise HTTPException(500, f"tryon failed: {ex}")
    img.save(job/"output.jpg", quality=95); rel=job.relative_to(WORK).as_posix()
    aw,ah=img.size
    return JSONResponse({"seconds":secs,"prompt":p,"steps":steps,"lora":lora_key,
                         "person_max_side":TRYON_PERSON_MAX_SIDE,"garment_max_side":TRYON_GARMENT_MAX_SIDE,
                         "person_processed_width":pw,"person_processed_height":ph,
                         "garment_processed_width":gw,"garment_processed_height":gh,
                         "output_width":aw,"output_height":ah,"matches_person":(aw==pw and ah==ph),
                         "person_url":f"/files/{rel}/person.jpg","garment_url":f"/files/{rel}/garment.jpg","output_url":f"/files/{rel}/output.jpg"})

@app.get("/files/{path:path}")
def files(path: str):
    full=WORK/path
    if not full.is_file() or not str(full.resolve()).startswith(str(WORK.resolve())): raise HTTPException(404,"nf")
    return FileResponse(full)

@app.get("/")
def index(): return HTMLResponse(PAGE)

PAGE="""
<!doctype html><html><head><meta charset=utf-8><title>Qwen Diffusers Tester — Extract + Try-on</title><style>
body{font-family:system-ui,Arial;margin:0;background:#0f1115;color:#e6e6e6}.wrap{max-width:1040px;margin:0 auto;padding:24px}
h1{font-size:20px}.sub{color:#9aa0aa;font-size:13px;margin-bottom:14px}#st{font-size:12px;margin-bottom:14px;padding:8px;border-radius:8px;background:#171a21}
.card{background:#171a21;border:1px solid #262b36;border-radius:12px;padding:18px;margin-bottom:16px}
label{display:block;font-size:12px;color:#9aa0aa;margin:10px 0 4px}
input,select{width:100%;padding:9px;border-radius:8px;border:1px solid #2c3340;background:#0f1115;color:#e6e6e6;box-sizing:border-box}
.row{display:flex;gap:12px}.row>div{flex:1}
button{margin-top:16px;width:100%;padding:12px;border:0;border-radius:8px;background:#3b82f6;color:#fff;font-size:15px;cursor:pointer}button:disabled{background:#374151;cursor:wait}
.imgs{display:flex;gap:16px}.imgs figure{flex:1;margin:0}.imgs img{width:100%;border-radius:10px;border:1px solid #262b36}
figcaption{font-size:12px;color:#9aa0aa;text-align:center;margin-top:6px}.meta{font-size:13px;color:#7dd3fc;margin-top:8px}
.tabs{display:flex;gap:8px;margin-bottom:14px}.tab{flex:1;padding:10px;text-align:center;border-radius:8px;background:#171a21;border:1px solid #262b36;cursor:pointer}.tab.on{background:#1f6feb;border-color:#1f6feb}
.hide{display:none}
</style></head><body><div class=wrap>
<h1>Qwen Diffusers Tester</h1>
<div class=sub>Same base Qwen-Image-Edit-2511; LoRA swapped per request. Lazy-load (LoRAs load on first use). No AI-Toolkit.</div>
<div id=st>checking status...</div>
<div class=tabs><div class="tab on" id=tab_ex onclick="tab('ex')">Extraction</div><div class=tab id=tab_ty onclick="tab('ty')">Try-on</div></div>

<div id=sec_ex>
<div class=card>
<label>Source image</label><input type=file id=ex_src accept=image/*>
<div class=row><div><label>LoRA</label><select id=ex_lora></select></div><div><label>Steps</label><input type=number id=ex_steps value=8 min=1 max=30></div></div>
<div class=row><div><label>Seed</label><input type=number id=ex_seed value=7777></div><div><label>Output width</label><input type=number id=ex_w value=640></div><div><label>Output height</label><input type=number id=ex_h value=960></div><div><label>LoRA scale</label><input type=number id=ex_scale value=1.0 step=0.1></div></div>
<label>Prompt (blank=auto)</label><input id=ex_prompt placeholder="auto">
<button id=ex_go onclick=runEx()>Generate extraction</button><div class=meta id=ex_meta></div></div>
<div class=card id=ex_rc style=display:none><div class=imgs><figure><img id=ex_osrc><figcaption>source</figcaption></figure><figure><img id=ex_oout><figcaption>extracted</figcaption></figure></div></div>
</div>

<div id=sec_ty class=hide>
<div class=card>
<div class=row><div><label>Person image</label><input type=file id=ty_person accept=image/*></div><div><label>Garment image</label><input type=file id=ty_garment accept=image/*></div></div>
<div class=row><div><label>Try-on LoRA</label><select id=ty_lora></select></div><div><label>Steps</label><input type=number id=ty_steps value=8 min=1 max=30></div></div>
<div class=row><div><label>Seed</label><input type=number id=ty_seed value=7777></div><div><label>LoRA scale</label><input type=number id=ty_scale value=1.0 step=0.1></div></div>
<div class=sub style="margin:6px 0">Output auto-matches the processed person size. Person capped at 1248px · garment capped at 768px (longest side).</div>
<label>Prompt (blank=auto)</label><input id=ty_prompt placeholder="auto">
<button id=ty_go onclick=runTy()>Generate try-on</button><div class=meta id=ty_meta></div></div>
<div class=card id=ty_rc style=display:none><div class=imgs><figure><img id=ty_oper><figcaption>person</figcaption></figure><figure><img id=ty_ogar><figcaption>garment</figcaption></figure><figure><img id=ty_oout><figcaption>try-on result</figcaption></figure></div></div>
</div>

</div><script>
function tab(m){let ex=m==='ex';tab_ex.classList.toggle('on',ex);tab_ty.classList.toggle('on',!ex);sec_ex.classList.toggle('hide',!ex);sec_ty.classList.toggle('hide',ex);}
async function st(){try{let d=await (await fetch('/api/status')).json();
document.getElementById('st').textContent=`engine=diffusers | ${d.warming?'WARMING UP...':(d.ready?'READY ✓':'idle')} | lazy-load | ${d.default_steps} default steps | input max ${d.max_input_side}px | extract LoRAs ${d.extract_loras} | tryon LoRAs ${d.tryon_loras}`;}catch(e){}}
st();setInterval(st,4000);
async function fill(sel,mode){let d=await (await fetch('/api/loras?mode='+mode)).json();sel.innerHTML=d.map(l=>`<option value="${l.key}">${l.category} (step ${l.step})</option>`).join('')}
fill(ex_lora,'extract');fill(ty_lora,'tryon');
async function runEx(){let f=ex_src.files[0];if(!f){alert('pick source image');return}let b=ex_go;b.disabled=true;b.textContent='Generating...';ex_meta.textContent='';
let fd=new FormData();fd.append('source',f);fd.append('lora_key',ex_lora.value);fd.append('steps',ex_steps.value);fd.append('seed',ex_seed.value);fd.append('width',ex_w.value);fd.append('height',ex_h.value);fd.append('lora_scale',ex_scale.value);fd.append('prompt',ex_prompt.value);
try{let r=await fetch('/api/extract',{method:'POST',body:fd});if(!r.ok)throw new Error(await r.text());let d=await r.json();
ex_osrc.src=d.source_url+'?t='+Date.now();ex_oout.src=d.output_url+'?t='+Date.now();ex_rc.style.display='block';
const im=d.input_metrics;ex_meta.innerHTML=`done in ${d.seconds}s | steps=${d.steps} | lora=${d.lora}<br>input ${im.original_width}×${im.original_height} → sent ${im.processed_width}×${im.processed_height} | ${im.resize_rule}<br>output ${d.actual_output_width}×${d.actual_output_height}`;}catch(e){ex_meta.textContent='ERROR: '+e.message}
b.disabled=false;b.textContent='Generate extraction'}
async function runTy(){let p=ty_person.files[0],g=ty_garment.files[0];if(!p||!g){alert('pick person AND garment images');return}let b=ty_go;b.disabled=true;b.textContent='Generating...';ty_meta.textContent='';
let fd=new FormData();fd.append('person',p);fd.append('garment',g);fd.append('lora_key',ty_lora.value);fd.append('steps',ty_steps.value);fd.append('seed',ty_seed.value);fd.append('lora_scale',ty_scale.value);fd.append('prompt',ty_prompt.value);
try{let r=await fetch('/api/tryon',{method:'POST',body:fd});if(!r.ok)throw new Error(await r.text());let d=await r.json();
ty_oper.src=d.person_url+'?t='+Date.now();ty_ogar.src=d.garment_url+'?t='+Date.now();ty_oout.src=d.output_url+'?t='+Date.now();ty_rc.style.display='block';
ty_meta.innerHTML=`done in ${d.seconds}s | steps=${d.steps} | lora=${d.lora}<br>person → ${d.person_processed_width}×${d.person_processed_height} (cap ${d.person_max_side}) · garment → ${d.garment_processed_width}×${d.garment_processed_height} (cap ${d.garment_max_side})<br>output ${d.output_width}×${d.output_height} ${d.matches_person?'✓ matches person':'✗ MISMATCH'}`;}catch(e){ty_meta.textContent='ERROR: '+e.message}
b.disabled=false;b.textContent='Generate try-on'}
</script></body></html>
"""
if __name__=="__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT","8000")), log_level="info")

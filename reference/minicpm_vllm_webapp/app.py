"""MiniCPM-V 4.5 garment captioner via vLLM. Port 8010.

Validated working config (RTX PRO 6000 Blackwell, coexisting with the 8000 app):
  - vLLM 0.22.1 + transformers 4.57.x  (transformers v5 removed tokenizer.im_start_id -> MiniCPM-V breaks)
  - FlashInfer DISABLED (fails sm_120); vLLM auto-uses FLASH_ATTN
  - enforce_eager + max_num_seqs=1 + mm_processor_kwargs.max_slice_nums=4 + expandable_segments
    so it fits in the ~28GB left beside the 8000 Qwen model (~67GB)
  - ~0.5s / 30-45 word caption
Launch env (see scripts/launch_8010_minicpm_vllm.sh):
  HF_HOME=/mnt/hf_cache VLLM_ATTENTION_BACKEND=TORCH_SDPA VLLM_USE_FLASHINFER_SAMPLER=0
  VLLM_USE_FLASHINFER=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
Run with /mnt/venvs/vllm/bin/python (transformers pinned <5).
"""
from __future__ import annotations
import io, os, time, threading, logging
from PIL import Image
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("minicpm-vllm")

MODEL = os.environ.get("CAPTION_MODEL", "openbmb/MiniCPM-V-4_5")
GPU_UTIL = float(os.environ.get("VLLM_GPU_UTIL", "0.27"))
MAX_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "90"))
RESIZE_LONG = int(os.environ.get("RESIZE_LONG", "1024"))
MAX_SLICE_NUMS = int(os.environ.get("MAX_SLICE_NUMS", "4"))
MAX_MODEL_LEN = int(os.environ.get("MAX_MODEL_LEN", "4096"))
SAMPLE = "/mnt/tryon-data/webapp_outputs/outputs/1780590015_extract_bottom_30000_seed7777_69f80bef180f/source.jpg"

def _garment_prompt(part_label, exclude):
    return (
        "You are a fashion product specialist writing a precise description that will be used to "
        f"regenerate this garment as an image. Describe ONLY the {part_label} in this image. "
        f"Completely ignore {exclude} — do not mention them at all. "
        "Write one clear, flowing paragraph of about 30-45 words capturing every visible detail needed "
        "to recreate it: garment type, collar/neckline, closure (buttons/zip/ties), sleeves/straps or "
        "waistline/legs, fit and silhouette, fabric and texture, all colours, and the print/pattern "
        "(motif types, their colours, and where they sit). Only what is clearly visible; never guess "
        "hidden parts. Plain factual prose — no labels, no lists, no headings, no preamble."
    )

PROMPTS = {
    "general": _garment_prompt("main clothing garment", "the background and anything that is not the garment"),
    "top":     _garment_prompt("upper-body garment (the top / shirt / jacket)",
        "the person (face, hair, skin, body, midriff, pose), the lower-body garment (trousers/pants/skirt), footwear, accessories, and the background"),
    "bottom":  _garment_prompt("lower-body garment (the trousers / pants / skirt / shorts)",
        "the person, the upper-body garment (top/shirt), footwear, accessories, and the background"),
    "dress":   _garment_prompt("dress",
        "the person (face, hair, skin, body, pose), footwear, accessories, and the background"),
}

def resize_long(img, target=RESIZE_LONG):
    w, h = img.size
    if max(w, h) <= target: return img
    if w >= h: nw, nh = target, max(8, round(h*target/w))
    else: nh, nw = target, max(8, round(w*target/h))
    return img.resize((nw, nh), Image.Resampling.LANCZOS)

class Engine:
    def __init__(self):
        self.llm=None; self.tok=None; self.sp=None; self.lock=threading.Lock(); self.ready=False; self.warming=False
    def _ensure(self):
        if self.llm is None:
            from vllm import LLM, SamplingParams
            from transformers import AutoTokenizer
            log.info("loading %s via vLLM ...", MODEL)
            self.tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
            self.llm = LLM(model=MODEL, trust_remote_code=True, gpu_memory_utilization=GPU_UTIL,
                           max_model_len=MAX_MODEL_LEN, limit_mm_per_prompt={"image": 1}, dtype="bfloat16",
                           enforce_eager=True, max_num_seqs=1, mm_processor_kwargs={"max_slice_nums": MAX_SLICE_NUMS})
            try:
                sids=[self.tok.convert_tokens_to_ids(t) for t in ["<|im_end|>","<|endoftext|>"]]
                sids=[s for s in sids if isinstance(s,int) and s>=0] or None
            except Exception:
                sids=None
            self.sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS, stop_token_ids=sids)
            log.info("vLLM ready")
    def _caption(self, image, prompt):
        messages=[{"role":"user","content":"(<image>./</image>)\n"+prompt.strip()}]
        # MiniCPM-V 4.5 is Qwen3-based and emits <think>...</think> by default — turn it OFF for clean captions.
        try:
            text=self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text=self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        out=self.llm.generate({"prompt": text, "multi_modal_data": {"image": image}}, self.sp, use_tqdm=False)
        txt=out[0].outputs[0].text.strip()
        # safety net: strip any reasoning block if one still slips through
        if "</think>" in txt: txt=txt.rsplit("</think>",1)[-1].strip()
        txt=txt.replace("<think>","").strip()
        return txt
    def caption(self, image, prompt):
        with self.lock:
            self._ensure()
            t=time.perf_counter(); txt=self._caption(image, prompt)
            return txt, round((time.perf_counter()-t)*1000)
    def warmup(self):
        self.warming=True
        try:
            with self.lock:
                self._ensure()
                if os.path.exists(SAMPLE):
                    for _ in range(2):
                        t=time.perf_counter(); self._caption(resize_long(Image.open(SAMPLE).convert("RGB")), PROMPTS["top"])
                        log.info("warm pass %.2fs", time.perf_counter()-t)
                self.ready=True; log.info("READY")
        except Exception:
            log.exception("warmup failed")
        finally:
            self.warming=False

ENG=Engine()
app=FastAPI(title="MiniCPM-V 4.5 Garment Captioner (vLLM)")

@app.on_event("startup")
def _startup():
    threading.Thread(target=ENG.warmup, daemon=True).start()

@app.get("/api/status")
def status():
    return JSONResponse({"engine":"vllm","model":MODEL,"ready":ENG.ready,"warming":ENG.warming,
                         "gpu_util":GPU_UTIL,"max_new_tokens":MAX_TOKENS,"resize_long":RESIZE_LONG,"max_slice_nums":MAX_SLICE_NUMS})

@app.get("/api/prompts")
def prompts(): return JSONResponse(PROMPTS)

@app.post("/api/caption")
async def caption(source: UploadFile=File(...), type: str=Form("general"), prompt: str=Form("")):
    p = prompt.strip() or PROMPTS.get(type, PROMPTS["general"])
    try:
        img = resize_long(Image.open(io.BytesIO(await source.read())).convert("RGB"))
        txt, ms = ENG.caption(img, p)
    except Exception as ex:
        log.exception("caption failed"); raise HTTPException(500, f"caption failed: {ex}")
    return JSONResponse({"caption":txt, "words":len(txt.split()), "ms":ms, "type":type, "prompt":p})

@app.get("/")
def index(): return HTMLResponse(PAGE)

PAGE="""
<!doctype html><html><head><meta charset=utf-8><title>MiniCPM-V 4.5 Garment Captioner (vLLM)</title><style>
body{font-family:system-ui,Arial;margin:0;background:#0f1115;color:#e6e6e6}.wrap{max-width:840px;margin:0 auto;padding:24px}
h1{font-size:20px}.sub{color:#9aa0aa;font-size:13px;margin-bottom:12px}#st{font-size:12px;margin-bottom:14px;padding:8px;border-radius:8px;background:#171a21}
.card{background:#171a21;border:1px solid #262b36;border-radius:12px;padding:18px;margin-bottom:16px}
label{display:block;font-size:12px;color:#9aa0aa;margin:10px 0 4px}
input,select,textarea{width:100%;padding:9px;border-radius:8px;border:1px solid #2c3340;background:#0f1115;color:#e6e6e6;box-sizing:border-box}
button{margin-top:14px;width:100%;padding:12px;border:0;border-radius:8px;background:#10b981;color:#fff;font-size:15px;cursor:pointer}button:disabled{background:#374151}
.out{margin-top:12px;font-size:15px;line-height:1.55}.meta{font-size:12px;color:#7dd3fc;margin-top:8px}
img#prev{max-width:240px;border-radius:10px;border:1px solid #262b36;margin-top:8px}
</style></head><body><div class=wrap>
<h1>MiniCPM-V 4.5 — Garment Captioner (vLLM)</h1><div class=sub>vLLM on Blackwell. Best-quality garment captions, ~0.5s. Select Type to auto-fill the prompt (editable; that exact text is sent).</div>
<div id=st>status...</div>
<div class=card>
<label>Garment image</label><input type=file id=src accept=image/* onchange="prev.src=URL.createObjectURL(this.files[0])">
<img id=prev>
<div class=row><div><label>Type</label><select id=type><option>general</option><option>top</option><option>bottom</option><option>dress</option></select></div></div>
<label>Prompt (auto-filled from Type — edit freely; THIS exact text is sent)</label><textarea id=pbox rows=7 placeholder="prompt"></textarea>
<button id=go onclick=run()>Generate caption</button>
<div class=out id=out></div><div class=meta id=meta></div></div>
</div><script>
const $=id=>document.getElementById(id);
let PROMPTS={};
function fillPrompt(){$('pbox').value=PROMPTS[$('type').value]||PROMPTS.general||'';}
async function loadPrompts(){try{PROMPTS=await (await fetch('/api/prompts')).json();fillPrompt();}catch(e){}}
$('type').addEventListener('change',fillPrompt); loadPrompts();
async function st(){try{let d=await (await fetch('/api/status')).json();$('st').textContent=`engine=vllm | ${d.model} | ${d.warming?'WARMING UP (first load ~1-2min)...':(d.ready?'READY ✓':'idle')} | max_tokens=${d.max_new_tokens}`;}catch(e){}}st();setInterval(st,4000);
async function run(){let f=$('src').files[0];if(!f){alert('pick image');return}let pr=$('pbox').value.trim();if(!pr){alert('prompt is empty');return}
let b=$('go');b.disabled=true;b.textContent='Generating...';$('out').textContent='';$('meta').textContent='';
let fd=new FormData();fd.append('source',f);fd.append('type',$('type').value);fd.append('prompt',pr);
try{let r=await fetch('/api/caption',{method:'POST',body:fd});if(!r.ok)throw new Error(await r.text());let d=await r.json();
$('out').textContent=d.caption;$('meta').textContent=`${d.ms} ms | ${d.words} words | type=${d.type}`;}catch(e){$('meta').textContent='ERROR: '+e.message}
b.disabled=false;b.textContent='Generate caption'}
</script></body></html>
"""
if __name__=="__main__":
    import uvicorn; uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT","8010")), log_level="info")

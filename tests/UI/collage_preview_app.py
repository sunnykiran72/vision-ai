from __future__ import annotations

import base64
from io import BytesIO
from typing import Annotated

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image, UnidentifiedImageError

from app.utils.tryon_collage import ProductReferenceInput, build_product_reference

app = FastAPI(title="Try-on Collage Preview")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Try-on Collage Preview</title>
    <style>
      :root {
        color-scheme: light;
        --bg: #f5f7fb;
        --panel: #ffffff;
        --ink: #172033;
        --muted: #5d6678;
        --line: #d8deea;
        --accent: #0b6f6a;
      }

      * { box-sizing: border-box; }

      body {
        margin: 0;
        background: var(--bg);
        color: var(--ink);
        font-family:
          Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
          "Segoe UI", sans-serif;
      }

      main {
        display: grid;
        grid-template-columns: minmax(280px, 380px) minmax(0, 1fr);
        min-height: 100vh;
      }

      aside {
        border-right: 1px solid var(--line);
        background: var(--panel);
        padding: 24px;
      }

      section { padding: 24px; }

      h1, h2 {
        margin: 0;
        letter-spacing: 0;
      }

      h1 {
        font-size: 22px;
        line-height: 1.25;
      }

      h2 {
        font-size: 15px;
        margin-bottom: 12px;
      }

      p {
        margin: 8px 0 0;
        color: var(--muted);
        font-size: 14px;
        line-height: 1.45;
      }

      form {
        display: grid;
        gap: 18px;
        margin-top: 24px;
      }

      .field {
        display: grid;
        gap: 8px;
      }

      label {
        color: var(--ink);
        font-size: 13px;
        font-weight: 650;
      }

      input, select, button {
        width: 100%;
        min-height: 40px;
        border: 1px solid var(--line);
        border-radius: 8px;
        background: #fff;
        color: var(--ink);
        font: inherit;
      }

      input { padding: 8px; }
      select { padding: 0 10px; }

      button {
        border-color: var(--accent);
        background: var(--accent);
        color: #fff;
        cursor: pointer;
        font-weight: 700;
      }

      .output {
        border: 1px solid var(--line);
        border-radius: 8px;
        background:
          linear-gradient(45deg, #eef2f7 25%, transparent 25%),
          linear-gradient(-45deg, #eef2f7 25%, transparent 25%),
          linear-gradient(45deg, transparent 75%, #eef2f7 75%),
          linear-gradient(-45deg, transparent 75%, #eef2f7 75%);
        background-color: #fff;
        background-position: 0 0, 0 8px, 8px -8px, -8px 0;
        background-size: 16px 16px;
        min-height: 580px;
        display: grid;
        place-items: center;
        overflow: auto;
        padding: 18px;
      }

      .output img {
        max-width: 100%;
        height: auto;
        background: #fff;
        box-shadow: 0 18px 50px rgb(23 32 51 / 16%);
      }

      .status {
        margin: 0 0 16px;
        font-size: 14px;
        color: var(--muted);
      }

      @media (max-width: 900px) {
        main { grid-template-columns: 1fr; }
        aside {
          border-right: 0;
          border-bottom: 1px solid var(--line);
        }
      }
    </style>
  </head>
  <body>
    <main>
      <aside>
        <h1>Try-on Collage Preview</h1>
        <p>Uploads are rendered by the real Python collage function.</p>

        <form id="form">
          <div class="field">
            <label for="top1_type">Top 1 type</label>
            <select id="top1_type" name="top1_type">
              <option value="top">Top</option>
              <option value="outer">Outer as top</option>
            </select>
          </div>

          <div class="field">
            <label for="top1">Top 1 / outer</label>
            <input id="top1" name="top1" type="file" accept="image/*" />
          </div>

          <div class="field">
            <label for="top2">Top 2</label>
            <input id="top2" name="top2" type="file" accept="image/*" />
          </div>

          <div class="field">
            <label for="bottom">Bottom</label>
            <input id="bottom" name="bottom" type="file" accept="image/*" />
          </div>

          <div class="field">
            <label for="dress">Dress</label>
            <input id="dress" name="dress" type="file" accept="image/*" />
          </div>

          <button type="submit">Render collage</button>
        </form>
      </aside>

      <section>
        <h2>Output</h2>
        <p class="status" id="status">Choose garment images and render.</p>
        <div class="output">
          <img id="preview" alt="Generated collage preview" hidden />
        </div>
      </section>
    </main>

    <script>
      const form = document.getElementById("form");
      const status = document.getElementById("status");
      const preview = document.getElementById("preview");

      form.addEventListener("submit", async (event) => {
        event.preventDefault();
        status.textContent = "Rendering with app.utils.tryon_collage...";
        preview.hidden = true;

        const response = await fetch("/api/collage", {
          method: "POST",
          body: new FormData(form),
        });
        const payload = await response.json();

        if (!response.ok) {
          status.textContent = payload.error || "Unable to render collage.";
          return;
        }

        preview.src = payload.image;
        preview.hidden = false;
        status.textContent = `${payload.mode} · ${payload.width} x ${payload.height}`;
      });
    </script>
  </body>
</html>
"""


@app.post("/api/collage")
async def preview_collage(
    top1_type: Annotated[str, Form()] = "top",
    top1: Annotated[UploadFile | None, File()] = None,
    top2: Annotated[UploadFile | None, File()] = None,
    bottom: Annotated[UploadFile | None, File()] = None,
    dress: Annotated[UploadFile | None, File()] = None,
) -> JSONResponse:
    try:
        products: list[ProductReferenceInput] = []
        await _append_upload(products, top1, top1_type)
        await _append_upload(products, top2, "top")
        await _append_upload(products, bottom, "bottom")
        await _append_upload(products, dress, "dress")

        if not products:
            return JSONResponse({"error": "Upload at least one garment image."}, status_code=400)

        result = build_product_reference(products)
        image_data = _encode_png_data_url(result.image)
        return JSONResponse(
            {
                "image": image_data,
                "mode": result.mode,
                "product_count": result.product_count,
                "width": result.image.width,
                "height": result.image.height,
            },
        )
    except UnidentifiedImageError:
        return JSONResponse({"error": "One uploaded file is not a valid image."}, status_code=400)


async def _append_upload(
    products: list[ProductReferenceInput],
    upload: UploadFile | None,
    product_type: str,
) -> None:
    if upload is None or not upload.filename:
        return

    content = await upload.read()
    if not content:
        return

    image = Image.open(BytesIO(content)).convert("RGB")
    products.append(ProductReferenceInput(image=image, type=product_type))


def _encode_png_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"

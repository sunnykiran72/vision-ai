from __future__ import annotations

import http.server
import base64
import hashlib
import hmac
import json
import os
import ssl
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import certifi


ROOT = Path(__file__).resolve().parent
HTML = ROOT / "glamify_api_tester.html"
AWQ_HTML = ROOT / "minicpm_awq_tester.html"
HOST = "127.0.0.1"
PORT = 8765
TOKEN_TTL_SECONDS = 3600
TOKEN_SECRET_FILES = (
    Path.cwd() / ".env",
    ROOT.parent / ".env",
    ROOT.parent.parent / "glamify_backend" / ".env",
    ROOT.parent.parent / "glamify_backend" / "api" / ".env",
)
# Azure write creds (for the /upload helper) live in the backend repo .env.
AZURE_CONTAINER = "wardrobe-outputs"
AZURE_ENV_FILES = (
    Path.cwd() / ".env",
    ROOT.parent / ".env",
    ROOT.parent.parent / "glamify_backend" / ".env",
    ROOT.parent.parent / "glamify_backend" / "api" / ".env",
)


def _load_azure_conn() -> str:
    value = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "").strip()
    if value:
        return value
    for path in AZURE_ENV_FILES:
        found = _read_env_value(path, "AZURE_STORAGE_CONNECTION_STRING")
        if found:
            return found
    return ""


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _read_env_value(path: Path, name: str) -> str:
    if not path.exists():
        return ""
    for line in path.read_text().splitlines():
        raw = line.strip()
        if not raw or raw.startswith("#") or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        if key.strip() != name:
            continue
        return value.strip().strip('"').strip("'")
    return ""


def _load_jwt_secret() -> str:
    value = os.environ.get("JWT_ACCESS_SECRET", "").strip().strip('"').strip("'")
    if value:
        return value
    for path in TOKEN_SECRET_FILES:
        value = _read_env_value(path, "JWT_ACCESS_SECRET")
        if value:
            return value
    return ""


def _make_jwt(secret: str, user_id: str, ttl_seconds: int) -> tuple[str, int]:
    now = int(time.time())
    expires_at = now + ttl_seconds
    resolved_user_id = _resolve_user_id(user_id)
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "userId": resolved_user_id,
        "authType": "EMAIL",
        "token_id": str(uuid.uuid4()),
        "iat": now,
        "exp": expires_at,
    }
    signing_input = ".".join(
        (
            _b64url(json.dumps(header, separators=(",", ":")).encode()),
            _b64url(json.dumps(payload, separators=(",", ":")).encode()),
        )
    )
    digest = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
    return f"{signing_input}.{_b64url(digest)}", expires_at


def _resolve_user_id(raw_user_id: str) -> str:
    raw = str(raw_user_id or "").strip()
    if raw:
        try:
            return str(uuid.UUID(raw))
        except ValueError:
            pass
    return str(uuid.uuid4())


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in {"/", "/glamify_api_tester.html"}:
            self._send_bytes(HTML.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path in {"/awq", "/minicpm_awq_tester.html"}:
            self._send_bytes(AWQ_HTML.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path.startswith("/proxy"):
            self._proxy_raw("GET")
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/token":
            self._token()
            return
        if self.path.startswith("/upload"):
            self._upload()
            return
        if self.path.startswith("/proxy"):
            self._proxy_raw("POST")
            return
        self.send_error(404)

    def _upload(self) -> None:
        """Resize an uploaded image to a chosen 2:3-preserving long edge (rounded to /16,
        which the Qwen pipeline requires) and push it to the public Azure container, so the
        tryon API (URL-only) can be fed arbitrary local images. Returns the public URL + dims."""
        try:
            from io import BytesIO

            from PIL import Image
            from azure.storage.blob import BlobServiceClient, ContentSettings

            query = parse_qs(urlparse(self.path).query)
            long_edge = int(query.get("long_edge", ["0"])[0] or 0)
            name = (query.get("name", ["img"])[0] or "img").replace("/", "_")[:24]
            raw = self.rfile.read(int(self.headers.get("content-length", "0")))
            if not raw:
                raise ValueError("empty upload body")

            image = Image.open(BytesIO(raw)).convert("RGB")
            orig_w, orig_h = image.size
            if long_edge and long_edge > 0:
                if orig_w >= orig_h:
                    new_w, new_h = long_edge, max(16, round(orig_h * long_edge / orig_w))
                else:
                    new_h, new_w = long_edge, max(16, round(orig_w * long_edge / orig_h))
            else:
                new_w, new_h = orig_w, orig_h
            # Qwen/VAE needs dims divisible by 16.
            new_w = max(16, (new_w // 16) * 16)
            new_h = max(16, (new_h // 16) * 16)
            if (new_w, new_h) != (orig_w, orig_h):
                image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

            buf = BytesIO()
            image.save(buf, format="JPEG", quality=95, subsampling=0)
            data = buf.getvalue()

            conn = _load_azure_conn()
            if not conn:
                raise ValueError("AZURE_STORAGE_CONNECTION_STRING not found in env or backend .env")
            blob_name = f"tryon-experiments/{name}_{new_w}x{new_h}_{uuid.uuid4().hex}.jpg"
            client = BlobServiceClient.from_connection_string(conn).get_blob_client(
                container=AZURE_CONTAINER, blob=blob_name
            )
            client.upload_blob(
                data, overwrite=True,
                content_settings=ContentSettings(content_type="image/jpeg"),
            )
            self._send_json(200, {
                "url": client.url,
                "width": new_w, "height": new_h,
                "orig_width": orig_w, "orig_height": orig_h,
                "bytes": len(data),
            })
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def _token(self) -> None:
        try:
            body = self.rfile.read(int(self.headers.get("content-length", "0")))
            payload = json.loads(body or b"{}")
            secret = (
                str(payload.get("secret") or "").strip().strip('"').strip("'")
                or _load_jwt_secret()
            )
            if not secret:
                raise ValueError("JWT_ACCESS_SECRET is not set for the local tester.")
            user_id = _resolve_user_id(str(payload.get("user_id") or "").strip())
            ttl = int(payload.get("ttl_seconds") or TOKEN_TTL_SECONDS)
            ttl = max(60, min(ttl, 24 * 60 * 60))
            token, expires_at = _make_jwt(secret=secret, user_id=user_id, ttl_seconds=ttl)
            self._send_json(
                200,
                {
                    "token": token,
                    "user_id": user_id,
                    "expires_at": expires_at,
                    "ttl_seconds": ttl,
                },
            )
        except Exception as exc:
            self._send_json(400, {"error": str(exc)})

    def _proxy_raw(self, method: str) -> None:
        try:
            parsed_path = urlparse(self.path)
            query = parse_qs(parsed_path.query)
            url = query.get("url", [""])[0]
            if not url:
                raise ValueError("Missing proxy target URL.")
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError("Proxy URL must be http or https.")

            body = None
            if method != "GET":
                body = self.rfile.read(int(self.headers.get("content-length", "0")))

            headers = {}
            for key in ("authorization", "content-type", "accept"):
                value = self.headers.get(key)
                if value:
                    headers[key] = value
            headers.setdefault(
                "user-agent",
                (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/148.0.0.0 Safari/537.36"
                ),
            )

            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            context = ssl.create_default_context(cafile=certifi.where())
            try:
                with urllib.request.urlopen(request, timeout=600, context=context) as response:
                    content = response.read()
                    status = response.status
                    content_type = response.headers.get("content-type", "application/octet-stream")
            except urllib.error.HTTPError as exc:
                content = exc.read()
                status = exc.code
                content_type = exc.headers.get("content-type", "application/json")

            self.send_response(status)
            self.send_header("content-type", content_type)
            self.send_header("content-length", str(len(content)))
            self.send_header("cache-control", "no-store")
            self.end_headers()
            self.wfile.write(content)
        except Exception as exc:
            self._send_json(500, {"error": str(exc)})

    def _send_json(self, status: int, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.send_header("cache-control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("access-control-allow-origin", "*")
        self.send_header("access-control-allow-methods", "GET,POST,OPTIONS")
        self.send_header("access-control-allow-headers", "authorization,content-type,accept")
        self.end_headers()

    def _send_bytes(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("content-type", content_type)
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"Open http://{HOST}:{PORT}")
    server.serve_forever()

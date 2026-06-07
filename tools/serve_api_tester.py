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


ROOT = Path(__file__).resolve().parent
HTML = ROOT / "glamify_api_tester.html"
HOST = "127.0.0.1"
PORT = 8765
TOKEN_TTL_SECONDS = 3600
TOKEN_SECRET_FILES = (
    Path.cwd() / ".env",
    ROOT.parent / ".env",
)


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
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "userId": user_id,
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


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path in {"/", "/glamify_api_tester.html"}:
            self._send_bytes(HTML.read_bytes(), "text/html; charset=utf-8")
            return
        if self.path.startswith("/proxy"):
            self._proxy_raw("GET")
            return
        self.send_error(404)

    def do_POST(self) -> None:
        if self.path == "/token":
            self._token()
            return
        if self.path.startswith("/proxy"):
            self._proxy_raw("POST")
            return
        self.send_error(404)

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
            user_id = str(payload.get("user_id") or "").strip() or str(uuid.uuid4())
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

            request = urllib.request.Request(url, data=body, headers=headers, method=method)
            context = ssl.create_default_context()
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

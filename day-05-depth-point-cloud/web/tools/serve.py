"""Local preview server with the same headers as Cloudflare (_headers).

    python3 web/tools/serve.py   ->  http://localhost:8788
"""
import http.server, functools
from pathlib import Path

ROOT = Path(__file__).parents[1] / "public"


class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


http.server.ThreadingHTTPServer(("127.0.0.1", 8788), functools.partial(H, directory=ROOT)).serve_forever()

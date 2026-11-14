"""The dashboard's web server: this machine only, standard library only.

    uv run lotto ui [--port 6037] [--no-browser] [--offline]

It listens on 127.0.0.1 and nowhere else. To use it from another machine,
forward the port over SSH (``ssh -L 6037:localhost:6037 user@host``).

Two checks keep other websites open in the same browser from driving it:
every request must be addressed to this server by name (the Host header), and
every request that changes something must come from this page (the Origin
header) and carry the token the page was given when it loaded.
"""

from __future__ import annotations

import argparse
import hmac
import json
import logging
import secrets
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from lotto.core import paths
from lotto.ui import report
from lotto.ui.jobs import Busy, JobRunner
from lotto.ui.registry import Invalid
from lotto.ui.service import Service
from lotto.ui.storage import Storage, valid_id

logger = logging.getLogger("lotto.ui")

DEFAULT_PORT = 6037  # six of thirty-seven
STATIC = Path(__file__).parent / "static"
STATIC_FILES = {"app.css": "text/css; charset=utf-8", "app.js": "text/javascript; charset=utf-8"}
MAX_BODY = 64 * 1024
SECURITY_HEADERS = {
    "Cache-Control": "no-store",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": ("default-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
                                "frame-ancestors 'none'; base-uri 'none'; form-action 'none'"),
}


def _reject_constant(value: str):
    raise ValueError(f"{value} is not a JSON number")


class App:
    """What one running dashboard holds: its service, its worker and its secret token."""

    def __init__(self, service: Service):
        self.service = service
        self.jobs = JobRunner()
        self.token = secrets.token_urlsafe(24)
        self.port = 0


class Refused(Exception):
    """A request that must not be served: carries the status and a short reason."""

    def __init__(self, status: HTTPStatus, message: str, fields: dict | None = None):
        super().__init__(message)
        self.status, self.message, self.fields = status, message, fields or {}


def make_handler(app: App) -> type[BaseHTTPRequestHandler]:
    """The request handler class for one dashboard."""

    class Handler(BaseHTTPRequestHandler):
        server_version = "ILotto"

        def log_message(self, format, *args):  # keep the terminal quiet; errors still log
            logger.debug("%s %s", self.address_string(), format % args)

        # ---- plumbing ---------------------------------------------------------
        def _send(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            for name, value in SECURITY_HEADERS.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, value, status: HTTPStatus = HTTPStatus.OK) -> None:
            self._send(status, "application/json; charset=utf-8", json.dumps(value, allow_nan=False).encode())

        def _own_names(self) -> tuple[str, ...]:
            return f"127.0.0.1:{app.port}", f"localhost:{app.port}"

        def _check_host(self) -> None:
            if self.headers.get("Host") not in self._own_names():
                raise Refused(HTTPStatus.FORBIDDEN, "This dashboard only answers on localhost.")

        def _check_mutation(self) -> None:
            if self.headers.get("Origin") not in tuple(f"http://{name}" for name in self._own_names()):
                raise Refused(HTTPStatus.FORBIDDEN, "Changes are only accepted from the dashboard page.")
            if not hmac.compare_digest(self.headers.get("X-Lotto-Token", ""), app.token):
                raise Refused(HTTPStatus.FORBIDDEN, "Reload the dashboard page and try again.")

        def _body(self) -> dict:
            if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
                raise Refused(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, "JSON is required.")
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError as exc:
                raise Refused(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required.") from exc
            if not 0 <= length <= MAX_BODY:
                raise Refused(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "The request is too large.")
            try:
                value = json.loads(self.rfile.read(length) or b"{}", parse_constant=_reject_constant)
            except ValueError as exc:
                raise Refused(HTTPStatus.BAD_REQUEST, "The request is not valid JSON.") from exc
            if not isinstance(value, dict):
                raise Refused(HTTPStatus.BAD_REQUEST, "A JSON object is required.")
            return value

        def _handle(self, route) -> None:
            try:
                self._check_host()
                route()
            except Refused as exc:
                self._json({"error": {"message": exc.message, "fields": exc.fields}}, exc.status)
            except Invalid as exc:
                self._json({"error": {"message": "Some settings are not valid.", "fields": exc.fields}},
                           HTTPStatus.BAD_REQUEST)
            except Busy:
                self._json({"error": {"message": "Too many jobs are waiting. Let one finish first.", "fields": {}}},
                           HTTPStatus.CONFLICT)
            except BrokenPipeError:
                pass
            except Exception:
                logger.exception("Request failed: %s %s", self.command, self.path)
                self._json({"error": {"message": "The dashboard hit an internal error; see the terminal.",
                                      "fields": {}}}, HTTPStatus.INTERNAL_SERVER_ERROR)

        # ---- routes -----------------------------------------------------------
        def do_GET(self):
            self._handle(self._get)

        def do_POST(self):
            self._handle(self._post)

        def do_DELETE(self):
            self._handle(self._delete)

        def _get(self) -> None:
            url = urlsplit(self.path)
            path = url.path
            if path == "/":
                page = (STATIC / "index.html").read_text().replace("{{TOKEN}}", app.token)
                return self._send(HTTPStatus.OK, "text/html; charset=utf-8", page.encode())
            if path.startswith("/static/") and path[8:] in STATIC_FILES:  # a fixed list, never a path
                return self._send(HTTPStatus.OK, STATIC_FILES[path[8:]], (STATIC / path[8:]).read_bytes())
            if path == "/api/state":
                active = app.jobs.active()
                return self._json({**app.service.state(), "active_job": None if active is None else active.id})
            if path == "/api/notebook":
                return self._json(app.service.notebook())
            if path == "/api/log":
                return self._json(app.service.log())
            if path.startswith("/api/jobs/"):
                job = app.jobs.get(path[10:])
                if job is None:
                    raise Refused(HTTPStatus.NOT_FOUND, "That job is no longer known.")
                return self._json(job.view())
            if path.startswith("/report/"):
                result = app.service.storage.get_result(path[8:]) if valid_id(path[8:]) else None
                if result is None:
                    raise Refused(HTTPStatus.NOT_FOUND, "That result is no longer available.")
                content_type, body = report.render(result, parse_qs(url.query).get("format") == ["md"])
                return self._send(HTTPStatus.OK, content_type, body.encode())
            raise Refused(HTTPStatus.NOT_FOUND, "Not found.")

        def _post(self) -> None:
            self._check_mutation()
            body = self._body()
            path = urlsplit(self.path).path
            service = app.service
            work = {"/api/predict": service.predict, "/api/compare": service.compare,
                    "/api/backtest": service.backtest}.get(path)
            if work is not None:
                kind = path.rsplit("/", 1)[1]
                service.validate_request(kind, body)
                key = path + json.dumps(body, sort_keys=True) + service.archive["sha256"]
                job = app.jobs.submit(kind, key, lambda j: work(body, j))
                return self._json(job.view(), HTTPStatus.ACCEPTED)
            if path == "/api/refresh":
                if service.offline:
                    raise Refused(HTTPStatus.CONFLICT, "The dashboard was started with --offline.")
                return self._json(app.jobs.submit("refresh", "refresh", service.refresh).view(), HTTPStatus.ACCEPTED)
            if path == "/api/log":
                return self._json(service.save_to_log(body.get("result_id")))
            raise Refused(HTTPStatus.NOT_FOUND, "Not found.")

        def _delete(self) -> None:
            self._check_mutation()
            path = urlsplit(self.path).path
            job = app.jobs.cancel(path[10:]) if path.startswith("/api/jobs/") else None
            if job is None:
                raise Refused(HTTPStatus.NOT_FOUND, "That job is no longer known.")
            self._json(job.view())

    return Handler


def create_server(service: Service, port: int = DEFAULT_PORT) -> tuple[ThreadingHTTPServer, App]:
    """A server bound to 127.0.0.1. Port 0 picks a free port (used by the tests)."""
    app = App(service)
    class DashboardServer(ThreadingHTTPServer):
        def server_close(self):
            app.jobs.close()
            super().server_close()

    try:
        server = DashboardServer(("127.0.0.1", port), make_handler(app))
    except BaseException:
        app.jobs.close()
        raise
    server.daemon_threads = True
    app.port = server.server_address[1]
    return server, app


def main(argv: list[str] | None = None) -> int:
    """Start the dashboard and serve until Ctrl-C."""
    parser = argparse.ArgumentParser(prog="lotto ui", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true", help="do not open the browser")
    parser.add_argument("--offline", action="store_true", help="never contact pais.co.il")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")

    service = Service(Storage(paths.REPO_ROOT / "results/ui"), offline=args.offline)
    try:
        server, app = create_server(service, args.port)
    except OSError as exc:
        parser.error(f"cannot listen on port {args.port} ({exc.strerror}); try --port with another number")
    url = f"http://localhost:{app.port}"
    print(f"\n  ILotto dashboard: {url}\n  {service.archive['draws']:,} draws through "
          f"{service.archive['last_date']}. Press Ctrl-C to stop.\n")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0

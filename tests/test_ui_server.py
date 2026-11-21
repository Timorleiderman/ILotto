"""The dashboard's web server: routing, protection, jobs and reports, on a free local port."""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from lotto.ui.server import STATIC, STATIC_FILES, create_server
from lotto.ui.service import Service
from lotto.ui.storage import Storage
from tests import PRIZES_SNAPSHOT, SNAPSHOT


@pytest.fixture(scope="module")
def dashboard(tmp_path_factory):
    root = tmp_path_factory.mktemp("ui")
    server, app = create_server(Service(Storage(root), raw_csv=SNAPSHOT, offline=True,
                                         prizes_csv=PRIZES_SNAPSHOT), port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield app, root
    server.shutdown()
    server.server_close()


def call(app, method, path, body=None, *, host=None, origin="same", token="right"):
    """One request; returns (status, parsed body or text)."""
    base = f"127.0.0.1:{app.port}"
    request = urllib.request.Request(f"http://{base}{path}", method=method,
                                     data=None if body is None else json.dumps(body).encode())
    request.add_header("Host", host or base)
    if method != "GET":
        if origin:
            request.add_header("Origin", f"http://{base}" if origin == "same" else origin)
        request.add_header("X-Lotto-Token", app.token if token == "right" else token)
        request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            status, raw, kind = response.status, response.read(), response.headers["Content-Type"]
    except urllib.error.HTTPError as error:
        status, raw, kind = error.code, error.read(), error.headers["Content-Type"]
    return status, json.loads(raw) if kind.startswith("application/json") else raw.decode()


def finish(app, job):
    for _ in range(600):
        if job["state"] not in ("queued", "running", "cancel_requested"):
            return job
        time.sleep(0.05)
        job = call(app, "GET", f"/api/jobs/{job['id']}")[1]
    raise AssertionError("job did not finish")


def test_page_and_every_file_it_names_are_served(dashboard):
    app, _ = dashboard
    status, page = call(app, "GET", "/")
    assert status == 200 and app.token in page and "{{TOKEN}}" not in page
    for name in STATIC_FILES:
        assert f"/static/{name}" in page and (STATIC / name).is_file()
        assert call(app, "GET", f"/static/{name}")[0] == 200


@pytest.mark.parametrize("path", ["/static/../server.py", "/static/%2e%2e/server.py", "/static/index.html",
                                  "/report/../../etc/passwd", "/report/zzzzzzzzzzzzzzzz", "/api/jobs/none"])
def test_unknown_and_escaping_paths_are_refused(dashboard, path):
    assert call(dashboard[0], "GET", path)[0] == 404


def test_state_lists_models_and_never_claims_a_check_that_did_not_happen(dashboard):
    status, state = call(dashboard[0], "GET", "/api/state")
    assert status == 200 and state["archive"]["draws"] == 1661
    assert state["refresh"] == {"state": "not_checked", "at": None, "offline": True}
    assert len(state["presets"]) == 19 and {"hot", "gru", "seed_world"} <= {m["id"] for m in state["models"]}


def test_requests_from_elsewhere_are_refused(dashboard):
    app, _ = dashboard
    assert call(app, "GET", "/api/state", host="evil.example")[0] == 403            # DNS rebinding
    body = {"model": "hot"}
    assert call(app, "POST", "/api/predict", body, origin="http://evil.example")[0] == 403
    assert call(app, "POST", "/api/predict", body, origin=None)[0] == 403           # no Origin at all
    assert call(app, "POST", "/api/predict", body, token="wrong")[0] == 403
    assert call(app, "DELETE", "/api/jobs/abc", token="wrong")[0] == 403
    assert call(app, "POST", "/api/refresh", {})[0] == 409                           # started offline


def test_predict_runs_as_a_job_and_reports_it(dashboard):
    app, root = dashboard
    status, job = call(app, "POST", "/api/predict", {"model": "hot", "params": {"window": 100}, "rows": 10})
    assert status == 202
    result = finish(app, job)["result"]
    assert result["preset"] == "hot_100" and len(result["form"]) == 10
    status, page = call(app, "GET", f"/report/{result['id']}")
    assert status == 200 and "Hot numbers" in page and "38.60%" in page
    status, markdown = call(app, "GET", f"/report/{result['id']}?format=md")
    assert status == 200 and markdown.startswith("# Hot numbers")
    written = {p.relative_to(root).parts[0] for p in root.rglob("*") if p.is_file()}
    assert written <= {"results", "log", "notebooks"}


def test_invalid_settings_come_back_by_field(dashboard):
    app, _ = dashboard
    status, body = call(app, "POST", "/api/predict", {"model": "hot", "params": {"window": 3}})
    assert status == 400 and "window" in body["error"]["fields"]
    status, body = call(app, "POST", "/api/log", {"result_id": "0000000000000000"})
    assert status == 400 and "result" in body["error"]["fields"]


def test_backtest_fills_the_notebook_and_log_saves_once(dashboard):
    app, _ = dashboard
    done = finish(app, call(app, "POST", "/api/backtest", {"model": "overdue", "n_test": 100})[1])
    assert done["state"] == "succeeded"
    notebook = call(app, "GET", "/api/notebook")[1]
    assert [a["state"] for a in notebook["attempts"]] == ["complete"] and notebook["attempts"][0]["draws"] == 100
    result = finish(app, call(app, "POST", "/api/predict", {"model": "overdue", "rows": 2})[1])["result"]
    first = call(app, "POST", "/api/log", {"result_id": result["id"]})
    assert first[0] == 200 and call(app, "POST", "/api/log", {"result_id": result["id"]})[1] == first[1]
    log = call(app, "GET", "/api/log")[1]
    assert log["summary"]["saved"] == 1 and log["entries"][0]["state"] == "waiting"


def test_a_queued_job_can_be_cancelled(dashboard):
    app, _ = dashboard
    entered, release = threading.Event(), threading.Event()

    def hold(job):
        entered.set()
        assert release.wait(5)
        job.step("Released")
        return {}

    slow = app.jobs.submit("test", "hold", hold)
    try:
        assert entered.wait(5)
        queued = call(app, "POST", "/api/predict", {"model": "ewma", "params": {"half_life": 77}})[1]
        status, cancelled = call(app, "DELETE", f"/api/jobs/{queued['id']}")
        assert status == 200 and cancelled["state"] == "cancelled"
        assert call(app, "DELETE", f"/api/jobs/{slow.id}")[1]["state"] == "cancel_requested"
    finally:
        release.set()
    assert finish(app, slow.view())["state"] == "cancelled"


@pytest.mark.parametrize("path,body,status", [
    ("/api/predict", {"model": "hot", "params": {"window": float("inf")}}, 400),
    ("/api/predict", {"model": "ensemble", "params": {"members": [{}]}}, 400),
    ("/api/predict", {"model": "seed_world", "date": []}, 400),
    ("/api/predict", {"model": "hot", "row": 14}, 400),
    ("/api/backtest", {"model": "date_cb", "params": {"ticket": "sampled"}}, 400),
    ("/api/compare", {"unexpected": True}, 400),
    ("/api/predict", {"payload": "x" * 65536}, 413),
])
def test_malformed_requests_are_rejected_before_queueing(dashboard, path, body, status):
    assert call(dashboard[0], "POST", path, body)[0] == status


def test_value_is_a_read_with_validated_query(dashboard):
    app, _ = dashboard
    status, value = call(app, "GET", "/api/value?rows=3&kind=double&tax=1")
    assert status == 200 and value["plan"]["cost"] == 18 and not value["plan"]["positive"]
    status, error = call(app, "GET", "/api/value?rows=-1")
    assert status == 400 and "rows" in error["error"]["fields"]
    assert call(app, "GET", "/api/value", host="evil.example:80")[0] == 403

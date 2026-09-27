"""Python served-path call graphs from the inert L2 analyzer."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from ditto_screener.l2_review import _graph_covers_l1_slice
from ditto_screener.policy import SourceReviewObservation

ANALYZER = Path(__file__).resolve().parents[1] / "tools" / "l2_analyzer.py"


def _graph(root: Path, entry: str = "main") -> dict[str, object]:
    result = subprocess.run(
        [sys.executable, str(ANALYZER), "call_graph"],
        input=json.dumps({"entry": entry}),
        env={**os.environ, "L2_ANALYZER_ROOT": str(root)},
        capture_output=True,
        text=True,
        check=True,
    )
    value = json.loads(result.stdout)
    assert isinstance(value, dict)
    return value


def _write(root: Path, files: dict[str, str]) -> Path:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    return root


def _cited(path: str, line: int) -> SourceReviewObservation:
    return SourceReviewObservation(
        ok=True,
        risk_level="medium",
        finding_digest="a" * 64,
        categories=("benchmark_emulation",),
        finding={
            "risk_level": "medium",
            "confidence": 0.8,
            "categories": ["benchmark_emulation"],
            "evidence": [
                {"path": path, "line": line, "category": "benchmark_emulation"}
            ],
            "summary": "bounded test routing lead",
        },
        clearance_certified=False,
    )


def _ids(graph: dict[str, object]) -> list[str]:
    nodes = graph["nodes"]
    assert isinstance(nodes, list)
    return [str(node["id"]) for node in nodes]


FASTAPI = {
    "app/__init__.py": "",
    "app/main.py": """\
from fastapi import FastAPI

from app import memory
from app.model import ask as ask_model

app = FastAPI()


@app.post("/run")
async def run(body: dict) -> dict:
    facts = memory.recall(body["user"])
    return {"answer": ask_model(body["q"], facts)}


@app.post("/seed")
def seed(body: dict) -> dict:
    memory.store(body["user"], body["facts"])
    return {}
""",
    "app/memory.py": """\
_DB = {}


def recall(user):
    return _DB.get(user, [])


def store(user, facts):
    _DB[user] = facts
""",
    "app/model.py": """\
import httpx


def ask(q, facts):
    return httpx.post("http://gw", json={"q": q, "facts": facts}).json()
""",
}


def test_route_handlers_root_a_resolved_python_served_path(tmp_path: Path) -> None:
    graph = _graph(_write(tmp_path, FASTAPI))

    assert graph["language"] == "python"
    assert graph["unresolved"] is False
    assert graph["entry_ambiguous"] is False
    assert graph["truncated"] is False
    assert graph["served_roots"] == ["app/main.py:10:run", "app/main.py:16:seed"]
    assert set(_ids(graph)) == {
        "app/main.py:10:run",
        "app/main.py:16:seed",
        "app/memory.py:4:recall",
        "app/memory.py:8:store",
        "app/model.py:4:ask",
    }
    # Every call the /run handler makes resolves across modules, so an L1
    # citation inside it is covered, which is what a supported CLEAR needs.
    assert _graph_covers_l1_slice(graph, _cited("app/main.py", 11))
    # Library calls stay unresolved, so a citation there is never covered.
    assert _graph_covers_l1_slice(graph, _cited("app/model.py", 5)) is False


def test_dynamic_dispatch_stays_unresolved_non_evidence(tmp_path: Path) -> None:
    root = _write(
        tmp_path,
        {
            "app.py": """\
from flask import Flask

app = Flask(__name__)
HANDLERS = {"a": lambda: 1}


def run():
    return HANDLERS["a"]() + getattr(app, "x")()


app.add_url_rule("/run", view_func=run, methods=["POST"])
"""
        },
    )

    graph = _graph(root)

    assert graph["served_roots"] == ["app.py:7:run"]
    assert graph["unresolved_count"] == 3
    assert _graph_covers_l1_slice(graph, _cited("app.py", 8)) is False


def test_stdlib_handler_methods_and_self_calls_resolve(tmp_path: Path) -> None:
    root = _write(
        tmp_path,
        {
            "server.py": """\
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        self.reply(answer())

    def reply(self, value):
        return value


def answer():
    return b"ok"


def main():
    HTTPServer(("", 8000), Handler).serve_forever()


if __name__ == "__main__":
    main()
"""
        },
    )

    graph = _graph(root)

    assert graph["served_roots"] == ["server.py:5:do_POST"]
    assert set(_ids(graph)) == {
        "server.py:16:main",
        "server.py:5:do_POST",
        "server.py:8:reply",
        "server.py:12:answer",
    }
    assert _graph_covers_l1_slice(graph, _cited("server.py", 6))


def test_same_named_modules_make_the_call_ambiguous(tmp_path: Path) -> None:
    root = _write(
        tmp_path,
        {
            "a/util.py": "def helper():\n    return 1\n",
            "b/util.py": "def helper():\n    return 2\n",
            "main.py": """\
from util import helper


def main():
    return helper()
""",
        },
    )

    graph = _graph(root)

    assert graph["ambiguous_count"] == 1
    assert _graph_covers_l1_slice(graph, _cited("main.py", 5)) is False


def test_unparseable_python_marks_the_graph_incomplete(tmp_path: Path) -> None:
    root = _write(
        tmp_path,
        {**FASTAPI, "app/broken.py": "def oops(:\n"},
    )

    graph = _graph(root)

    assert graph["analysis_truncated"] is True
    assert _graph_covers_l1_slice(graph, _cited("app/main.py", 11)) is False


def test_rust_workspaces_keep_the_rust_graph(tmp_path: Path) -> None:
    root = _write(
        tmp_path,
        {**FASTAPI, "src/main.rs": "fn main() {\n    helper();\n}\nfn helper() {}\n"},
    )

    graph = _graph(root)

    assert "language" not in graph
    assert _ids(graph) == ["src/main.rs:1:main", "src/main.rs:4:helper"]


def test_named_python_entries_do_not_add_route_roots(tmp_path: Path) -> None:
    graph = _graph(_write(tmp_path, FASTAPI), entry="recall")

    assert graph["served_roots"] == []
    assert _ids(graph) == ["app/memory.py:4:recall"]

import asyncio
import importlib
import json
import sys
from pathlib import Path

import httpx
import pytest


MCP_ROOT = Path(__file__).resolve().parents[4] / "cognee-mcp"
if str(MCP_ROOT) not in sys.path:
    sys.path.insert(0, str(MCP_ROOT))

CogneeClient = importlib.import_module("src.cognee_client").CogneeClient
server_utils = importlib.import_module("src.server_utils")
format_recall_results = server_utils.format_recall_results
format_search_results = server_utils.format_search_results
normalize_delete_mode = server_utils.normalize_delete_mode
parse_cognify_data = server_utils.parse_cognify_data
validate_cognify_file_paths = server_utils.validate_cognify_file_paths
validate_top_k = server_utils.validate_top_k


def test_parse_cognify_data_accepts_plain_text():
    parsed = parse_cognify_data("plain text")

    assert parsed.items == ["plain text"]
    assert parsed.is_batch is False


def test_parse_cognify_data_accepts_json_batch():
    parsed = parse_cognify_data(json.dumps(["/tmp/a.txt", "inline memory"]))

    assert parsed.items == ["/tmp/a.txt", "inline memory"]
    assert parsed.is_batch is True


@pytest.mark.parametrize("payload", ["[]", "[1]", '[""]', "[1, 2]"])
def test_parse_cognify_data_rejects_invalid_batches(payload):
    with pytest.raises(ValueError):
        parse_cognify_data(payload)


def test_parse_cognify_data_preserves_plain_text_starting_with_bracket():
    parsed = parse_cognify_data("[note: inline memory")

    assert parsed.items == ["[note: inline memory"]
    assert parsed.is_batch is False


def test_validate_cognify_file_paths_reports_batch_index():
    error = validate_cognify_file_paths(
        ["/missing/file.txt"],
        path_exists=lambda _: False,
    )

    assert "File not found: /missing/file.txt" in error

    batch_error = validate_cognify_file_paths(
        ["inline text", "/missing/file.txt"],
        path_exists=lambda _: False,
    )

    assert "Invalid batch item at index 1" in batch_error
    assert "File not found: /missing/file.txt" in batch_error


def test_validate_top_k_and_delete_mode():
    assert validate_top_k("3") == 3
    assert normalize_delete_mode(" HARD ") == "hard"

    with pytest.raises(ValueError):
        validate_top_k(0)
    with pytest.raises(ValueError):
        validate_top_k(101)
    with pytest.raises(ValueError):
        normalize_delete_mode("unsafe")


def test_format_search_results_handles_envelope_and_completion_rows():
    rendered = format_search_results(
        {
            "query": "what matters?",
            "results": [
                {"dataset_name": "alpha", "search_result": ["first answer", "second answer"]},
                {"dataset_name": "beta", "text": "third answer"},
            ],
        },
        "GRAPH_COMPLETION",
    )

    assert "[alpha] first answer" in rendered
    assert "[alpha] second answer" in rendered
    assert "[beta] third answer" in rendered


def test_format_recall_results_handles_normalized_rows():
    rendered = format_recall_results(
        {
            "results": [
                {"source": "session", "text": "cached answer"},
                {"_source": "graph", "answer": "graph answer"},
            ]
        }
    )

    assert "[session] cached answer" in rendered
    assert "[graph] graph answer" in rendered


@pytest.mark.asyncio
async def test_cognee_client_api_remember_sends_session_id():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "ok"})

    client = CogneeClient(api_url="http://cognee.local", api_token="token")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.remember("hello", dataset_name="ds", session_id="session-1")
    finally:
        await client.close()

    assert requests[0].url.path == "/api/v1/remember"
    body = requests[0].content.decode()
    assert 'name="session_id"' in body
    assert "session-1" in body


@pytest.mark.asyncio
async def test_cognee_client_api_recall_sends_session_id_and_null_search_type():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json=[])

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.recall("hello", session_id="session-1", top_k=5)
    finally:
        await client.close()

    payload = json.loads(requests[0].content.decode())
    assert requests[0].url.path == "/api/v1/recall"
    assert payload["session_id"] == "session-1"
    assert payload["search_type"] is None
    assert payload["top_k"] == 5


@pytest.mark.asyncio
async def test_cognee_client_api_delete_uses_mode_aware_endpoint():
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"status": "success"})

    client = CogneeClient(api_url="http://cognee.local")
    await client.client.aclose()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    try:
        await client.delete(
            data_id="00000000-0000-0000-0000-000000000001",
            dataset_id="00000000-0000-0000-0000-000000000002",
            mode="hard",
        )
    finally:
        await client.close()

    assert requests[0].method == "DELETE"
    assert requests[0].url.path == "/api/v1/delete"
    assert requests[0].url.params["data_id"] == "00000000-0000-0000-0000-000000000001"
    assert requests[0].url.params["dataset_id"] == "00000000-0000-0000-0000-000000000002"
    assert requests[0].url.params["mode"] == "hard"


@pytest.mark.asyncio
async def test_cognify_tool_batches_add_calls(monkeypatch, tmp_path):
    import src.server as server

    data_file = tmp_path / "memory.txt"
    data_file.write_text("memory", encoding="utf-8")

    class FakeClient:
        use_api = False

        def __init__(self):
            self.added = []
            self.cognified = None

        async def add(self, data, dataset_name="main_dataset"):
            self.added.append((data, dataset_name))

        async def cognify(self, datasets=None, custom_prompt=None, graph_model=None):
            self.cognified = {
                "datasets": datasets,
                "custom_prompt": custom_prompt,
                "graph_model": graph_model,
            }

    fake_client = FakeClient()
    created_tasks = []
    original_create_task = asyncio.create_task

    def capture_task(coro):
        task = original_create_task(coro)
        created_tasks.append(task)
        return task

    monkeypatch.setattr(server, "cognee_client", fake_client)
    monkeypatch.setattr(server.asyncio, "create_task", capture_task)

    result = await server.cognify(
        json.dumps([str(data_file), "inline memory"]),
        dataset_name="batch_ds",
        custom_prompt="extract carefully",
    )

    assert "Queued 2 item(s)" in result[0].text
    await created_tasks[0]
    assert fake_client.added == [(str(data_file), "batch_ds"), ("inline memory", "batch_ds")]
    assert fake_client.cognified == {
        "datasets": ["batch_ds"],
        "custom_prompt": "extract carefully",
        "graph_model": None,
    }

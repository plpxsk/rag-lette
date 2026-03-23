from __future__ import annotations

from types import SimpleNamespace

from rag.gemini import GeminiFileApiClient


class _DummyFiles:
    def __init__(self, states: list[str]) -> None:
        self._states = iter(states)

    def get(self, *, name: str) -> SimpleNamespace:
        return SimpleNamespace(name=name, state=next(self._states))


class _DummyClient:
    def __init__(self, states: list[str]) -> None:
        self.files = _DummyFiles(states)


class _ServerError(Exception):
    def __init__(self, message: str = "server error") -> None:
        super().__init__(message)
        self.status_code = 500


class _Importer:
    def __init__(self, responses: list[object]) -> None:
        self._responses = iter(responses)
        self.calls = 0

    def import_file(self, *, file_search_store_name: str, file_name: str) -> object:
        self.calls += 1
        response = next(self._responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_wait_for_file_processing_accepts_prefixed_active_state() -> None:
    client = GeminiFileApiClient.__new__(GeminiFileApiClient)
    client.client = _DummyClient(["FileState.ACTIVE"])

    client._wait_for_file_processing("files/test", timeout_seconds=1.0)


def test_wait_for_file_processing_transitions_from_processing_to_active(monkeypatch) -> None:
    client = GeminiFileApiClient.__new__(GeminiFileApiClient)
    client.client = _DummyClient(["FileState.PROCESSING", "FileState.ACTIVE"])
    monkeypatch.setattr("rag.gemini.time.sleep", lambda _seconds: None)

    client._wait_for_file_processing("files/test", timeout_seconds=1.0)


def test_import_file_to_search_store_retries_server_error(monkeypatch) -> None:
    importer = _Importer([_ServerError("boom"), SimpleNamespace(done=False),])
    client = GeminiFileApiClient.__new__(GeminiFileApiClient)
    client.client = SimpleNamespace(file_search_stores=importer)
    monkeypatch.setattr("rag.gemini.time.sleep", lambda _seconds: None)

    op = client._import_file_to_search_store("fileSearchStores/test", "files/abc", "doc.docx")

    assert op.done is False
    assert importer.calls == 2


def test_import_file_to_search_store_raises_non_retryable_error() -> None:
    importer = _Importer([ValueError("bad request")])
    client = GeminiFileApiClient.__new__(GeminiFileApiClient)
    client.client = SimpleNamespace(file_search_stores=importer)

    try:
        client._import_file_to_search_store("fileSearchStores/test", "files/abc", "doc.docx")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "Gemini File Search import failed for doc.docx" in str(exc)
    assert importer.calls == 1

from __future__ import annotations

import sys
import types

import pytest

from rag.db import QueryChunk, WeaviateAdapter, _weaviate_collection_name, _weaviate_parse_uri, get_db_adapter


class _FakeFilter:
    def __init__(self, prop: str) -> None:
        self.prop = prop

    @classmethod
    def by_property(cls, prop: str) -> "_FakeFilter":
        return cls(prop)

    def equal(self, value: str) -> tuple[str, str, str]:
        return ("eq", self.prop, value)


class _FakeBatchContext:
    def __init__(self, collection: "_FakeCollection") -> None:
        self.collection = collection

    def __enter__(self) -> "_FakeBatchContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def add_object(self, *, properties: dict, vector: list[float]) -> None:
        self.collection.objects.append({"properties": properties, "vector": vector})


class _FakeBatchManager:
    def __init__(self, collection: "_FakeCollection") -> None:
        self.collection = collection
        self.failed_objects: list[object] = []

    def fixed_size(self, batch_size: int) -> _FakeBatchContext:
        return _FakeBatchContext(self.collection)


class _FakeQueryOps:
    def __init__(self, collection: "_FakeCollection") -> None:
        self.collection = collection

    def near_vector(self, *, near_vector: list[float], limit: int, return_properties: list[str]):
        objects = [
            types.SimpleNamespace(properties=obj["properties"])
            for obj in self.collection.objects[:limit]
        ]
        return types.SimpleNamespace(objects=objects)

    def fetch_objects(self, *, return_properties: list[str], limit: int, filters=None):
        objects = self.collection.objects
        if filters is not None:
            _, prop, value = filters
            objects = [obj for obj in objects if obj["properties"].get(prop) == value]
        return types.SimpleNamespace(
            objects=[types.SimpleNamespace(properties=obj["properties"]) for obj in objects[:limit]]
        )


class _FakeAggregateOps:
    def __init__(self, collection: "_FakeCollection") -> None:
        self.collection = collection

    def over_all(self, total_count: bool = False):
        return types.SimpleNamespace(total_count=len(self.collection.objects))


class _FakeDataOps:
    def __init__(self, collection: "_FakeCollection") -> None:
        self.collection = collection

    def delete_many(self, *, where) -> None:
        _, prop, value = where
        self.collection.objects = [
            obj for obj in self.collection.objects if obj["properties"].get(prop) != value
        ]


class _FakeCollection:
    def __init__(self, name: str) -> None:
        self.name = name
        self.objects: list[dict] = []
        self.batch = _FakeBatchManager(self)
        self.query = _FakeQueryOps(self)
        self.aggregate = _FakeAggregateOps(self)
        self.data = _FakeDataOps(self)


class _FakeCollectionsManager:
    def __init__(self) -> None:
        self.collections: dict[str, _FakeCollection] = {}

    def exists(self, name: str) -> bool:
        return name in self.collections

    def create(self, name: str, vector_config=None, properties=None) -> _FakeCollection:
        collection = _FakeCollection(name)
        self.collections[name] = collection
        return collection

    def use(self, name: str) -> _FakeCollection:
        return self.collections[name]


class _FakeClient:
    def __init__(self) -> None:
        self.collections = _FakeCollectionsManager()
        self.closed = False

    def is_ready(self) -> bool:
        return True

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_weaviate(monkeypatch):
    clients: list[_FakeClient] = []

    def _client_factory(*args, **kwargs) -> _FakeClient:
        client = _FakeClient()
        clients.append(client)
        return client

    weaviate_module = types.ModuleType("weaviate")
    weaviate_module.connect_to_local = _client_factory
    weaviate_module.connect_to_custom = _client_factory
    weaviate_module.connect_to_weaviate_cloud = _client_factory

    classes_module = types.ModuleType("weaviate.classes")
    classes_module.config = types.SimpleNamespace(
        Configure=types.SimpleNamespace(
            Vectors=types.SimpleNamespace(self_provided=lambda: "self-provided")
        ),
        Property=lambda **kwargs: kwargs,
        DataType=types.SimpleNamespace(TEXT="text", INT="int"),
    )
    classes_module.query = types.SimpleNamespace(Filter=_FakeFilter)

    init_module = types.ModuleType("weaviate.classes.init")
    init_module.Auth = types.SimpleNamespace(api_key=lambda value: ("api_key", value))

    monkeypatch.setitem(sys.modules, "weaviate", weaviate_module)
    monkeypatch.setitem(sys.modules, "weaviate.classes", classes_module)
    monkeypatch.setitem(sys.modules, "weaviate.classes.init", init_module)

    return clients


def test_weaviate_parse_uri_local() -> None:
    parsed = _weaviate_parse_uri("weaviate://localhost:8080")
    assert parsed == {"mode": "local", "host": "localhost", "port": 8080, "grpc_port": 50051}


def test_weaviate_collection_name_is_sanitized() -> None:
    assert _weaviate_collection_name("embeddings") == "Embeddings"
    assert _weaviate_collection_name("my-docs.v1") == "My_docs_v1"


def test_weaviate_adapter_round_trip(fake_weaviate) -> None:
    adapter = WeaviateAdapter("weaviate://localhost:8080", "embeddings")
    adapter.setup(embedding_dim=2)
    adapter.record_embedding_config(
        embed_provider="openai",
        embed_model="text-embedding-3-small",
        embedding_dim=2,
    )
    adapter.add(
        [
            {"text": "alpha", "source": "a.txt", "vector": [0.1, 0.2]},
            {"text": "beta", "source": "b.txt", "vector": [0.3, 0.4]},
        ]
    )

    assert adapter.exists()
    assert adapter.has_source("a.txt")
    assert adapter.list_sources() == ["a.txt", "b.txt"]
    assert adapter.query(query_vector=[0.5, 0.6], k=1) == ["alpha"]
    assert adapter.query_chunks(query_vector=[0.5, 0.6], k=2) == [
        QueryChunk(text="alpha", source="a.txt"),
        QueryChunk(text="beta", source="b.txt"),
    ]
    assert adapter.info()["rows"] == 2

    adapter.delete_source("a.txt")
    assert adapter.list_sources() == ["b.txt"]
    adapter.validate_embedding_config(
        embed_provider="openai",
        embed_model="text-embedding-3-small",
    )

    adapter.close()
    assert fake_weaviate[0].closed


def test_weaviate_adapter_routing(fake_weaviate) -> None:
    adapter = get_db_adapter("weaviate://localhost:8080", "embeddings")
    assert isinstance(adapter, WeaviateAdapter)


def test_weaviate_adapter_rejects_mismatched_embedding_config(fake_weaviate) -> None:
    adapter = WeaviateAdapter("weaviate://localhost:8080", "embeddings")
    adapter.setup(embedding_dim=2)
    adapter.record_embedding_config(
        embed_provider="openai",
        embed_model="text-embedding-3-small",
        embedding_dim=2,
    )

    with pytest.raises(RuntimeError, match="Stored embedding config"):
        adapter.validate_embedding_config(
            embed_provider="mistral",
            embed_model="mistral-embed",
        )

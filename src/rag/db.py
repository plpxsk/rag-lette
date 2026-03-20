"""
Database adapter interface and placeholder implementations.

To add a new backend:
  1. Subclass DbAdapter and implement all abstract methods.
  2. Add a URI-prefix match in get_db_adapter().
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence
from urllib.parse import parse_qs, urlparse

if TYPE_CHECKING:
    from psycopg2.extensions import connection as PgConnection


class DbAdapter(ABC):

    @abstractmethod
    def setup(self, *, embedding_dim: int) -> None:
        """Create table/index if it doesn't exist."""

    @abstractmethod
    def add(self, rows: Sequence[dict[str, Any]]) -> None:
        """Insert embedding rows: [{"text": str, "embedding": list[float], ...}]"""

    @abstractmethod
    def query(self, *, query_vector: Sequence[float], k: int) -> list[str]:
        """Return the top-k text chunks nearest to query_vector."""

    @abstractmethod
    def info(self) -> dict[str, Any]:
        """Return metadata about this store (backend, table, row count, etc.)."""

    @abstractmethod
    def exists(self) -> bool:
        """Return True if the database/table already exists and has data."""

    @abstractmethod
    def has_source(self, source: str) -> bool:
        """Return True if any rows with the given source filename exist."""

    @abstractmethod
    def delete_source(self, source: str) -> None:
        """Delete all rows with the given source filename."""

    @abstractmethod
    def list_sources(self) -> list[str]:
        """Return a sorted list of distinct source filenames stored in the DB."""

    def preflight(self) -> None:
        """Validate connectivity and write permissions before expensive work.

        Raises RuntimeError with a descriptive message if the check fails.
        The default implementation is a no-op; backends that need validation
        (e.g. Postgres) override this.
        """


class LanceDbAdapter(DbAdapter):
    """Local or S3-backed LanceDB.  URI: ./db  |  lancedb://./db  |  s3://bucket/path"""

    def __init__(self, uri: str, table_name: str) -> None:
        self.uri = uri
        self.table_name = table_name
        self._db: Any = None

    def _connect(self) -> Any:
        if self._db is None:
            import lancedb
            self._db = lancedb.connect(self.uri)
        return self._db

    def _table_names(self) -> set[str]:
        db = self._connect()
        if hasattr(db, "list_tables"):
            table_names: set[str] = set()
            listing = db.list_tables()
            entries = getattr(listing, "tables", listing)
            for entry in entries:
                if isinstance(entry, str):
                    table_names.add(entry)
                elif isinstance(entry, (tuple, list)) and entry and isinstance(entry[0], str):
                    table_names.add(entry[0])
                elif name := getattr(entry, "name", None):
                    table_names.add(name)
            return table_names
        return set(db.table_names())

    def setup(self, *, embedding_dim: int) -> None:
        self._connect()

    def add(self, rows: Sequence[dict[str, Any]]) -> None:
        db = self._connect()
        if self.table_name in self._table_names():
            db.open_table(self.table_name).add(list(rows))
        else:
            db.create_table(self.table_name, data=list(rows))

    def query(self, *, query_vector: Sequence[float], k: int) -> list[str]:
        db = self._connect()
        table = db.open_table(self.table_name)
        results = table.search(list(query_vector)).limit(k).to_list()
        return [r["text"] for r in results]

    def info(self) -> dict[str, Any]:
        db = self._connect()
        rows = db.open_table(self.table_name).count_rows() if self.exists() else None
        return {"backend": "lancedb", "uri": self.uri, "table": self.table_name, "rows": rows}

    def exists(self) -> bool:
        return self.table_name in self._table_names()

    def has_source(self, source: str) -> bool:
        if not self.exists():
            return False
        db = self._connect()
        table = db.open_table(self.table_name)
        safe = source.replace("'", "''")
        results = table.search().where(f"source = '{safe}'").limit(1).to_list()
        return bool(results)

    def delete_source(self, source: str) -> None:
        if not self.exists():
            return
        db = self._connect()
        table = db.open_table(self.table_name)
        safe = source.replace("'", "''")
        table.delete(f"source = '{safe}'")

    def list_sources(self) -> list[str]:
        if not self.exists():
            return []
        db = self._connect()
        table = db.open_table(self.table_name)
        rows = table.search().select(["source"]).to_list()
        return sorted({r["source"] for r in rows if r.get("source")})


def _pg_parse_uri(uri: str) -> dict[str, Any]:
    """Parse a postgres[ql]:// URI into connection kwargs + admin_db."""
    parsed = urlparse(uri)
    query_params = parse_qs(parsed.query)
    return {
        "host": parsed.hostname,
        "port": parsed.port or 5432,
        "user": parsed.username,
        "password": parsed.password,
        "dbname": parsed.path.lstrip("/") or "postgres",
        "admin_db": query_params.get("admin_db", ["postgres"])[0],
    }


def _pg_content_hash(source: str | None, text: str) -> str:
    h = hashlib.sha256()
    h.update((source or "").encode())
    h.update(b"\0")
    h.update(text.encode())
    return h.hexdigest()


class PostgresAdapter(DbAdapter):
    """pgvector-backed Postgres.

    URI: postgres://user:pass@host:5432/dbname
    Optional query param: ?admin_db=postgres  (DB used to CREATE DATABASE)

    Requires extras: pip install 'rag[postgres]'  (psycopg2-binary + pgvector)
    """

    def __init__(self, uri: str, table_name: str) -> None:
        self.uri = uri
        self.table_name = table_name
        self._params = _pg_parse_uri(uri)
        self._embedding_dim: int | None = None

    def _connect(self, *, dbname: str | None = None) -> PgConnection:
        import psycopg2
        params = self._params
        return psycopg2.connect(
            host=params["host"],
            port=params["port"],
            user=params["user"],
            password=params["password"],
            dbname=dbname or params["dbname"],
        )

    def _ensure_db(self) -> None:
        from psycopg2 import sql as pgsql
        dbname = self._params["dbname"]
        admin = self._params["admin_db"]
        conn = self._connect(dbname=admin)
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
                if not cur.fetchone():
                    cur.execute(pgsql.SQL("CREATE DATABASE {}").format(pgsql.Identifier(dbname)))
        finally:
            conn.close()

    def preflight(self) -> None:
        """Validate Postgres connectivity and write access before ingestion starts.

        Raises RuntimeError with a human-readable message on the first failure so
        users aren't left waiting through chunking/embedding only to hit a DB error.
        """
        try:
            import psycopg2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "psycopg2 is required for the Postgres backend.\n"
                "Install it with:  pip install \"rag[postgres]\"\n"
                f"Original error: {exc}"
            ) from exc

        params = self._params

        try:
            conn = self._connect(dbname=params["admin_db"])
            conn.close()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot connect to Postgres at {params['host']}:{params['port']} "
                f"(admin database: {params['admin_db']!r}).\n"
                f"Check that Postgres is running and credentials are correct.\n"
                f"Error: {exc}"
            ) from exc

        try:
            self._ensure_db()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot create or access target database {params['dbname']!r}.\n"
                f"The connected user may need CREATE DATABASE privilege (or superuser).\n"
                f"Error: {exc}"
            ) from exc

        try:
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "CREATE TEMP TABLE _rag_preflight (id INT); "
                        "DROP TABLE _rag_preflight;"
                    )
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            raise RuntimeError(
                f"No write access to database {params['dbname']!r}.\n"
                f"The connected user needs CREATE and INSERT privileges.\n"
                f"Error: {exc}"
            ) from exc

    def setup(self, *, embedding_dim: int) -> None:
        from psycopg2 import sql as pgsql
        self._embedding_dim = embedding_dim
        self._ensure_db()
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
                cur.execute(
                    pgsql.SQL(
                        "CREATE TABLE IF NOT EXISTS {} ("
                        "id BIGSERIAL PRIMARY KEY,"
                        "text TEXT NOT NULL,"
                        "source TEXT,"
                        "vector vector({}),"
                        "content_hash TEXT NOT NULL UNIQUE"
                        ")"
                    ).format(pgsql.Identifier(self.table_name), pgsql.SQL(str(embedding_dim)))
                )
                cur.execute(
                    pgsql.SQL(
                        "CREATE INDEX IF NOT EXISTS {} ON {} USING ivfflat (vector vector_cosine_ops)"
                    ).format(
                        pgsql.Identifier(f"{self.table_name}_vector_idx"),
                        pgsql.Identifier(self.table_name),
                    )
                )
            conn.commit()
        finally:
            conn.close()

    def add(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        from pgvector.psycopg2 import register_vector, Vector
        from psycopg2 import sql as pgsql
        conn = self._connect()
        register_vector(conn)
        try:
            insert_sql = pgsql.SQL(
                "INSERT INTO {} (text, source, vector, content_hash) "
                "VALUES (%s, %s, %s, %s) "
                "ON CONFLICT (content_hash) DO UPDATE "
                "SET vector = EXCLUDED.vector, text = EXCLUDED.text, source = EXCLUDED.source"
            ).format(pgsql.Identifier(self.table_name))
            with conn.cursor() as cur:
                for row in rows:
                    text = str(row.get("text", ""))
                    source = row.get("source")
                    vector = row.get("vector")
                    if not isinstance(vector, (list, tuple)):
                        raise TypeError("row['vector'] must be a list[float]")
                    content_hash = _pg_content_hash(str(source) if source else None, text)
                    cur.execute(insert_sql, (text, source, Vector(vector), content_hash))
            conn.commit()
        finally:
            conn.close()

    def query(self, *, query_vector: Sequence[float], k: int) -> list[str]:
        if k <= 0:
            return []
        from pgvector.psycopg2 import register_vector, Vector
        from psycopg2 import sql as pgsql
        conn = self._connect()
        register_vector(conn)
        try:
            with conn.cursor() as cur:
                cur.execute(
                    pgsql.SQL("SELECT text FROM {} ORDER BY vector <=> %s LIMIT %s").format(
                        pgsql.Identifier(self.table_name)
                    ),
                    (Vector(list(query_vector)), k),
                )
                return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()

    def info(self) -> dict[str, Any]:
        rows: int | None = None
        try:
            from psycopg2 import sql as pgsql
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        pgsql.SQL("SELECT COUNT(*) FROM {}").format(
                            pgsql.Identifier(self.table_name)
                        )
                    )
                    result = cur.fetchone()
                    rows = result[0] if result else 0
            finally:
                conn.close()
        except Exception:
            pass
        return {"backend": "postgres", "uri": self.uri, "table": self.table_name, "rows": rows}

    def exists(self) -> bool:
        try:
            from psycopg2 import sql as pgsql
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT EXISTS ("
                        "  SELECT FROM information_schema.tables"
                        "  WHERE table_schema = 'public' AND table_name = %s"
                        ")",
                        (self.table_name,),
                    )
                    if not cur.fetchone()[0]:
                        return False
                    cur.execute(
                        pgsql.SQL("SELECT 1 FROM {} LIMIT 1").format(
                            pgsql.Identifier(self.table_name)
                        )
                    )
                    return cur.fetchone() is not None
            finally:
                conn.close()
        except Exception:
            return False

    def has_source(self, source: str) -> bool:
        if not self.exists():
            return False
        try:
            from psycopg2 import sql as pgsql
            conn = self._connect()
            try:
                with conn.cursor() as cur:
                    cur.execute(
                        pgsql.SQL("SELECT 1 FROM {} WHERE source = %s LIMIT 1").format(
                            pgsql.Identifier(self.table_name)
                        ),
                        (source,),
                    )
                    return cur.fetchone() is not None
            finally:
                conn.close()
        except Exception:
            return False

    def delete_source(self, source: str) -> None:
        if not self.exists():
            return
        from psycopg2 import sql as pgsql
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    pgsql.SQL("DELETE FROM {} WHERE source = %s").format(
                        pgsql.Identifier(self.table_name)
                    ),
                    (source,),
                )
            conn.commit()
        finally:
            conn.close()

    def list_sources(self) -> list[str]:
        if not self.exists():
            return []
        from psycopg2 import sql as pgsql
        conn = self._connect()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    pgsql.SQL(
                        "SELECT DISTINCT source FROM {} WHERE source IS NOT NULL ORDER BY source"
                    ).format(pgsql.Identifier(self.table_name))
                )
                return [r[0] for r in cur.fetchall()]
        finally:
            conn.close()


class SqliteAdapter(DbAdapter):
    """SQLite vector store.  URI: sqlite:///path/to/db"""

    def __init__(self, uri: str, table_name: str) -> None:
        self.uri = uri
        self.table_name = table_name
        self._db_path = self._parse_path(uri)
        self._embedding_dim: int | None = None

    @staticmethod
    def _parse_path(uri: str) -> Path:
        if not uri.startswith("sqlite://"):
            raise ValueError(f"Invalid SQLite URI: {uri!r}")
        raw_path = uri.removeprefix("sqlite://")
        if not raw_path:
            raise ValueError(f"SQLite URI must include a path, got {uri!r}")
        return Path(raw_path).expanduser()

    @staticmethod
    def _cosine_distance(left: str, right: str) -> float:
        try:
            left_values = json.loads(left)
            right_values = json.loads(right)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid vector payload in SQLite store: {exc}") from exc

        if len(left_values) != len(right_values):
            raise ValueError(
                f"query dim {len(right_values)} doesn't match index dim {len(left_values)}"
            )

        dot = sum(float(a) * float(b) for a, b in zip(left_values, right_values))
        left_norm = math.sqrt(sum(float(value) * float(value) for value in left_values))
        right_norm = math.sqrt(sum(float(value) * float(value) for value in right_values))
        if left_norm == 0.0 or right_norm == 0.0:
            return 1.0
        cosine_similarity = dot / (left_norm * right_norm)
        return 1.0 - cosine_similarity

    def _connect(self) -> sqlite3.Connection:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.create_function("cosine_distance", 2, self._cosine_distance)
        return conn

    def _meta_key(self) -> str:
        return f"{self.table_name}:embedding_dim"

    def _get_stored_dim(self) -> int | None:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT value FROM rag_meta WHERE key = ?",
                (self._meta_key(),),
            )
            row = cur.fetchone()
            return int(row["value"]) if row is not None else None
        except sqlite3.OperationalError:
            return None
        finally:
            conn.close()

    def _require_matching_dim(self, embedding_dim: int) -> None:
        stored_dim = self._get_stored_dim()
        if stored_dim is not None and stored_dim != embedding_dim:
            raise RuntimeError(
                f"embedding dim {embedding_dim} doesn't match existing index dim {stored_dim}"
            )

    def setup(self, *, embedding_dim: int) -> None:
        self._require_matching_dim(embedding_dim)
        self._embedding_dim = embedding_dim
        conn = self._connect()
        try:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS rag_meta ("
                "key TEXT PRIMARY KEY,"
                "value TEXT NOT NULL"
                ")"
            )
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {self.table_name} ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "text TEXT NOT NULL,"
                "source TEXT,"
                "vector TEXT NOT NULL,"
                "content_hash TEXT NOT NULL UNIQUE"
                ")"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS {self.table_name}_source_idx "
                f"ON {self.table_name} (source)"
            )
            conn.execute(
                "INSERT INTO rag_meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (self._meta_key(), str(embedding_dim)),
            )
            conn.commit()
        finally:
            conn.close()

    def add(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        conn = self._connect()
        try:
            insert_sql = (
                f"INSERT INTO {self.table_name} (text, source, vector, content_hash) "
                "VALUES (?, ?, ?, ?) "
                "ON CONFLICT(content_hash) DO UPDATE "
                "SET vector = excluded.vector, text = excluded.text, source = excluded.source"
            )
            for row in rows:
                text = str(row.get("text", ""))
                source = row.get("source")
                vector = row.get("vector")
                if not isinstance(vector, (list, tuple)):
                    raise TypeError("row['vector'] must be a list[float]")
                vector_values = [float(value) for value in vector]
                if self._embedding_dim is None:
                    self._embedding_dim = self._get_stored_dim()
                if self._embedding_dim is not None and len(vector_values) != self._embedding_dim:
                    raise RuntimeError(
                        f"embedding dim {len(vector_values)} doesn't match "
                        f"existing index dim {self._embedding_dim}"
                    )
                content_hash = _pg_content_hash(str(source) if source else None, text)
                conn.execute(
                    insert_sql,
                    (text, source, json.dumps(vector_values), content_hash),
                )
            conn.commit()
        finally:
            conn.close()

    def query(self, *, query_vector: Sequence[float], k: int) -> list[str]:
        if k <= 0:
            return []
        if self._embedding_dim is None:
            self._embedding_dim = self._get_stored_dim()
        if self._embedding_dim is not None and len(query_vector) != self._embedding_dim:
            raise RuntimeError(
                f"query dim {len(query_vector)} doesn't match index dim {self._embedding_dim}"
            )
        query_payload = json.dumps([float(value) for value in query_vector])
        conn = self._connect()
        try:
            cur = conn.execute(
                f"SELECT text FROM {self.table_name} "
                "ORDER BY cosine_distance(vector, ?) ASC "
                "LIMIT ?",
                (query_payload, k),
            )
            return [row["text"] for row in cur.fetchall()]
        finally:
            conn.close()

    def info(self) -> dict[str, Any]:
        rows: int | None = None
        stored_dim = self._get_stored_dim()
        try:
            conn = self._connect()
            try:
                cur = conn.execute(f"SELECT COUNT(*) AS count FROM {self.table_name}")
                result = cur.fetchone()
                rows = int(result["count"]) if result is not None else 0
            finally:
                conn.close()
        except sqlite3.OperationalError:
            pass
        return {
            "backend": "sqlite",
            "uri": self.uri,
            "table": self.table_name,
            "rows": rows,
            "embedding_dim": stored_dim,
        }

    def exists(self) -> bool:
        conn = self._connect()
        try:
            cur = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
                (self.table_name,),
            )
            if cur.fetchone() is None:
                return False
            cur = conn.execute(f"SELECT 1 FROM {self.table_name} LIMIT 1")
            return cur.fetchone() is not None
        finally:
            conn.close()

    def has_source(self, source: str) -> bool:
        if not self.exists():
            return False
        conn = self._connect()
        try:
            cur = conn.execute(
                f"SELECT 1 FROM {self.table_name} WHERE source = ? LIMIT 1",
                (source,),
            )
            return cur.fetchone() is not None
        finally:
            conn.close()

    def delete_source(self, source: str) -> None:
        if not self.exists():
            return
        conn = self._connect()
        try:
            conn.execute(f"DELETE FROM {self.table_name} WHERE source = ?", (source,))
            conn.commit()
        finally:
            conn.close()

    def list_sources(self) -> list[str]:
        if not self.exists():
            return []
        conn = self._connect()
        try:
            cur = conn.execute(
                f"SELECT DISTINCT source FROM {self.table_name} "
                "WHERE source IS NOT NULL ORDER BY source"
            )
            return [str(row["source"]) for row in cur.fetchall()]
        finally:
            conn.close()


def _weaviate_parse_uri(uri: str) -> dict[str, Any]:
    """Parse weaviate:// URIs for local, custom, or cloud-backed connections."""
    parsed = urlparse(uri)
    if parsed.scheme != "weaviate":
        raise ValueError(f"Invalid Weaviate URI: {uri!r}")

    params = parse_qs(parsed.query)
    host = parsed.hostname
    port = parsed.port
    secure = params.get("secure", ["false"])[0].lower() in {"1", "true", "yes"}
    grpc_host = params.get("grpc_host", [host])[0]
    grpc_port = int(params.get("grpc_port", ["443" if secure else "50051"])[0])
    grpc_secure = params.get("grpc_secure", [str(secure).lower()])[0].lower() in {
        "1",
        "true",
        "yes",
    }
    api_key_env = params.get("api_key_env", ["WEAVIATE_API_KEY"])[0]
    cluster_url = params.get("cluster_url", [None])[0]

    if cluster_url:
        return {
            "mode": "cloud",
            "cluster_url": cluster_url,
            "api_key_env": api_key_env,
        }

    if host is None:
        raise ValueError(
            "Weaviate URI must include a host, for example "
            "'weaviate://localhost:8080' or "
            "'weaviate://cluster.example.com:443?secure=true'."
        )

    if host in {"localhost", "127.0.0.1"} and not secure and parsed.username is None:
        return {
            "mode": "local",
            "host": host,
            "port": port or 8080,
            "grpc_port": grpc_port,
        }

    if secure and api_key_env in os.environ and parsed.username is None and grpc_host in {None, host}:
        scheme = "https" if secure else "http"
        return {
            "mode": "cloud",
            "cluster_url": f"{scheme}://{host}:{port or 443}",
            "api_key_env": api_key_env,
        }

    return {
        "mode": "custom",
        "http_host": host,
        "http_port": port or (443 if secure else 8080),
        "http_secure": secure,
        "grpc_host": grpc_host or host,
        "grpc_port": grpc_port,
        "grpc_secure": grpc_secure,
        "api_key_env": api_key_env,
    }


def _weaviate_collection_name(table_name: str) -> str:
    """Map the repo's free-form table name to a valid Weaviate collection name."""
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", table_name.strip())
    if not cleaned:
        cleaned = "Embeddings"
    if not cleaned[0].isalpha():
        cleaned = f"C_{cleaned}"
    return cleaned[0].upper() + cleaned[1:]


class WeaviateAdapter(DbAdapter):
    """Weaviate-backed vector store. URI: weaviate://HOST:PORT[?secure=true&...]"""

    def __init__(self, uri: str, table_name: str) -> None:
        self.uri = uri
        self.table_name = table_name
        self.collection_name = _weaviate_collection_name(table_name)
        self._conn = _weaviate_parse_uri(uri)
        self._client: Any = None

    def _connect(self) -> Any:
        if self._client is not None:
            return self._client
        try:
            import weaviate
            import weaviate.classes as wvc
            from weaviate.classes.init import Auth
        except ImportError as exc:
            raise RuntimeError(
                "The 'weaviate-client' package is required for the Weaviate backend.\n"
                "Install it with:  pip install \"rag[weaviate]\"\n"
                f"Original error: {exc}"
            ) from exc

        mode = self._conn["mode"]
        if mode == "local":
            client = weaviate.connect_to_local(
                host=self._conn["host"],
                port=self._conn["port"],
                grpc_port=self._conn["grpc_port"],
            )
        elif mode == "cloud":
            api_key = os.environ.get(self._conn["api_key_env"])
            if not api_key:
                raise RuntimeError(
                    f"{self._conn['api_key_env']} environment variable is not set for Weaviate."
                )
            client = weaviate.connect_to_weaviate_cloud(
                cluster_url=self._conn["cluster_url"],
                auth_credentials=Auth.api_key(api_key),
            )
        else:
            api_key = os.environ.get(self._conn["api_key_env"])
            auth = Auth.api_key(api_key) if api_key else None
            client = weaviate.connect_to_custom(
                http_host=self._conn["http_host"],
                http_port=self._conn["http_port"],
                http_secure=self._conn["http_secure"],
                grpc_host=self._conn["grpc_host"],
                grpc_port=self._conn["grpc_port"],
                grpc_secure=self._conn["grpc_secure"],
                auth_credentials=auth,
            )

        if not client.is_ready():
            client.close()
            raise RuntimeError(f"Weaviate at {self.uri!r} is not ready.")

        self._client = client
        self._wvc = wvc
        return client

    def _collection(self) -> Any:
        client = self._connect()
        return client.collections.use(self.collection_name)

    def setup(self, *, embedding_dim: int) -> None:
        client = self._connect()
        if client.collections.exists(self.collection_name):
            return
        client.collections.create(
            self.collection_name,
            vector_config=self._wvc.config.Configure.Vectors.self_provided(),
            properties=[
                self._wvc.config.Property(
                    name="text",
                    data_type=self._wvc.config.DataType.TEXT,
                ),
                self._wvc.config.Property(
                    name="source",
                    data_type=self._wvc.config.DataType.TEXT,
                ),
            ],
        )

    def add(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        collection = self._collection()
        with collection.batch.fixed_size(batch_size=200) as batch:
            for row in rows:
                vector = row.get("vector")
                if not isinstance(vector, (list, tuple)):
                    raise TypeError("row['vector'] must be a list[float]")
                batch.add_object(
                    properties={
                        "text": str(row.get("text", "")),
                        "source": row.get("source"),
                    },
                    vector=list(vector),
                )
        failed = getattr(collection.batch, "failed_objects", [])
        if failed:
            raise RuntimeError(f"Failed to import {len(failed)} object(s) into Weaviate.")

    def query(self, *, query_vector: Sequence[float], k: int) -> list[str]:
        if k <= 0:
            return []
        response = self._collection().query.near_vector(
            near_vector=list(query_vector),
            limit=k,
            return_properties=["text"],
        )
        return [
            obj.properties["text"]
            for obj in response.objects
            if isinstance(obj.properties, dict) and obj.properties.get("text")
        ]

    def info(self) -> dict[str, Any]:
        rows: int | None = None
        if self.exists():
            try:
                response = self._collection().aggregate.over_all(total_count=True)
                rows = getattr(response, "total_count", None)
            except Exception:
                rows = None
        return {
            "backend": "weaviate",
            "uri": self.uri,
            "table": self.table_name,
            "collection": self.collection_name,
            "rows": rows,
        }

    def exists(self) -> bool:
        try:
            client = self._connect()
            if not client.collections.exists(self.collection_name):
                return False
            response = self._collection().query.fetch_objects(limit=1, return_properties=["source"])
            return len(response.objects) > 0
        except Exception:
            return False

    def has_source(self, source: str) -> bool:
        if not self._connect().collections.exists(self.collection_name):
            return False
        response = self._collection().query.fetch_objects(
            filters=self._wvc.query.Filter.by_property("source").equal(source),
            limit=1,
            return_properties=["source"],
        )
        return len(response.objects) > 0

    def delete_source(self, source: str) -> None:
        if not self._connect().collections.exists(self.collection_name):
            return
        self._collection().data.delete_many(
            where=self._wvc.query.Filter.by_property("source").equal(source)
        )

    def list_sources(self) -> list[str]:
        if not self._connect().collections.exists(self.collection_name):
            return []
        response = self._collection().query.fetch_objects(
            return_properties=["source"],
            limit=10_000,
        )
        return sorted(
            {
                obj.properties["source"]
                for obj in response.objects
                if isinstance(obj.properties, dict) and obj.properties.get("source")
            }
        )

    def preflight(self) -> None:
        self._connect()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _vertex_parse_uri(uri: str) -> tuple[str, str]:
    """Parse vertex://project_id/corpus_display_name into (project_id, corpus_display_name)."""
    if not uri.startswith("vertex://"):
        raise ValueError(f"Invalid Vertex URI: {uri!r}")
    rest = uri.removeprefix("vertex://").strip("/")
    if "/" not in rest:
        raise ValueError(f"Vertex URI must be vertex://PROJECT_ID/CORPUS_NAME, got {uri!r}")
    project_id, _, corpus_name = rest.partition("/")
    return project_id.strip(), corpus_name.strip() or rest


class VertexAdapter(DbAdapter):
    """Vertex AI RAG Engine.  URI: vertex://PROJECT_ID/CORPUS_NAME

    Ingest: uploads local files; Vertex handles chunking and embedding.
    Ask: retrieves by question text (use query_by_text), then generate with Gemini.
    Requires: pip install 'rag[vertex]', GCP project, gcloud auth application-default login.
    """

    def __init__(self, uri: str, table_name: str) -> None:
        self.uri = uri
        self.table_name = table_name
        self._project_id, self._corpus_display_name = _vertex_parse_uri(uri)
        from rag.vertex import VertexRagService
        self._service = VertexRagService(project_id=self._project_id)
        self._corpus_name: str | None = None

    def setup(self, *, embedding_dim: int) -> None:
        self._corpus_name = self._service.create_or_get_corpus(self._corpus_display_name)

    def add(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        if "path" not in rows[0]:
            raise NotImplementedError(
                "Vertex adapter expects rows with 'path' and 'source' (file paths); "
                "use vertex:// ingest path in CLI."
            )
        corpus = self._corpus_name or self._service.create_or_get_corpus(self._corpus_display_name)
        self._corpus_name = corpus
        paths = [Path(r["path"]) for r in rows if r.get("path")]
        self._service.upload_files(corpus, paths)

    def query(self, *, query_vector: Sequence[float], k: int) -> list[str]:
        raise NotImplementedError(
            "Vertex RAG retrieves by text; use query_by_text(question, k) from the CLI."
        )

    def query_by_text(self, query_text: str, k: int) -> list[str]:
        """Retrieve top-k chunks by question text (Vertex RAG uses semantic search by text)."""
        corpus = self._corpus_name or self._service.create_or_get_corpus(self._corpus_display_name)
        self._corpus_name = corpus
        return self._service.retrieval_query(corpus, query_text, top_k=k)

    def info(self) -> dict[str, Any]:
        corpus = self._corpus_name or self._service.get_corpus_by_display_name(self._corpus_display_name)
        files = self._service.list_files(corpus) if corpus else []
        return {
            "backend": "vertex",
            "uri": self.uri,
            "table": self.table_name,
            "corpus": corpus,
            "files": len(files),
        }

    def exists(self) -> bool:
        try:
            corpus = self._corpus_name or self._service.get_corpus_by_display_name(self._corpus_display_name)
            if corpus is None:
                return False
            self._corpus_name = corpus
            files = self._service.list_files(corpus)
            return len(files) > 0
        except Exception:
            return False

    def has_source(self, source: str) -> bool:
        corpus = self._corpus_name or self._service.get_corpus_by_display_name(self._corpus_display_name)
        if corpus is None:
            return False
        self._corpus_name = corpus
        for display_name, _ in self._service.list_files(corpus):
            if display_name == source:
                return True
        return False

    def delete_source(self, source: str) -> None:
        corpus = self._corpus_name or self._service.get_corpus_by_display_name(self._corpus_display_name)
        if corpus is None:
            return
        self._corpus_name = corpus
        for display_name, full_name in self._service.list_files(corpus):
            if display_name == source:
                self._service.delete_file(full_name)
                return

    def list_sources(self) -> list[str]:
        corpus = self._corpus_name or self._service.get_corpus_by_display_name(self._corpus_display_name)
        if corpus is None:
            return []
        self._corpus_name = corpus
        return sorted(display_name for display_name, _ in self._service.list_files(corpus))

    def preflight(self) -> None:
        self._service.create_or_get_corpus(self._corpus_display_name)


def _bedrock_kb_parse_uri(uri: str) -> tuple[str, str | None]:
    """Parse bedrock-kb://KB_ID or bedrock-kb://KB_ID/DS_ID into (kb_id, ds_id | None)."""
    if not uri.startswith("bedrock-kb://"):
        raise ValueError(f"Invalid Bedrock KB URI: {uri!r}")
    rest = uri.removeprefix("bedrock-kb://").strip("/")
    if not rest:
        raise ValueError(f"Bedrock KB URI must include a Knowledge Base ID, got {uri!r}")
    if "/" in rest:
        kb_id, _, ds_id = rest.partition("/")
        return kb_id.strip(), ds_id.strip() or None
    return rest.strip(), None


class BedrockKbAdapter(DbAdapter):
    """Amazon Bedrock Knowledge Bases.  URI: bedrock-kb://KB_ID  or  bedrock-kb://KB_ID/DS_ID

    Ingest: uploads local files to S3 and triggers KB ingestion; AWS handles chunking and embedding.
    Ask: retrieves by question text (use query_by_text), then generate with Bedrock LLM.
    Requires: pip install 'rag[bedrock]', AWS credentials, and a Knowledge Base in AWS Console.
    """

    def __init__(self, uri: str, table_name: str) -> None:
        self.uri = uri
        self.table_name = table_name
        self._kb_id, self._ds_id = _bedrock_kb_parse_uri(uri)
        from rag.bedrock_kb import BedrockKbService
        self._service = BedrockKbService(knowledge_base_id=self._kb_id)
        self._last_job_id: str | None = None

    def setup(self, *, embedding_dim: int) -> None:
        pass

    def add(self, rows: Sequence[dict[str, Any]]) -> None:
        if not rows:
            return
        if "path" not in rows[0]:
            raise NotImplementedError(
                "Bedrock KB adapter expects rows with 'path' and 'source' (file paths); "
                "use bedrock-kb:// ingest path in CLI."
            )
        paths = [Path(r["path"]) for r in rows if r.get("path")]
        self._last_job_id = self._service.upload_files(paths, self._ds_id)

    def wait_for_ingestion(self) -> None:
        """Block until the KB ingestion job completes."""
        if self._last_job_id:
            self._service.wait_for_ingestion(self._last_job_id, self._ds_id)

    def query(self, *, query_vector: Sequence[float], k: int) -> list[str]:
        raise NotImplementedError(
            "Bedrock KB retrieves by text; use query_by_text(question, k) from the CLI."
        )

    def query_by_text(self, query_text: str, k: int) -> list[str]:
        """Retrieve top-k chunks by question text (Bedrock KB uses semantic search by text)."""
        return self._service.retrieval_query(query_text, top_k=k)

    def info(self) -> dict[str, Any]:
        return {
            "backend": "bedrock-kb",
            "uri": self.uri,
            "table": self.table_name,
            "knowledge_base_id": self._kb_id,
            "data_source_id": self._ds_id,
        }

    def exists(self) -> bool:
        try:
            keys = self._service.list_s3_objects(self._ds_id)
            return len(keys) > 0
        except Exception:
            return False

    def has_source(self, source: str) -> bool:
        try:
            return self._service.s3_object_exists(source, self._ds_id)
        except Exception:
            return False

    def delete_source(self, source: str) -> None:
        try:
            self._service.delete_s3_object(source, self._ds_id)
        except Exception:
            pass

    def list_sources(self) -> list[str]:
        try:
            return sorted(self._service.list_s3_objects(self._ds_id))
        except Exception:
            return []

    def preflight(self) -> None:
        try:
            self._service.get_knowledge_base()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot access Bedrock Knowledge Base {self._kb_id!r}.\n"
                f"Check that the KB exists and AWS credentials have bedrock:GetKnowledgeBase permission.\n"
                f"Error: {exc}"
            ) from exc


def get_db_adapter(uri: str, table_name: str) -> DbAdapter:
    """Instantiate the correct DbAdapter from a URI string."""
    if uri.startswith(("postgres://", "postgresql://")):
        return PostgresAdapter(uri, table_name)
    if uri.startswith("weaviate://"):
        return WeaviateAdapter(uri, table_name)
    if uri.startswith("sqlite://"):
        return SqliteAdapter(uri, table_name)
    if uri.startswith("vertex://"):
        return VertexAdapter(uri, table_name)
    if uri.startswith("bedrock-kb://"):
        return BedrockKbAdapter(uri, table_name)
    return LanceDbAdapter(uri, table_name)

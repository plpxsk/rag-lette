"""
Database adapter interface and placeholder implementations.

To add a new backend:
  1. Subclass DbAdapter and implement all abstract methods.
  2. Add a URI-prefix match in get_db_adapter().
"""
from __future__ import annotations

import hashlib
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

    def setup(self, *, embedding_dim: int) -> None:
        self._connect()

    def add(self, rows: Sequence[dict[str, Any]]) -> None:
        db = self._connect()
        if self.table_name in db.table_names():
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
        db = self._connect()
        return self.table_name in db.table_names()

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


class SqliteAdapter(DbAdapter):
    """SQLite vector store.  URI: sqlite:///path/to/db"""

    def __init__(self, uri: str, table_name: str) -> None:
        self.uri = uri
        self.table_name = table_name

    def setup(self, *, embedding_dim: int) -> None:
        raise NotImplementedError("SQLite backend not yet implemented")

    def add(self, rows: Sequence[dict[str, Any]]) -> None:
        raise NotImplementedError("SQLite backend not yet implemented")

    def query(self, *, query_vector: Sequence[float], k: int) -> list[str]:
        raise NotImplementedError("SQLite backend not yet implemented")

    def info(self) -> dict[str, Any]:
        return {"backend": "sqlite", "uri": self.uri, "table": self.table_name}

    def exists(self) -> bool:
        raise NotImplementedError("SQLite backend not yet implemented")

    def has_source(self, source: str) -> bool:
        raise NotImplementedError("SQLite backend not yet implemented")

    def delete_source(self, source: str) -> None:
        raise NotImplementedError("SQLite backend not yet implemented")


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

    def preflight(self) -> None:
        self._service.create_or_get_corpus(self._corpus_display_name)


def get_db_adapter(uri: str, table_name: str) -> DbAdapter:
    """Instantiate the correct DbAdapter from a URI string."""
    if uri.startswith(("postgres://", "postgresql://")):
        return PostgresAdapter(uri, table_name)
    if uri.startswith("sqlite://"):
        return SqliteAdapter(uri, table_name)
    if uri.startswith("vertex://"):
        return VertexAdapter(uri, table_name)
    return LanceDbAdapter(uri, table_name)

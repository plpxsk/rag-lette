"""Vertex AI RAG Engine — managed corpus, chunking, embedding, and retrieval.

Requires: pip install 'rag[vertex]', GCP project with Vertex AI API enabled,
and Application Default Credentials (gcloud auth application-default login).
"""
from __future__ import annotations

from pathlib import Path

DEFAULT_LOCATION = "us-central1"


class VertexRagService:
    """Wrapper around Vertex AI RAG Engine: corpus, file upload, retrieval."""

    def __init__(self, project_id: str, location: str = DEFAULT_LOCATION) -> None:
        try:
            import vertexai
            from vertexai import rag
        except ImportError as exc:
            raise ImportError(
                "The 'google-cloud-aiplatform' package is required for Vertex AI RAG.\n"
                "Install it with:  pip install \"rag[vertex]\"\n"
                f"Original error: {exc}"
            ) from exc
        self._vertexai = vertexai
        self._rag = rag
        self.project_id = project_id
        self.location = location
        self._initialized = False

    def _init(self) -> None:
        if not self._initialized:
            self._vertexai.init(project=self.project_id, location=self.location)
            self._initialized = True

    def get_corpus_by_display_name(self, display_name: str) -> str | None:
        """Return corpus resource name if it exists, else None (does not create)."""
        self._init()
        for corpus in self._rag.list_corpora():
            if corpus.display_name == display_name:
                return corpus.name
        return None

    def create_or_get_corpus(self, display_name: str) -> str:
        """Return corpus resource name, creating the corpus if it does not exist."""
        existing = self.get_corpus_by_display_name(display_name)
        if existing is not None:
            return existing
        self._init()
        embedding_model_config = self._rag.RagEmbeddingModelConfig(
            vertex_prediction_endpoint=self._rag.VertexPredictionEndpoint(
                publisher_model="publishers/google/models/text-embedding-005"
            )
        )
        rag_corpus = self._rag.create_corpus(
            display_name=display_name,
            backend_config=self._rag.RagVectorDbConfig(
                rag_embedding_model_config=embedding_model_config
            ),
        )
        return rag_corpus.name

    def upload_files(self, corpus_name: str, paths: list[Path]) -> None:
        """Upload local files to the RAG corpus. Vertex handles chunking and embedding."""
        self._init()
        for path in paths:
            path = Path(path).resolve()
            if not path.is_file():
                continue
            self._rag.upload_file(
                corpus_name=corpus_name,
                path=str(path),
                display_name=path.name,
                description=path.name,
            )

    def retrieval_query(self, corpus_name: str, text: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k relevant context chunks from the corpus. Returns list of text strings."""
        self._init()
        config = self._rag.RagRetrievalConfig(top_k=top_k)
        response = self._rag.retrieval_query(
            rag_resources=[self._rag.RagResource(rag_corpus=corpus_name)],
            text=text,
            rag_retrieval_config=config,
        )
        chunks: list[str] = []
        if hasattr(response, "contexts") and response.contexts:
            for ctx in response.contexts:
                if hasattr(ctx, "text") and ctx.text:
                    chunks.append(ctx.text)
        return chunks

    def list_files(self, corpus_name: str) -> list[tuple[str, str]]:
        """List files in the corpus. Returns list of (display_name, full_resource_name)."""
        self._init()
        result: list[tuple[str, str]] = []
        for f in self._rag.list_files(corpus_name=corpus_name):
            result.append((getattr(f, "display_name", "") or "", getattr(f, "name", "") or ""))
        return result

    def delete_file(self, file_name: str) -> None:
        """Delete a RAG file by its full resource name."""
        self._init()
        self._rag.delete_file(name=file_name)

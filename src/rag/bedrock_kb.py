"""Amazon Bedrock Knowledge Bases — managed chunking, embedding, and retrieval.

Requires: pip install 'rag[bedrock]', AWS credentials configured, and a Knowledge Base
created in the AWS Console (wizard provisions IAM role, OpenSearch Serverless, S3 data source).
"""
from __future__ import annotations

from pathlib import Path


class BedrockKbService:
    """Wrapper around boto3 bedrock-agent and bedrock-agent-runtime clients."""

    def __init__(self, knowledge_base_id: str, region: str | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise ImportError(
                "boto3 is required for Bedrock Knowledge Bases.\n"
                "Install it with:  pip install \"rag[bedrock]\"\n"
                f"Original error: {exc}"
            ) from exc
        self.knowledge_base_id = knowledge_base_id
        self._region = region
        session = boto3.Session(region_name=region)
        self._agent = session.client("bedrock-agent")
        self._agent_runtime = session.client("bedrock-agent-runtime")
        self._s3 = session.client("s3")

    def get_knowledge_base(self) -> dict:
        """Return KB metadata; raises if KB does not exist."""
        response = self._agent.get_knowledge_base(knowledgeBaseId=self.knowledge_base_id)
        return response["knowledgeBase"]

    def get_data_source(self, ds_id: str | None = None) -> dict:
        """Return data source config. Uses ds_id if given, else returns first S3 data source."""
        if ds_id:
            response = self._agent.get_data_source(
                knowledgeBaseId=self.knowledge_base_id, dataSourceId=ds_id
            )
            return response["dataSource"]
        paginator = self._agent.get_paginator("list_data_sources")
        for page in paginator.paginate(knowledgeBaseId=self.knowledge_base_id):
            for ds in page.get("dataSourceSummaries", []):
                if ds.get("dataSourceId"):
                    response = self._agent.get_data_source(
                        knowledgeBaseId=self.knowledge_base_id,
                        dataSourceId=ds["dataSourceId"],
                    )
                    return response["dataSource"]
        raise RuntimeError(
            f"No data sources found for Knowledge Base {self.knowledge_base_id!r}"
        )

    def get_s3_bucket_from_data_source(self, ds_id: str | None = None) -> tuple[str, str]:
        """Return (bucket_name, key_prefix) from the S3 data source config."""
        ds = self.get_data_source(ds_id)
        try:
            s3_config = ds["dataSourceConfiguration"]["s3Configuration"]
            bucket_arn = s3_config["bucketArn"]
            # ARN format: arn:aws:s3:::bucket-name
            bucket = bucket_arn.split(":::")[-1]
            prefix = s3_config.get("inclusionPrefixes", [""])[0] or ""
            prefix = prefix.rstrip("/")
            return bucket, prefix
        except (KeyError, IndexError) as exc:
            raise RuntimeError(
                f"Cannot extract S3 bucket from data source config: {exc}"
            ) from exc

    def upload_files(self, paths: list[Path], ds_id: str | None = None) -> str:
        """Upload files to the KB's S3 bucket and start an ingestion job. Returns job ID."""
        bucket, prefix = self.get_s3_bucket_from_data_source(ds_id)
        ds = self.get_data_source(ds_id)
        actual_ds_id = ds["dataSourceId"]

        for path in paths:
            path = Path(path).resolve()
            if not path.is_file():
                continue
            key = f"{prefix}/{path.name}" if prefix else path.name
            self._s3.upload_file(str(path), bucket, key)

        response = self._agent.start_ingestion_job(
            knowledgeBaseId=self.knowledge_base_id, dataSourceId=actual_ds_id
        )
        return response["ingestionJob"]["ingestionJobId"]

    def wait_for_ingestion(self, job_id: str, ds_id: str | None = None) -> None:
        """Poll until the ingestion job reaches COMPLETE or FAILED."""
        import time

        ds = self.get_data_source(ds_id)
        actual_ds_id = ds["dataSourceId"]

        while True:
            response = self._agent.get_ingestion_job(
                knowledgeBaseId=self.knowledge_base_id,
                dataSourceId=actual_ds_id,
                ingestionJobId=job_id,
            )
            status = response["ingestionJob"]["status"]
            if status == "COMPLETE":
                return
            if status == "FAILED":
                raise RuntimeError(
                    f"Bedrock ingestion job {job_id!r} failed. "
                    "Check AWS Console for details."
                )
            time.sleep(5)

    def retrieval_query(self, text: str, top_k: int = 5) -> list[str]:
        """Retrieve top-k relevant chunks from the KB. Returns list of text strings."""
        response = self._agent_runtime.retrieve(
            knowledgeBaseId=self.knowledge_base_id,
            retrievalQuery={"text": text},
            retrievalConfiguration={
                "vectorSearchConfiguration": {"numberOfResults": top_k}
            },
        )
        chunks: list[str] = []
        for result in response.get("retrievalResults", []):
            content = result.get("content", {})
            chunk_text = content.get("text", "")
            if chunk_text:
                chunks.append(chunk_text)
        return chunks

    def s3_object_exists(self, key: str, ds_id: str | None = None) -> bool:
        """Return True if an S3 object with the given key exists in the KB's bucket."""
        bucket, prefix = self.get_s3_bucket_from_data_source(ds_id)
        full_key = f"{prefix}/{key}" if prefix else key
        try:
            self._s3.head_object(Bucket=bucket, Key=full_key)
            return True
        except Exception:
            return False

    def delete_s3_object(self, key: str, ds_id: str | None = None) -> None:
        """Delete an S3 object from the KB's bucket."""
        bucket, prefix = self.get_s3_bucket_from_data_source(ds_id)
        full_key = f"{prefix}/{key}" if prefix else key
        self._s3.delete_object(Bucket=bucket, Key=full_key)

    def list_s3_objects(self, ds_id: str | None = None) -> list[str]:
        """Return list of object keys (filenames) in the KB's S3 bucket/prefix."""
        bucket, prefix = self.get_s3_bucket_from_data_source(ds_id)
        kwargs: dict = {"Bucket": bucket}
        if prefix:
            kwargs["Prefix"] = prefix + "/"
        keys: list[str] = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(**kwargs):
            for obj in page.get("Contents", []):
                raw_key = obj["Key"]
                name = raw_key[len(prefix) + 1:] if prefix else raw_key
                if name:
                    keys.append(name)
        return keys

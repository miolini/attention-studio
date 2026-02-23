from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class BatchStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchItem:
    id: str
    input_data: dict[str, Any]
    status: BatchStatus = BatchStatus.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: str | None = None
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "input_data": self.input_data,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchItem:
        return cls(
            id=data["id"],
            input_data=data["input_data"],
            status=BatchStatus(data["status"]),
            result=data.get("result"),
            error=data.get("error"),
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
        )


@dataclass
class BatchJob:
    id: str
    name: str
    items: list[BatchItem] = field(default_factory=list)
    status: BatchStatus = BatchStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    processor_type: str = ""
    processor_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "items": [item.to_dict() for item in self.items],
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "processor_type": self.processor_type,
            "processor_config": self.processor_config,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchJob:
        return cls(
            id=data["id"],
            name=data["name"],
            items=[BatchItem.from_dict(i) for i in data["items"]],
            status=BatchStatus(data["status"]),
            created_at=data["created_at"],
            started_at=data.get("started_at"),
            completed_at=data.get("completed_at"),
            processor_type=data.get("processor_type", ""),
            processor_config=data.get("processor_config", {}),
        )

    def get_progress(self) -> tuple[int, int]:
        completed = sum(1 for item in self.items if item.status == BatchStatus.COMPLETED)
        failed = sum(1 for item in self.items if item.status == BatchStatus.FAILED)
        return completed + failed, len(self.items)

    def get_success_rate(self) -> float:
        if not self.items:
            return 0.0
        completed = sum(1 for item in self.items if item.status == BatchStatus.COMPLETED)
        return completed / len(self.items)


class BatchProcessor:
    def __init__(self, storage_dir: Path | None = None):
        if storage_dir is None:
            storage_dir = Path.home() / ".attention_studio" / "batch"
        self._storage_dir = storage_dir
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._jobs: dict[str, BatchJob] = {}
        self._processor: Callable[[dict[str, Any]], dict[str, Any]] | None = None
        self._load_jobs()

    def _load_jobs(self) -> None:
        for path in self._storage_dir.glob("*.json"):
            try:
                data = json.loads(path.read_text())
                job = BatchJob.from_dict(data)
                self._jobs[job.id] = job
            except (json.JSONDecodeError, KeyError):
                continue

    def _save_job(self, job: BatchJob) -> None:
        path = self._storage_dir / f"{job.id}.json"
        path.write_text(json.dumps(job.to_dict(), indent=2))

    def set_processor(self, processor: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._processor = processor

    def create_job(
        self,
        name: str,
        inputs: list[dict[str, Any]],
        processor_type: str = "",
        processor_config: dict[str, Any] | None = None,
    ) -> BatchJob:
        job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(self._jobs)}"
        items = [
            BatchItem(id=f"{job_id}_{i}", input_data=inp)
            for i, inp in enumerate(inputs)
        ]
        job = BatchJob(
            id=job_id,
            name=name,
            items=items,
            processor_type=processor_type,
            processor_config=processor_config or {},
        )
        self._jobs[job_id] = job
        self._save_job(job)
        return job

    def get_job(self, job_id: str) -> BatchJob | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[BatchJob]:
        return list(self._jobs.values())

    def delete_job(self, job_id: str) -> bool:
        if job_id in self._jobs:
            del self._jobs[job_id]
            path = self._storage_dir / f"{job_id}.json"
            if path.exists():
                path.unlink()
            return True
        return False

    def run_job(
        self,
        job_id: str,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> BatchJob | None:
        job = self._jobs.get(job_id)
        if job is None or self._processor is None:
            return None

        job.status = BatchStatus.RUNNING
        job.started_at = datetime.now().isoformat()
        self._save_job(job)

        total = len(job.items)

        for completed, item in enumerate(job.items, start=1):
            item.status = BatchStatus.RUNNING
            item.started_at = datetime.now().isoformat()

            try:
                result = self._processor(item.input_data)
                item.result = result
                item.status = BatchStatus.COMPLETED
            except Exception as e:
                item.error = str(e)
                item.status = BatchStatus.FAILED

            item.completed_at = datetime.now().isoformat()

            if progress_callback:
                progress_callback(completed, total)

            self._save_job(job)

        job.status = BatchStatus.COMPLETED
        job.completed_at = datetime.now().isoformat()
        self._save_job(job)

        return job

    def cancel_job(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False

        job.status = BatchStatus.CANCELLED
        job.completed_at = datetime.now().isoformat()

        for item in job.items:
            if item.status == BatchStatus.PENDING or item.status == BatchStatus.RUNNING:
                item.status = BatchStatus.CANCELLED
                item.completed_at = datetime.now().isoformat()

        self._save_job(job)
        return True

    def retry_failed(self, job_id: str) -> BatchJob | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None

        for item in job.items:
            if item.status == BatchStatus.FAILED:
                item.status = BatchStatus.PENDING
                item.result = None
                item.error = None
                item.started_at = None
                item.completed_at = None

        job.status = BatchStatus.PENDING
        self._save_job(job)
        return job

    def export_results(self, job_id: str, path: Path | str) -> bool:
        job = self._jobs.get(job_id)
        if job is None:
            return False

        path = Path(path)
        results = {
            "job_id": job.id,
            "name": job.name,
            "exported_at": datetime.now().isoformat(),
            "processor_type": job.processor_type,
            "results": [
                {
                    "id": item.id,
                    "input": item.input_data,
                    "result": item.result,
                    "error": item.error,
                    "status": item.status.value,
                }
                for item in job.items
            ],
        }

        path.write_text(json.dumps(results, indent=2))
        return True


_processor: BatchProcessor | None = None


def get_batch_processor() -> BatchProcessor:
    global _processor
    if _processor is None:
        _processor = BatchProcessor()
    return _processor

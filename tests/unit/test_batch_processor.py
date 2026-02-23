from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from attention_studio.core.batch_processor import (
    BatchJob,
    BatchItem,
    BatchProcessor,
    BatchStatus,
    get_batch_processor,
)


class TestBatchItem:
    def test_default_values(self):
        item = BatchItem(id="item_1", input_data={"prompt": "test"})
        assert item.id == "item_1"
        assert item.status == BatchStatus.PENDING
        assert item.result is None
        assert item.error is None

    def test_to_dict(self):
        item = BatchItem(
            id="item_1",
            input_data={"prompt": "test"},
            status=BatchStatus.COMPLETED,
            result={"output": "result"},
        )
        data = item.to_dict()
        assert data["id"] == "item_1"
        assert data["status"] == "completed"
        assert data["result"]["output"] == "result"

    def test_from_dict(self):
        data = {
            "id": "item_1",
            "input_data": {"prompt": "test"},
            "status": "completed",
            "result": {"output": "result"},
        }
        item = BatchItem.from_dict(data)
        assert item.id == "item_1"
        assert item.status == BatchStatus.COMPLETED


class TestBatchJob:
    def test_default_values(self):
        job = BatchJob(id="job_1", name="Test Job")
        assert job.id == "job_1"
        assert job.name == "Test Job"
        assert job.status == BatchStatus.PENDING
        assert job.items == []

    def test_add_items(self):
        job = BatchJob(id="job_1", name="Test Job")
        job.items = [
            BatchItem(id="item_1", input_data={"prompt": "test1"}),
            BatchItem(id="item_2", input_data={"prompt": "test2"}),
        ]
        assert len(job.items) == 2

    def test_get_progress(self):
        job = BatchJob(id="job_1", name="Test Job")
        job.items = [
            BatchItem(id="item_1", input_data={}, status=BatchStatus.COMPLETED),
            BatchItem(id="item_2", input_data={}, status=BatchStatus.COMPLETED),
            BatchItem(id="item_3", input_data={}, status=BatchStatus.PENDING),
        ]
        completed, total = job.get_progress()
        assert completed == 2
        assert total == 3

    def test_get_success_rate(self):
        job = BatchJob(id="job_1", name="Test Job")
        job.items = [
            BatchItem(id="item_1", input_data={}, status=BatchStatus.COMPLETED),
            BatchItem(id="item_2", input_data={}, status=BatchStatus.COMPLETED),
            BatchItem(id="item_3", input_data={}, status=BatchStatus.FAILED),
        ]
        rate = job.get_success_rate()
        assert rate == pytest.approx(2 / 3)

    def test_to_dict(self):
        job = BatchJob(
            id="job_1",
            name="Test Job",
            processor_type="feature_extraction",
        )
        job.items = [BatchItem(id="item_1", input_data={})]
        data = job.to_dict()
        assert data["id"] == "job_1"
        assert data["processor_type"] == "feature_extraction"
        assert len(data["items"]) == 1


class TestBatchProcessor:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture(autouse=True)
    def reset_global(self):
        import attention_studio.core.batch_processor as bp
        bp._processor = None
        yield
        bp._processor = None

    def test_create_job(self, temp_dir):
        processor = BatchProcessor(temp_dir)
        inputs = [{"prompt": "test1"}, {"prompt": "test2"}]
        job = processor.create_job("Test Job", inputs, processor_type="extraction")

        assert job.name == "Test Job"
        assert len(job.items) == 2
        assert job.processor_type == "extraction"

    def test_get_job(self, temp_dir):
        processor = BatchProcessor(temp_dir)
        job = processor.create_job("Test Job", [{"prompt": "test"}])

        retrieved = processor.get_job(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id

    def test_list_jobs(self, temp_dir):
        processor = BatchProcessor(temp_dir)
        processor.create_job("Job 1", [{"prompt": "test1"}])
        processor.create_job("Job 2", [{"prompt": "test2"}])

        jobs = processor.list_jobs()
        assert len(jobs) == 2

    def test_delete_job(self, temp_dir):
        processor = BatchProcessor(temp_dir)
        job = processor.create_job("Test Job", [{"prompt": "test"}])

        result = processor.delete_job(job.id)
        assert result is True
        assert processor.get_job(job.id) is None

    def test_run_job(self, temp_dir):
        processor = BatchProcessor(temp_dir)
        processor.set_processor(lambda x: {"result": x["prompt"].upper()})

        job = processor.create_job("Test Job", [{"prompt": "hello"}, {"prompt": "world"}])
        processor.run_job(job.id)

        job = processor.get_job(job.id)
        assert job.status == BatchStatus.COMPLETED
        assert job.items[0].result["result"] == "HELLO"

    def test_run_job_with_progress(self, temp_dir):
        processor = BatchProcessor(temp_dir)
        processor.set_processor(lambda x: {"result": "ok"})

        job = processor.create_job("Test Job", [{"prompt": f"test{i}"} for i in range(3)])
        progress = []

        def callback(completed: int, total: int) -> None:
            progress.append((completed, total))

        processor.run_job(job.id, callback)

        assert len(progress) == 3
        assert progress[-1] == (3, 3)

    def test_run_job_with_error(self, temp_dir):
        processor = BatchProcessor(temp_dir)

        def failing_processor(x):
            raise ValueError("Test error")

        processor.set_processor(failing_processor)
        job = processor.create_job("Test Job", [{"prompt": "test"}])
        processor.run_job(job.id)

        job = processor.get_job(job.id)
        assert job.status == BatchStatus.COMPLETED
        assert job.items[0].status == BatchStatus.FAILED
        assert job.items[0].error == "Test error"

    def test_cancel_job(self, temp_dir):
        processor = BatchProcessor(temp_dir)
        job = processor.create_job("Test Job", [{"prompt": f"test{i}"} for i in range(5)])

        processor.cancel_job(job.id)
        job = processor.get_job(job.id)

        assert job.status == BatchStatus.CANCELLED

    def test_retry_failed(self, temp_dir):
        processor = BatchProcessor(temp_dir)

        def failing_processor(x):
            raise ValueError("Test error")

        processor.set_processor(failing_processor)
        job = processor.create_job("Test Job", [{"prompt": "test"}])
        processor.run_job(job.id)

        processor.set_processor(lambda x: {"result": "ok"})
        processor.retry_failed(job.id)
        processor.run_job(job.id)

        job = processor.get_job(job.id)
        assert job.items[0].status == BatchStatus.COMPLETED

    def test_export_results(self, temp_dir):
        processor = BatchProcessor(temp_dir)
        processor.set_processor(lambda x: {"result": x["prompt"]})

        job = processor.create_job("Test Job", [{"prompt": "hello"}])
        processor.run_job(job.id)

        export_path = temp_dir / "results.json"
        processor.export_results(job.id, export_path)

        data = json.loads(export_path.read_text())
        assert data["job_id"] == job.id
        assert len(data["results"]) == 1
        assert data["results"][0]["result"]["result"] == "hello"


def test_get_batch_processor_singleton():
    import attention_studio.core.batch_processor as bp

    bp._processor = None
    processor1 = get_batch_processor()
    processor2 = get_batch_processor()
    assert processor1 is processor2
    bp._processor = None

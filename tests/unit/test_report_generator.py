from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from attention_studio.core.report_generator import (
    Report,
    ReportFormat,
    ReportGenerator,
    ReportMetadata,
    ReportSectionType,
    get_report_generator,
)


class TestReportMetadata:
    def test_default_values(self):
        metadata = ReportMetadata(title="Test Report")
        assert metadata.title == "Test Report"
        assert metadata.author == ""
        assert metadata.description == ""
        assert metadata.tags == []
        assert metadata.model_name == ""
        assert metadata.dataset_name == ""


class TestReport:
    def test_create_report(self):
        metadata = ReportMetadata(title="Test", author="John")
        report = Report(metadata)
        assert report.metadata.title == "Test"
        assert report.metadata.author == "John"
        assert report.sections == []

    def test_add_title(self):
        report = Report(ReportMetadata(title="Test"))
        report.add_title("My Title")
        assert len(report.sections) == 1
        assert report.sections[0].section_type == ReportSectionType.TITLE

    def test_add_text(self):
        report = Report(ReportMetadata(title="Test"))
        report.add_text("Some content")
        assert len(report.sections) == 1
        assert report.sections[0].section_type == ReportSectionType.TEXT
        assert report.sections[0].content == "Some content"

    def test_add_heading(self):
        report = Report(ReportMetadata(title="Test"))
        report.add_heading("Section", level=2)
        assert report.sections[0].data["level"] == 2

    def test_add_metrics(self):
        report = Report(ReportMetadata(title="Test"))
        report.add_metrics({"accuracy": 0.95, "loss": 0.05})
        assert len(report.sections) == 1
        assert report.sections[0].section_type == ReportSectionType.METRICS
        assert report.sections[0].data["metrics"]["accuracy"] == 0.95

    def test_add_table(self):
        report = Report(ReportMetadata(title="Test"))
        report.add_table(["Col1", "Col2"], [["a", "b"], ["c", "d"]], title="Table")
        assert report.sections[0].data["headers"] == ["Col1", "Col2"]
        assert len(report.sections[0].data["rows"]) == 2

    def test_add_list(self):
        report = Report(ReportMetadata(title="Test"))
        report.add_list(["item1", "item2"])
        assert report.sections[0].data["items"] == ["item1", "item2"]

    def test_add_code(self):
        report = Report(ReportMetadata(title="Test"))
        report.add_code("print('hello')", language="python")
        assert report.sections[0].content == "print('hello')"
        assert report.sections[0].data["language"] == "python"

    def test_add_divider(self):
        report = Report(ReportMetadata(title="Test"))
        report.add_divider()
        assert report.sections[0].section_type == ReportSectionType.DIVIDER


class TestReportGenerator:
    def test_create_report(self):
        gen = ReportGenerator()
        report = gen.create_report("Test", author="John", description="A test")
        assert report.metadata.title == "Test"
        assert report.metadata.author == "John"
        assert report.metadata.description == "A test"

    def test_generate_analysis_report(self):
        gen = ReportGenerator()
        report = gen.generate_analysis_report(
            title="Analysis",
            model_name="gpt2",
            dataset_name="test",
            metrics={"accuracy": 0.95},
            features=[{"name": "feature1", "score": 0.9}],
        )
        assert report.metadata.model_name == "gpt2"
        assert report.metadata.dataset_name == "test"

    def test_generate_comparison_report(self):
        gen = ReportGenerator()
        report = gen.generate_comparison_report(
            title="Comparison",
            items=[{"name": "model1", "acc": 0.9}, {"name": "model2", "acc": 0.85}],
            metric_names=["acc"],
        )
        assert report.metadata.title == "Comparison"

    def test_generate_training_report(self):
        gen = ReportGenerator()
        report = gen.generate_training_report(
            title="Training",
            model_name="gpt2",
            epochs=10,
            train_metrics={"loss": [1.0, 0.5, 0.1]},
            val_metrics={"loss": [1.1, 0.6, 0.2]},
        )
        assert report.metadata.model_name == "gpt2"

    def test_to_markdown(self):
        gen = ReportGenerator()
        report = gen.create_report("Test")
        report.add_title("Main Title")
        report.add_text("Some text")
        report.add_metrics({"acc": 0.95})

        md = gen.to_markdown(report)
        assert "# Main Title" in md
        assert "Some text" in md
        assert "**acc**" in md

    def test_to_html(self):
        gen = ReportGenerator()
        report = gen.create_report("Test")
        report.add_title("Test Report")
        report.add_text("Content")

        html = gen.to_html(report)
        assert "<!DOCTYPE html>" in html
        assert "<title>Test</title>" in html

    def test_to_text(self):
        gen = ReportGenerator()
        report = gen.create_report("Test")
        report.add_title("Test Title")
        report.add_text("Some text")

        text = gen.to_text(report)
        assert "TEST" in text
        assert "Some text" in text

    def test_save_markdown(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator()
            report = gen.create_report("Test")
            report.add_text("Content")

            path = Path(tmpdir) / "report.md"
            gen.save(report, path, ReportFormat.MARKDOWN)

            assert path.exists()
            assert "Content" in path.read_text()

    def test_save_html(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator()
            report = gen.create_report("Test")
            report.add_text("Content")

            path = Path(tmpdir) / "report.html"
            gen.save(report, path, ReportFormat.HTML)

            assert path.exists()
            assert "<!DOCTYPE html>" in path.read_text()

    def test_save_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            gen = ReportGenerator()
            report = gen.create_report("Test")
            report.add_text("Content")

            path = Path(tmpdir) / "report.txt"
            gen.save(report, path, ReportFormat.TEXT)

            assert path.exists()
            assert "Content" in path.read_text()

    def test_add_figure(self):
        gen = ReportGenerator()
        report = gen.create_report("Test")
        report.add_figure("plot.png", caption="A plot")

        assert report.sections[0].section_type == ReportSectionType.FIGURE
        assert report.sections[0].data["path"] == "plot.png"


def test_get_report_generator_singleton():
    gen1 = get_report_generator()
    gen2 = get_report_generator()
    assert gen1 is gen2

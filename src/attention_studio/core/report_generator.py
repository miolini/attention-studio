from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any


class ReportFormat(Enum):
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"


class ReportSectionType(Enum):
    TITLE = "title"
    METRICS = "metrics"
    TABLE = "table"
    FIGURE = "figure"
    TEXT = "text"
    CODE = "code"
    LIST = "list"
    DIVIDER = "divider"


@dataclass
class ReportSection:
    section_type: ReportSectionType
    title: str = ""
    content: Any = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReportMetadata:
    title: str
    author: str = ""
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    tags: list[str] = field(default_factory=list)
    model_name: str = ""
    dataset_name: str = ""


class Report:
    def __init__(self, metadata: ReportMetadata):
        self.metadata = metadata
        self.sections: list[ReportSection] = []

    def add_title(self, title: str) -> Report:
        self.sections.append(ReportSection(
            section_type=ReportSectionType.TITLE,
            title=title,
        ))
        return self

    def add_heading(self, title: str, level: int = 2) -> Report:
        self.sections.append(ReportSection(
            section_type=ReportSectionType.TEXT,
            title=title,
            data={"level": level},
        ))
        return self

    def add_text(self, content: str) -> Report:
        self.sections.append(ReportSection(
            section_type=ReportSectionType.TEXT,
            content=content,
        ))
        return self

    def add_metrics(self, metrics: dict[str, float], title: str = "Metrics") -> Report:
        self.sections.append(ReportSection(
            section_type=ReportSectionType.METRICS,
            title=title,
            data={"metrics": metrics},
        ))
        return self

    def add_table(
        self,
        headers: list[str],
        rows: list[list[str]],
        title: str = "",
    ) -> Report:
        self.sections.append(ReportSection(
            section_type=ReportSectionType.TABLE,
            title=title,
            data={"headers": headers, "rows": rows},
        ))
        return self

    def add_figure(self, path: str, caption: str = "") -> Report:
        self.sections.append(ReportSection(
            section_type=ReportSectionType.FIGURE,
            title=caption,
            data={"path": path},
        ))
        return self

    def add_code(self, code: str, language: str = "") -> Report:
        self.sections.append(ReportSection(
            section_type=ReportSectionType.CODE,
            content=code,
            data={"language": language},
        ))
        return self

    def add_list(self, items: list[str], ordered: bool = False) -> Report:
        self.sections.append(ReportSection(
            section_type=ReportSectionType.LIST,
            data={"items": items, "ordered": ordered},
        ))
        return self

    def add_divider(self) -> Report:
        self.sections.append(ReportSection(
            section_type=ReportSectionType.DIVIDER,
        ))
        return self


class ReportGenerator:
    def __init__(self):
        self._templates: dict[str, Report] = {}

    def create_report(self, title: str, author: str = "", description: str = "") -> Report:
        metadata = ReportMetadata(
            title=title,
            author=author,
            description=description,
        )
        return Report(metadata)

    def generate_analysis_report(
        self,
        title: str,
        model_name: str,
        dataset_name: str,
        metrics: dict[str, float],
        features: list[dict[str, Any]],
        notes: str = "",
    ) -> Report:
        report = self.create_report(title)
        report.metadata.model_name = model_name
        report.metadata.dataset_name = dataset_name

        report.add_title(title)
        report.add_text(f"**Model:** {model_name}")
        report.add_text(f"**Dataset:** {dataset_name}")
        report.add_text(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.add_divider()

        if notes:
            report.add_heading("Overview")
            report.add_text(notes)
            report.add_divider()

        report.add_heading("Metrics")
        report.add_metrics(metrics)
        report.add_divider()

        if features:
            report.add_heading("Top Features")
            rows = []
            for f in features[:20]:
                name = f.get("name", f.get("feature", "N/A"))
                score = f.get("score", f.get("importance", 0.0))
                rows.append([name, f"{score:.4f}"])
            report.add_table(["Feature", "Score"], rows)
            report.add_divider()

        return report

    def generate_comparison_report(
        self,
        title: str,
        items: list[dict[str, Any]],
        metric_names: list[str],
    ) -> Report:
        report = self.create_report(title)

        report.add_title(title)
        report.add_text(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.add_divider()

        rows = []
        for item in items:
            row = [item.get("name", "Unknown")]
            for m in metric_names:
                row.append(f"{item.get(m, 0.0):.4f}")
            rows.append(row)

        headers = ["Item"] + metric_names
        report.add_heading("Comparison")
        report.add_table(headers, rows)

        return report

    def generate_training_report(
        self,
        title: str,
        model_name: str,
        epochs: int,
        train_metrics: dict[str, list[float]],
        val_metrics: dict[str, list[float]],
    ) -> Report:
        report = self.create_report(title)
        report.metadata.model_name = model_name

        report.add_title(title)
        report.add_text(f"**Model:** {model_name}")
        report.add_text(f"**Epochs:** {epochs}")
        report.add_text(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.add_divider()

        final_metrics = {}
        for name, values in train_metrics.items():
            if values:
                final_metrics[f"train_{name}"] = values[-1]
        for name, values in val_metrics.items():
            if values:
                final_metrics[f"val_{name}"] = values[-1]

        report.add_heading("Final Metrics")
        report.add_metrics(final_metrics)
        report.add_divider()

        rows = []
        epoch_nums = list(range(1, min(epochs + 1, 11)))
        for epoch in epoch_nums:
            idx = epoch - 1
            row = [f"Epoch {epoch}"]
            for name in list(train_metrics.keys())[:3]:
                if idx < len(train_metrics[name]):
                    row.append(f"{train_metrics[name][idx]:.4f}")
                else:
                    row.append("N/A")
            rows.append(row)

        if rows:
            headers = ["Epoch"] + list(train_metrics.keys())[:3]
            report.add_heading("Training Progress")
            report.add_table(headers, rows)

        return report

    def to_markdown(self, report: Report) -> str:
        lines = []
        lines.append(f"# {report.metadata.title}")
        lines.append("")

        if report.metadata.author:
            lines.append(f"*Author: {report.metadata.author}*")
        if report.metadata.description:
            lines.append(f"_{report.metadata.description}_")
        lines.append("")

        for section in report.sections:
            if section.section_type == ReportSectionType.TITLE:
                lines.append(f"# {section.title}")
                lines.append("")

            elif section.section_type == ReportSectionType.TEXT:
                level = section.data.get("level", 1)
                prefix = "#" * level
                lines.append(f"{prefix} {section.title or section.content}")
                lines.append("")

            elif section.section_type == ReportSectionType.METRICS:
                if section.title:
                    lines.append(f"## {section.title}")
                for name, value in section.data.get("metrics", {}).items():
                    lines.append(f"- **{name}**: {value:.4f}" if isinstance(value, float) else f"- **{name}**: {value}")
                lines.append("")

            elif section.section_type == ReportSectionType.TABLE:
                if section.title:
                    lines.append(f"## {section.title}")
                headers = section.data.get("headers", [])
                rows = section.data.get("rows", [])
                if headers:
                    lines.append("| " + " | ".join(headers) + " |")
                    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
                for row in rows:
                    lines.append("| " + " | ".join(str(c) for c in row) + " |")
                lines.append("")

            elif section.section_type == ReportSectionType.FIGURE:
                lines.append(f"![{section.title}]({section.data.get('path', '')})")
                lines.append("")

            elif section.section_type == ReportSectionType.CODE:
                lang = section.data.get("language", "")
                lines.append(f"```{lang}")
                lines.append(str(section.content))
                lines.append("```")
                lines.append("")

            elif section.section_type == ReportSectionType.LIST:
                items = section.data.get("items", [])
                ordered = section.data.get("ordered", False)
                for i, item in enumerate(items, 1):
                    prefix = f"{i}." if ordered else "-"
                    lines.append(f"{prefix} {item}")
                lines.append("")

            elif section.section_type == ReportSectionType.DIVIDER:
                lines.append("---")
                lines.append("")

        return "\n".join(lines)

    def to_html(self, report: Report) -> str:
        md = self.to_markdown(report)
        html = self._markdown_to_html(md)
        return self._wrap_html(report.metadata.title, html)

    def _markdown_to_html(self, md: str) -> str:
        html = md
        import re

        html = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
        html = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
        html = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html)
        html = re.sub(r'^\* (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^- (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)
        html = re.sub(r'^\d+\. (.+)$', r'<li>\1</li>', html, flags=re.MULTILINE)

        html = re.sub(r'\| (.+) \|', r'<tr><td>\1</td></tr>', html)
        html = re.sub(r'<tr><td>(.+)</td></tr>', lambda m: '<tr>' + ''.join(f'<td{c}</td>' for c in m.group(1).split(' | ')) + '</tr>', html)

        html = re.sub(r'^---$', r'<hr>', html, flags=re.MULTILINE)
        html = re.sub(r'```(\w*)\n([\s\S]*?)```', r'<pre><code class="language-\1">\2</code></pre>', html)
        html = re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', r'<figure><img src="\2" alt="\1"><figcaption>\1</figcaption></figure>', html)

        html = re.sub(r'\n\n+', r'\n', html)
        return html

    def _wrap_html(self, title: str, body: str) -> str:
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            line-height: 1.6;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            color: #333;
        }}
        h1, h2, h3 {{ color: #2c3e50; margin-top: 1.5em; }}
        table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        li {{ margin: 0.3em 0; }}
        code {{ background: #f4f4f4; padding: 2px 6px; border-radius: 3px; }}
        pre {{ background: #f4f4f4; padding: 15px; border-radius: 5px; overflow-x: auto; }}
        hr {{ border: none; border-top: 1px solid #eee; margin: 2em 0; }}
        figure {{ margin: 1.5em 0; }}
        img {{ max-width: 100%; height: auto; }}
        figcaption {{ color: #666; font-size: 0.9em; text-align: center; margin-top: 0.5em; }}
    </style>
</head>
<body>
{body}
</body>
</html>"""

    def to_text(self, report: Report) -> str:
        lines = []
        lines.append(report.metadata.title.upper())
        lines.append("=" * len(report.metadata.title))
        lines.append("")

        if report.metadata.author:
            lines.append(f"Author: {report.metadata.author}")
        if report.metadata.description:
            lines.append(f"Description: {report.metadata.description}")
        lines.append("")

        for section in report.sections:
            if section.section_type == ReportSectionType.TITLE:
                lines.append(section.title)
                lines.append("-" * len(section.title))
                lines.append("")

            elif section.section_type == ReportSectionType.TEXT:
                level = section.data.get("level", 1)
                prefix = "#" * level
                lines.append(f"{prefix} {section.title or section.content}")
                lines.append("")

            elif section.section_type == ReportSectionType.METRICS:
                if section.title:
                    lines.append(f"[{section.title}]")
                for name, value in section.data.get("metrics", {}).items():
                    lines.append(f"  {name}: {value:.4f}" if isinstance(value, float) else f"  {name}: {value}")
                lines.append("")

            elif section.section_type == ReportSectionType.TABLE:
                if section.title:
                    lines.append(f"[{section.title}]")
                headers = section.data.get("headers", [])
                rows = section.data.get("rows", [])
                if headers:
                    col_widths = [len(h) for h in headers]
                    for row in rows:
                        for i, cell in enumerate(row):
                            if i < len(col_widths):
                                col_widths[i] = max(col_widths[i], len(str(cell)))
                    lines.append("  " + "  ".join(h.ljust(w) for h, w in zip(headers, col_widths, strict=True)))
                    lines.append("  " + "  ".join("-" * w for w in col_widths))
                    for row in rows:
                        lines.append("  " + "  ".join(str(c).ljust(w) for c, w in zip(row, col_widths, strict=True)))
                lines.append("")

            elif section.section_type == ReportSectionType.LIST:
                items = section.data.get("items", [])
                ordered = section.data.get("ordered", False)
                for i, item in enumerate(items, 1):
                    prefix = f"{i}." if ordered else "  -"
                    lines.append(f"{prefix} {item}")
                lines.append("")

            elif section.section_type == ReportSectionType.DIVIDER:
                lines.append("-" * 60)
                lines.append("")

        return "\n".join(lines)

    def save(self, report: Report, path: Path | str, output_format: ReportFormat = ReportFormat.MARKDOWN) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == ReportFormat.MARKDOWN:
            content = self.to_markdown(report)
        elif output_format == ReportFormat.HTML:
            content = self.to_html(report)
        else:
            content = self.to_text(report)

        path.write_text(content)


_report_generator: ReportGenerator | None = None


def get_report_generator() -> ReportGenerator:
    global _report_generator
    if _report_generator is None:
        _report_generator = ReportGenerator()
    return _report_generator

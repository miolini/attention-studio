from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import networkx as nx
import torch


def serialize_value(obj: Any) -> Any:
    if obj is None:
        return None
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    if is_dataclass(obj):
        return {k: serialize_value(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: serialize_value(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [serialize_value(item) for item in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def export_graph_to_json(graph: nx.DiGraph, nodes: dict, edges: dict, output_path: Path) -> None:
    data = {
        "nodes": {},
        "edges": [],
        "graph_info": {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "exported_at": datetime.now().isoformat(),
        },
    }

    for node_id, node_data in nodes.items():
        if is_dataclass(node_data):
            node_dict = asdict(node_data)
            data["nodes"][node_id] = serialize_value(node_dict)
        else:
            data["nodes"][node_id] = serialize_value(node_data)

    for edge_key, edge_data in edges.items():
        edge_dict = {
            "source": edge_key[0],
            "target": edge_key[1],
        }
        if is_dataclass(edge_data):
            edge_dict.update(serialize_value(asdict(edge_data)))
        else:
            edge_dict.update(serialize_value(edge_data))
        data["edges"].append(edge_dict)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def export_graph_to_graphml(graph: nx.DiGraph, output_path: Path) -> None:
    for node in graph.nodes():
        node_data = graph.nodes[node]
        for key, value in list(node_data.items()):
            if isinstance(value, torch.Tensor):
                node_data[key] = str(value.shape)
            elif not isinstance(value, str | int | float | bool | type(None)):
                node_data[key] = str(value)

    for u, v in graph.edges():
        edge_data = graph.edges[u, v]
        for key, value in list(edge_data.items()):
            if isinstance(value, torch.Tensor):
                edge_data[key] = str(value.shape)
            elif not isinstance(value, str | int | float | bool | type(None)):
                edge_data[key] = str(value)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, output_path)


def export_features_to_csv(features: list[dict[str, Any]], output_path: Path) -> None:
    import csv

    if not features:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = []
    for feature in features:
        for key in feature:
            if key not in fieldnames:
                fieldnames.append(key)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feature in features:
            row = {}
            for key in fieldnames:
                value = feature.get(key, "")
                if isinstance(value, list | dict):
                    row[key] = json.dumps(value)
                elif isinstance(value, torch.Tensor):
                    row[key] = str(value.tolist())
                else:
                    row[key] = value
            writer.writerow(row)


def export_features_to_json(features: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialize_value(features), f, indent=2, ensure_ascii=False)


def save_session(
    output_path: Path,
    model_name: str | None = None,
    transcoder_configs: list[dict] | None = None,
    layer_indices: list[int] | None = None,
    graph_data: dict | None = None,
    features_data: list | None = None,
    circuits_data: dict | None = None,
    metadata: dict | None = None,
) -> None:
    session = {
        "version": "1.0",
        "saved_at": datetime.now().isoformat(),
        "model": {
            "name": model_name,
        },
        "crm": {
            "transcoder_configs": transcoder_configs,
            "layer_indices": layer_indices,
        },
        "analysis": {
            "graph": graph_data,
            "features": features_data,
            "circuits": circuits_data,
        },
        "metadata": metadata or {},
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialize_value(session), f, indent=2, ensure_ascii=False)


def load_session(input_path: Path) -> dict[str, Any]:
    with open(input_path, encoding="utf-8") as f:
        return json.load(f)


def export_report_to_html(
    output_path: Path,
    model_name: str,
    prompt: str,
    features: list[dict],
    circuits: dict,
    graph_stats: dict | None = None,
) -> None:
    features_html = ""
    for i, feat in enumerate(features[:50]):
        features_html += f"""
        <tr>
            <td>{feat.get('idx', i)}</td>
            <td>{feat.get('layer', 'N/A')}</td>
            <td>{feat.get('activation', 0):.4f}</td>
            <td>{feat.get('norm', 0):.4f}</td>
        </tr>
        """

    circuits_html = ""
    for circuit_type, circuit_list in circuits.items():
        if circuit_list:
            circuits_html += f"""
            <div class="circuit-section">
                <h3>{circuit_type.replace('_', ' ').title()}</h3>
                <p>Found {len(circuit_list)} circuits</p>
            </div>
            """

    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AttentionStudio Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #1e1e1e;
            color: #cccccc;
        }}
        h1, h2, h3 {{ color: #0e639c; }}
        .header {{
            border-bottom: 2px solid #0e639c;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }}
        .section {{
            background: #252526;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #3c3c3c;
        }}
        th {{
            background: #2d2d2d;
            color: #0e639c;
        }}
        .circuit-section {{
            background: #2d2d2d;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
            border-left: 4px solid #0e639c;
        }}
        .prompt-box {{
            background: #2d2d2d;
            padding: 15px;
            border-radius: 4px;
            font-family: 'SF Mono', monospace;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>AttentionStudio Analysis Report</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Model: {model_name}</p>
    </div>

    <div class="section">
        <h2>Input Prompt</h2>
        <div class="prompt-box">{prompt}</div>
    </div>

    <div class="section">
        <h2>Feature Analysis</h2>
        <p>Total features analyzed: {len(features)}</p>
        <table>
            <thead>
                <tr>
                    <th>Feature ID</th>
                    <th>Layer</th>
                    <th>Activation</th>
                    <th>Norm</th>
                </tr>
            </thead>
            <tbody>
                {features_html}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>Circuit Detection</h2>
        {circuits_html}
    </div>

    {'<div class="section"><h2>Graph Statistics</h2><p>Nodes: ' + str(graph_stats.get('num_nodes', 0)) + '</p><p>Edges: ' + str(graph_stats.get('num_edges', 0)) + '</p></div>' if graph_stats else ''}

    <footer style="margin-top: 40px; text-align: center; color: #666;">
        <p>Generated by AttentionStudio</p>
    </footer>
</body>
</html>
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)


def export_graph_to_png(
    scene,
    output_path: Path,
    width: int = 1920,
    height: int = 1080,
) -> bool:
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QImage, QPainter

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        rect = scene.sceneRect()
        if rect.isEmpty():
            rect = QRectF(0, 0, width, height)

        image = QImage(int(rect.width()), int(rect.height()), QImage.Format.Format_ARGB32)
        image.fill(0)

        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        scene.render(painter, QRectF(0, 0, rect.width(), rect.height()), rect)

        painter.end()

        return image.save(str(output_path))
    except Exception:
        return False

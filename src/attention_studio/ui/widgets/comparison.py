from __future__ import annotations

from typing import Any

from PySide6.QtCore import Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class PromptComparisonWidget(QWidget):
    comparison_complete = Signal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
        self._result_a = None
        self._result_b = None

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        input_group = QGroupBox("Prompts")
        input_layout = QHBoxLayout(input_group)

        prompt_a_layout = QVBoxLayout()
        prompt_a_layout.addWidget(QLabel("Prompt A:"))
        self._prompt_a_edit = QLineEdit()
        self._prompt_a_edit.setPlaceholderText("Enter first prompt...")
        prompt_a_layout.addWidget(self._prompt_a_edit)

        prompt_b_layout = QVBoxLayout()
        prompt_b_layout.addWidget(QLabel("Prompt B:"))
        self._prompt_b_edit = QLineEdit()
        self._prompt_b_edit.setPlaceholderText("Enter second prompt...")
        prompt_b_layout.addWidget(self._prompt_b_edit)

        input_layout.addLayout(prompt_a_layout)
        input_layout.addLayout(prompt_b_layout)

        self._compare_btn = QPushButton("Compare")
        self._compare_btn.clicked.connect(self._on_compare)
        input_layout.addWidget(self._compare_btn)

        layout.addWidget(input_group)

        self._splitter = QSplitter()

        self._panel_a = self._create_result_panel("Prompt A")
        self._splitter.addWidget(self._panel_a)

        self._panel_b = self._create_result_panel("Prompt B")
        self._splitter.addWidget(self._panel_b)

        self._diff_panel = self._create_diff_panel()
        self._splitter.addWidget(self._diff_panel)

        layout.addWidget(self._splitter)

    def _create_result_panel(self, title: str) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        label = QLabel(title)
        label.setStyleSheet("font-weight: bold; color: #0e639c;")
        layout.addWidget(label)

        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["Feature", "Layer", "Activation", "Norm"])
        table.setMaximumHeight(200)
        layout.addWidget(table)

        circuit_text = QTextEdit()
        circuit_text.setReadOnly(True)
        circuit_text.setMaximumHeight(150)
        circuit_text.setPlaceholderText("Circuits will appear here...")
        layout.addWidget(circuit_text)

        widget._feature_table = table
        widget._circuit_text = circuit_text

        return widget

    def _create_diff_panel(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(4, 4, 4, 4)

        label = QLabel("Differences")
        label.setStyleSheet("font-weight: bold; color: #ce9178;")
        layout.addWidget(label)

        self._diff_table = QTableWidget()
        self._diff_table.setColumnCount(4)
        self._diff_table.setHorizontalHeaderLabels(["Feature", "Layer", "Diff A", "Diff B"])
        self._diff_table.setMaximumHeight(200)
        layout.addWidget(self._diff_table)

        self._diff_text = QTextEdit()
        self._diff_text.setReadOnly(True)
        self._diff_text.setPlaceholderText("Feature differences will appear here...")
        layout.addWidget(self._diff_text)

        return widget

    def _on_compare(self):
        prompt_a = self._prompt_a_edit.text()
        prompt_b = self._prompt_b_edit.text()

        if not prompt_a or not prompt_b:
            return

        self._compare_btn.setEnabled(False)
        self._compare_btn.setText("Comparing...")

        self.comparison_complete.emit({
            "prompt_a": prompt_a,
            "prompt_b": prompt_b,
        })

    def set_results(
        self,
        features_a: list[dict[str, Any]],
        features_b: list[dict[str, Any]],
        circuits_a: dict[str, list],
        circuits_b: dict[str, list],
    ):
        self._result_a = {"features": features_a, "circuits": circuits_a}
        self._result_b = {"features": features_b, "circuits": circuits_b}

        self._populate_table(self._panel_a._feature_table, features_a[:20])
        self._populate_table(self._panel_b._feature_table, features_b[:20])

        self._panel_a._circuit_text.setPlainText(self._format_circuits(circuits_a))
        self._panel_b._circuit_text.setPlainText(self._format_circuits(circuits_b))

        self._compute_diff()

        self._compare_btn.setEnabled(True)
        self._compare_btn.setText("Compare")

    def _populate_table(self, table: QTableWidget, features: list[dict[str, Any]]):
        table.setRowCount(len(features))
        for i, feat in enumerate(features):
            table.setItem(i, 0, QTableWidgetItem(str(feat.get("idx", ""))))
            table.setItem(i, 1, QTableWidgetItem(str(feat.get("layer", ""))))
            table.setItem(i, 2, QTableWidgetItem(f"{feat.get('activation', 0):.4f}"))
            table.setItem(i, 3, QTableWidgetItem(f"{feat.get('norm', 0):.4f}"))

    def _format_circuits(self, circuits: dict[str, list]) -> str:
        lines = []
        for circuit_type, circuit_list in circuits.items():
            if circuit_list:
                strength = circuit_list[0].strength if hasattr(circuit_list[0], 'strength') else 0
                lines.append(f"{circuit_type}: {len(circuit_list)} (strength: {strength:.3f})")
        return "\n".join(lines) if lines else "No circuits detected"

    def _compute_diff(self):
        if not self._result_a or not self._result_b:
            return

        features_a = {f.get("idx"): f for f in self._result_a["features"]}
        features_b = {f.get("idx"): f for f in self._result_b["features"]}

        all_feature_ids = set(features_a.keys()) | set(features_b.keys())

        diffs = []
        for feat_id in all_feature_ids:
            feat_a = features_a.get(feat_id, {})
            feat_b = features_b.get(feat_id, {})

            act_a = feat_a.get("activation", 0)
            act_b = feat_b.get("activation", 0)

            if abs(act_a - act_b) > 0.01:
                diffs.append({
                    "feature": feat_id,
                    "layer": feat_a.get("layer", feat_b.get("layer", "?")),
                    "act_a": act_a,
                    "act_b": act_b,
                    "diff": abs(act_a - act_b),
                })

        diffs.sort(key=lambda x: x["diff"], reverse=True)

        self._diff_table.setRowCount(len(diffs[:30]))
        for i, diff in enumerate(diffs[:30]):
            self._diff_table.setItem(i, 0, QTableWidgetItem(str(diff["feature"])))
            self._diff_table.setItem(i, 1, QTableWidgetItem(str(diff["layer"])))

            item_a = QTableWidgetItem(f"{diff['act_a']:.4f}")
            item_b = QTableWidgetItem(f"{diff['act_b']:.4f}")

            if diff["act_a"] > diff["act_b"]:
                item_a.setForeground(QColor(46, 204, 113))
                item_b.setForeground(QColor(231, 76, 60))
            else:
                item_a.setForeground(QColor(231, 76, 60))
                item_b.setForeground(QColor(46, 204, 113))

            self._diff_table.setItem(i, 2, item_a)
            self._diff_table.setItem(i, 3, item_b)

        diff_text = f"Total differences: {len(diffs)}\n"
        diff_text += f"Features unique to A: {len(set(features_a.keys()) - set(features_b.keys()))}\n"
        diff_text += f"Features unique to B: {len(set(features_b.keys()) - set(features_a.keys()))}\n"
        self._diff_text.setPlainText(diff_text)

    def get_prompts(self) -> tuple[str, str]:
        return self._prompt_a_edit.text(), self._prompt_b_edit.text()

from __future__ import annotations

import asyncio
from typing import Any

from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class AgentWorker(QThread):
    result_ready = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, agent_func, *args, parent=None):
        super().__init__(parent)
        self._agent_func = agent_func
        self._args = args

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self._agent_func(*self._args))
            loop.close()
            self.result_ready.emit(result.content if hasattr(result, 'content') else str(result))
        except Exception as e:
            self.error_occurred.emit(str(e))


class AgentPanel(QWidget):
    interpretation_ready = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._agent_manager = None
        self._current_worker = None
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)

        config_group = QGroupBox("Agent Configuration")
        config_layout = QVBoxLayout(config_group)

        api_layout = QHBoxLayout()
        api_layout.addWidget(QLabel("API Key:"))
        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("Enter API key...")
        api_layout.addWidget(self._api_key_edit)
        config_layout.addLayout(api_layout)

        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self._model_combo = QComboBox()
        self._model_combo.addItems(["minimax-m2.5", "gpt-4", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet"])
        model_layout.addWidget(self._model_combo)
        model_layout.addStretch()
        config_layout.addLayout(model_layout)

        self._connect_btn = QPushButton("Connect")
        self._connect_btn.clicked.connect(self._on_connect)
        config_layout.addWidget(self._connect_btn)

        self._status_label = QLabel("Status: Not connected")
        self._status_label.setStyleSheet("color: #808080;")
        config_layout.addWidget(self._status_label)

        layout.addWidget(config_group)

        tools_group = QGroupBox("Agent Tools")
        tools_layout = QVBoxLayout(tools_group)

        interpret_layout = QHBoxLayout()
        interpret_layout.addWidget(QLabel("Feature ID:"))
        self._feature_id_spin = QLineEdit()
        self._feature_id_spin.setPlaceholderText("Feature index")
        self._feature_id_spin.setMaximumWidth(80)
        interpret_layout.addWidget(self._feature_id_spin)
        self._interpret_btn = QPushButton("Interpret Feature")
        self._interpret_btn.clicked.connect(self._on_interpret)
        self._interpret_btn.setEnabled(False)
        interpret_layout.addWidget(self._interpret_btn)
        tools_layout.addLayout(interpret_layout)

        analyze_layout = QHBoxLayout()
        self._analyze_btn = QPushButton("Analyze Circuits")
        self._analyze_btn.clicked.connect(self._on_analyze_circuits)
        self._analyze_btn.setEnabled(False)
        analyze_layout.addWidget(self._analyze_btn)

        self._verify_btn = QPushButton("Verify Hypothesis")
        self._verify_btn.clicked.connect(self._on_verify)
        self._verify_btn.setEnabled(False)
        analyze_layout.addWidget(self._verify_btn)
        tools_layout.addLayout(analyze_layout)

        layout.addWidget(tools_group)

        output_group = QGroupBox("Agent Output")
        output_layout = QVBoxLayout(output_group)

        self._output_text = QTextEdit()
        self._output_text.setReadOnly(True)
        self._output_text.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #3c3c3c;
                font-family: 'SF Mono', monospace;
            }
        """)
        output_layout.addWidget(self._output_text)

        layout.addWidget(output_group)

        self._feature_extractor = None
        self._prompt = ""
        self._feature_info = []
        self._graph_stats = {}

    def set_feature_extractor(self, extractor: Any, prompt: str):
        self._feature_extractor = extractor
        self._prompt = prompt

    def set_feature_info(self, features: list[dict], graph_stats: dict | None = None):
        self._feature_info = features
        self._graph_stats = graph_stats or {}

    def _on_connect(self):
        api_key = self._api_key_edit.text()
        if not api_key:
            self._status_label.setText("Status: API key required")
            self._status_label.setStyleSheet("color: #f44336;")
            return

        from attention_studio.agents.agents import AgentConfig, AgentManager

        config = AgentConfig(
            api_key=api_key,
            model=self._model_combo.currentText(),
        )

        try:
            self._agent_manager = AgentManager(config)
            self._status_label.setText("Status: Connected")
            self._status_label.setStyleSheet("color: #4caf50;")
            self._connect_btn.setText("Reconnect")
            self._interpret_btn.setEnabled(True)
            self._analyze_btn.setEnabled(True)
            self._verify_btn.setEnabled(True)
        except Exception as e:
            self._status_label.setText(f"Status: Error - {e}")
            self._status_label.setStyleSheet("color: #f44336;")

    def _on_interpret(self):
        if not self._agent_manager or not self._feature_extractor:
            self._output_text.setText("Error: Agent not connected or no feature extractor available")
            return

        try:
            feature_idx = int(self._feature_id_spin.text())
        except ValueError:
            self._output_text.setText("Error: Invalid feature index")
            return

        self._output_text.setText("Interpreting feature...")
        self._interpret_btn.setEnabled(False)

        async def interpret():
            return await self._agent_manager.feature_agent.interpret_feature(
                self._feature_extractor,
                feature_idx,
                self._prompt,
            )

        self._current_worker = AgentWorker(interpret)
        self._current_worker.result_ready.connect(self._on_interpret_result)
        self._current_worker.error_occurred.connect(self._on_error)
        self._current_worker.start()

    @Slot(str)
    def _on_interpret_result(self, result: str):
        self._output_text.setText(f"Feature Interpretation:\n\n{result}")
        self._interpret_btn.setEnabled(True)
        self.interpretation_ready.emit(result)

    def _on_analyze_circuits(self):
        if not self._agent_manager or not self._feature_info:
            self._output_text.setText("Error: Agent not connected or no feature info available")
            return

        self._output_text.setText("Analyzing circuits...")
        self._analyze_btn.setEnabled(False)

        async def analyze():
            return await self._agent_manager.circuit_agent.find_patterns(
                self._feature_info,
                self._graph_stats,
            )

        self._current_worker = AgentWorker(analyze)
        self._current_worker.result_ready.connect(self._on_analyze_result)
        self._current_worker.error_occurred.connect(self._on_error)
        self._current_worker.start()

    @Slot(str)
    def _on_analyze_result(self, result: str):
        self._output_text.setText(f"Circuit Analysis:\n\n{result}")
        self._analyze_btn.setEnabled(True)

    def _on_verify(self):
        if not self._agent_manager or not self._feature_info:
            self._output_text.setText("Error: Agent not connected or no feature info available")
            return

        top_feature = self._feature_info[0] if self._feature_info else {}
        hypothesis = f"Feature {top_feature.get('idx', '?')} at layer {top_feature.get('layer', '?')} influences model output"

        self._output_text.setText("Verifying hypothesis...")
        self._verify_btn.setEnabled(False)

        async def verify():
            return await self._agent_manager.verifier_agent.verify_hypothesis(
                hypothesis,
                top_feature,
            )

        self._current_worker = AgentWorker(verify)
        self._current_worker.result_ready.connect(self._on_verify_result)
        self._current_worker.error_occurred.connect(self._on_error)
        self._current_worker.start()

    @Slot(str)
    def _on_verify_result(self, result: str):
        self._output_text.setText(f"Verification Plan:\n\n{result}")
        self._verify_btn.setEnabled(True)

    @Slot(str)
    def _on_error(self, error: str):
        self._output_text.setText(f"Error: {error}")
        self._interpret_btn.setEnabled(True)
        self._analyze_btn.setEnabled(True)
        self._verify_btn.setEnabled(True)

    def cleanup(self):
        if self._current_worker and self._current_worker.isRunning():
            self._current_worker.quit()
            self._current_worker.wait()

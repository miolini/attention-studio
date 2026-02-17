from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class TrainingPanel(QWidget):
    buildRequested = Signal()  # noqa: N815
    trainRequested = Signal(int)  # noqa: N815
    loadDatasetRequested = Signal()  # noqa: N815

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(12)

        step1 = self._create_load_model_group()
        step2 = self._create_build_group()
        step3 = self._create_train_group()

        layout.addWidget(step1)
        layout.addWidget(step2)
        layout.addWidget(step3)
        layout.addStretch()

    def _create_load_model_group(self):
        group = QGroupBox("1. Load Model")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
        """)
        layout = QVBoxLayout(group)

        self._model_status = QLabel("No model loaded")
        self._model_status.setStyleSheet("color: #888;")
        layout.addWidget(self._model_status)

        self._load_model_btn = QPushButton("Load Model")
        self._load_model_btn.setToolTip("Load a transformer model (e.g., GPT-2)")
        self._load_model_btn.clicked.connect(self.loadDatasetRequested.emit)
        layout.addWidget(self._load_model_btn)

        return group

    def _create_build_group(self):
        group = QGroupBox("2. Build Transcoders")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
        """)
        layout = QVBoxLayout(group)

        form = QFormLayout()
        form.setSpacing(8)

        self._dict_size_combo = QComboBox()
        self._dict_size_combo.addItems(["1024", "2048", "4096", "8192", "16384"])
        self._dict_size_combo.setCurrentText("8192")
        self._dict_size_combo.setToolTip("Dictionary size for sparse autoencoder")

        self._top_k_combo = QComboBox()
        self._top_k_combo.addItems(["16", "32", "64", "128"])
        self._top_k_combo.setCurrentText("32")
        self._top_k_combo.setToolTip("Top-k sparsity parameter")

        form.addRow("Dictionary Size:", self._dict_size_combo)
        form.addRow("Top-K:", self._top_k_combo)
        layout.addLayout(form)

        self._build_btn = QPushButton("Build Transcoders")
        self._build_btn.setToolTip("Build sparse autoencoders for each layer")
        self._build_btn.clicked.connect(self.buildRequested.emit)
        layout.addWidget(self._build_btn)

        self._build_status = QLabel("")
        self._build_status.setStyleSheet("color: #888;")
        layout.addWidget(self._build_status)

        return group

    def _create_train_group(self):
        group = QGroupBox("3. Train")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3c3c3c;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
        """)
        layout = QVBoxLayout(group)

        form = QFormLayout()
        form.setSpacing(8)

        self._layer_combo = QComboBox()
        self._layer_combo.addItem("Layer 0")
        self._layer_combo.setToolTip("Which layer to train transcoder for")

        self._batch_size_combo = QComboBox()
        self._batch_size_combo.addItems(["4", "8", "16", "32"])
        self._batch_size_combo.setCurrentText("8")

        self._epochs_combo = QComboBox()
        self._epochs_combo.addItems(["1", "5", "10", "20"])
        self._epochs_combo.setCurrentText("10")

        form.addRow("Layer:", self._layer_combo)
        form.addRow("Batch Size:", self._batch_size_combo)
        form.addRow("Epochs:", self._epochs_combo)
        layout.addLayout(form)

        self._train_btn = QPushButton("Train")
        self._train_btn.setToolTip("Start training the transcoder")
        self._train_btn.setEnabled(False)
        self._train_btn.clicked.connect(self._on_train_clicked)
        layout.addWidget(self._train_btn)

        self._train_status = QLabel("")
        self._train_status.setStyleSheet("color: #888;")
        layout.addWidget(self._train_status)

        return group

    def _on_train_clicked(self):
        layer_idx = self._layer_combo.currentIndex()
        self.trainRequested.emit(layer_idx)

    def set_model_loaded(self, loaded: bool):
        self._model_status.setText("Model loaded ✓" if loaded else "No model loaded")
        self._model_status.setStyleSheet("color: #5cb85c;" if loaded else "color: #888;")

    def set_build_complete(self, num_layers: int):
        self._build_status.setText(f"Built {num_layers} transcoders ✓")
        self._build_status.setStyleSheet("color: #5cb85c;")
        self._train_btn.setEnabled(True)

    def set_training_status(self, status: str):
        self._train_status.setText(status)
        if "done" in status.lower() or "complete" in status.lower():
            self._train_status.setStyleSheet("color: #5cb85c;")
        elif "error" in status.lower() or "failed" in status.lower():
            self._train_status.setStyleSheet("color: #d9534f;")
        else:
            self._train_status.setStyleSheet("color: #f0ad4e;")

    def set_building(self, building: bool):
        self._build_btn.setEnabled(not building)
        if building:
            self._build_status.setText("Building...")
            self._build_status.setStyleSheet("color: #f0ad4e;")

    def set_training(self, training: bool):
        self._train_btn.setEnabled(not training)
        if training:
            self._train_status.setText("Training...")
            self._train_status.setStyleSheet("color: #f0ad4e;")

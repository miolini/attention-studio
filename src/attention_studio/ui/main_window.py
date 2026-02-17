from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger
from PySide6.QtCore import QObject, QSize, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from attention_studio.agents.agents import AgentConfig, AgentManager
from attention_studio.core.dataset import DatasetConfig, DatasetManager
from attention_studio.core.feature_extractor import (
    FeatureExtractor,
    GlobalCircuitAnalyzer,
    GraphBuilder,
)
from attention_studio.core.model_manager import ModelConfig, ModelManager
from attention_studio.core.trainer import CRMTrainer, TrainingConfig, TranscoderConfig
from attention_studio.ui.model_viz import ModelVisualizationWidget


class StudioMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Attention Studio")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)

        self.model_manager = ModelManager()
        self.dataset_manager = DatasetManager()
        self.trainer: CRMTrainer | None = None
        self.agent_manager: AgentManager | None = None

        self._left_sidebar_visible = True
        self._right_sidebar_visible = True
        self._bottom_pane_visible = True

        self._async_helpers: list[QWidget] = []

        self._setup_ui()
        self._setup_logging()
        self._apply_dark_theme()

    def _setup_logging(self):
        logger.add(
            sys.stderr,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        )

        def log_to_textedit(message: str):
            self.log_text.append(message)

        logger.add(
            log_to_textedit,
            format="{time:HH:mm:ss} | {level: <8} | {message}",
        )

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; color: #cccccc; }
            QWidget { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
            QMenuBar { background-color: #2d2d2d; color: #cccccc; border-bottom: 1px solid #3c3c3c; }
            QMenuBar::item { padding: 6px 12px; }
            QMenuBar::item:selected { background-color: #094771; }
            QMenu { background-color: #252526; color: #cccccc; border: 1px solid #3c3c3c; }
            QMenu::item { padding: 6px 24px 6px 12px; }
            QMenu::item:selected { background-color: #094771; }
            QToolBar { background-color: #2d2d2d; border: none; padding: 4px; spacing: 4px; }
            QToolButton { background-color: transparent; color: #cccccc; border: none; padding: 6px; border-radius: 4px; }
            QToolButton:hover { background-color: #3c3c3c; }
            QDockWidget { color: #cccccc; border: 1px solid #3c3c3c; }
            QDockWidget::title { background-color: #2d2d2d; padding: 6px 8px; border-bottom: 1px solid #3c3c3c; }
            QPushButton { background-color: #0e639c; color: white; border: none; padding: 6px 14px; border-radius: 3px; font-size: 12px; }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #0d5a8f; }
            QPushButton:disabled { background-color: #3d3d3d; color: #6d6d6d; }
            QLineEdit, QTextEdit, QSpinBox, QComboBox { background-color: #3c3c3c; color: #cccccc; border: 1px solid #555; padding: 5px 8px; border-radius: 3px; font-size: 12px; }
            QLineEdit:focus, QTextEdit:focus { border: 1px solid #0e639c; }
            QLabel { color: #cccccc; font-size: 12px; }
            QGroupBox { color: #cccccc; border: 1px solid #3c3c3c; border-radius: 4px; margin-top: 8px; padding-top: 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
            QTabWidget::pane { border: 1px solid #3c3c3c; background-color: #1e1e1e; }
            QTabBar::tab { background-color: #2d2d2d; color: #cccccc; padding: 8px 16px; border: 1px solid #3c3c3c; border-bottom: none; }
            QTabBar::tab:selected { background-color: #1e1e1e; border-bottom: 2px solid #0e639c; }
            QTabBar::tab:hover { background-color: #3c3c3c; }
            QTreeWidget, QListWidget, QTableWidget { background-color: #252526; color: #cccccc; border: 1px solid #3c3c3c; outline: none; }
            QTreeWidget::item:selected, QListWidget::item:selected, QTableWidget::item:selected { background-color: #094771; }
            QHeaderView::section { background-color: #2d2d2d; color: #cccccc; padding: 6px; border: 1px solid #3c3c3c; font-weight: 600; }
            QProgressBar { border: 1px solid #3c3c3c; border-radius: 3px; background-color: #3c3c3c; text-align: center; }
            QProgressBar::chunk { background-color: #0e639c; }
            QSplitter::handle { background-color: #3c3c3c; }
            QScrollBar:vertical { background-color: #1e1e1e; width: 12px; border: none; }
            QScrollBar::handle:vertical { background-color: #424242; border-radius: 6px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background-color: #4e4e4e; }
            QScrollBar:horizontal { background-color: #1e1e1e; height: 12px; border: none; }
            QScrollBar::handle:horizontal { background-color: #424242; border-radius: 6px; min-width: 20px; }
            QStatusBar { background-color: #007acc; color: white; border-top: 1px solid #3c3c3c; }
            QStatusBar::item { border: none; }
            QSlider::groove:horizontal { height: 4px; background: #3c3c3c; border-radius: 2px; }
            QSlider::handle:horizontal { width: 14px; margin: -5px 0; border-radius: 7px; background: #0e639c; }
            QSlider::sub-page:horizontal { background: #0e639c; border-radius: 2px; }
        """)

    def _setup_ui(self):
        self._create_menu_bar()
        self._create_toolbar()
        self._create_central_area()
        self._create_left_sidebar()
        self._create_right_sidebar()
        self._create_bottom_pane()
        self._create_status_bar()

    def _create_menu_bar(self):
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        open_action = file_menu.addAction("Open Model...", self._on_open_model)
        open_action.setShortcut("Ctrl+O")
        dataset_action = file_menu.addAction("Load Dataset...", self._on_load_dataset)
        dataset_action.setShortcut("Ctrl+D")
        file_menu.addSeparator()
        save_action = file_menu.addAction("Save Workflow...")
        save_action.setShortcut("Ctrl+S")
        load_action = file_menu.addAction("Load Workflow...")
        load_action.setShortcut("Ctrl+Shift+S")
        file_menu.addSeparator()
        prefs_action = file_menu.addAction("Preferences...", self._on_preferences)
        prefs_action.setShortcut("Ctrl+,")
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit", self.close)
        exit_action.setShortcut("Ctrl+Q")

        edit_menu = menubar.addMenu("Edit")
        undo_action = edit_menu.addAction("Undo")
        undo_action.setShortcut("Ctrl+Z")
        redo_action = edit_menu.addAction("Redo")
        redo_action.setShortcut("Ctrl+Shift+Z")
        edit_menu.addSeparator()
        cut_action = edit_menu.addAction("Cut")
        cut_action.setShortcut("Ctrl+X")
        copy_action = edit_menu.addAction("Copy")
        copy_action.setShortcut("Ctrl+C")
        paste_action = edit_menu.addAction("Paste")
        paste_action.setShortcut("Ctrl+V")

        view_menu = menubar.addMenu("View")
        view_menu.addAction("Toggle Left Sidebar", self.toggle_left_sidebar)
        view_menu.addAction("Toggle Right Sidebar", self.toggle_right_sidebar)
        view_menu.addAction("Toggle Bottom Pane", self.toggle_bottom_pane)
        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("Theme")
        dark_action = theme_menu.addAction("Dark", lambda: self._set_theme("dark"))
        dark_action.setShortcut("Ctrl+Alt+D")
        light_action = theme_menu.addAction("Light", lambda: self._set_theme("light"))
        light_action.setShortcut("Ctrl+Alt+L")

        model_menu = menubar.addMenu("Model")
        load_action = model_menu.addAction("Load Model...", self._on_load_model)
        load_action.setShortcut("Ctrl+L")
        unload_action = model_menu.addAction("Unload Model", self._on_unload_model)
        unload_action.setShortcut("Ctrl+U")

        training_menu = menubar.addMenu("Training")
        build_action = training_menu.addAction("Build CRM", self._on_build_crm)
        build_action.setShortcut("Ctrl+B")
        train_action = training_menu.addAction("Train Transcoder", self._on_train)
        train_action.setShortcut("Ctrl+T")

    def _set_theme(self, theme: str):
        if theme == "dark":
            self._apply_dark_theme()
        else:
            self._apply_light_theme()

    def _apply_light_theme(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #f5f5f5; color: #333; }
            QWidget { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }
            QMenuBar { background-color: #e0e0e0; color: #333; border-bottom: 1px solid #ccc; }
            QMenuBar::item { padding: 6px 12px; }
            QMenuBar::item:selected { background-color: #d0d0d0; }
            QMenu { background-color: #fff; color: #333; border: 1px solid #ccc; }
            QMenu::item { padding: 6px 24px 6px 12px; }
            QMenu::item:selected { background-color: #e0e0e0; }
            QToolBar { background-color: #e0e0e0; border: none; padding: 4px; spacing: 4px; }
            QToolButton { background-color: transparent; color: #333; border: none; padding: 6px; border-radius: 4px; }
            QToolButton:hover { background-color: #d0d0d0; }
            QDockWidget { color: #333; border: 1px solid #ccc; }
            QDockWidget::title { background-color: #e0e0e0; padding: 6px 8px; border-bottom: 1px solid #ccc; }
            QPushButton { background-color: #0e639c; color: white; border: none; padding: 6px 14px; border-radius: 3px; font-size: 12px; }
            QPushButton:hover { background-color: #1177bb; }
            QPushButton:pressed { background-color: #0d5a8f; }
            QPushButton:disabled { background-color: #ccc; color: #888; }
            QLineEdit, QTextEdit, QSpinBox, QComboBox { background-color: #fff; color: #333; border: 1px solid #ccc; padding: 5px 8px; border-radius: 3px; font-size: 12px; }
            QLineEdit:focus, QTextEdit:focus { border: 1px solid #0e639c; }
            QLabel { color: #333; font-size: 12px; }
            QGroupBox { color: #333; border: 1px solid #ccc; border-radius: 4px; margin-top: 8px; padding-top: 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 6px; }
            QTabWidget::pane { border: 1px solid #ccc; background-color: #f5f5f5; }
            QTabBar::tab { background-color: #e0e0e0; color: #333; padding: 8px 16px; border: 1px solid #ccc; border-bottom: none; }
            QTabBar::tab:selected { background-color: #f5f5f5; border-bottom: 2px solid #0e639c; }
            QTabBar::tab:hover { background-color: #d0d0d0; }
            QTreeWidget, QListWidget, QTableWidget { background-color: #fff; color: #333; border: 1px solid #ccc; outline: none; }
            QTreeWidget::item:selected, QListWidget::item:selected, QTableWidget::item:selected { background-color: #cce5ff; }
            QHeaderView::section { background-color: #e0e0e0; color: #333; padding: 6px; border: 1px solid #ccc; font-weight: 600; }
            QProgressBar { border: 1px solid #ccc; border-radius: 3px; background-color: #fff; text-align: center; }
            QProgressBar::chunk { background-color: #0e639c; }
            QSplitter::handle { background-color: #ccc; }
            QScrollBar:vertical { background-color: #f5f5f5; width: 12px; border: none; }
            QScrollBar::handle:vertical { background-color: #aaa; border-radius: 6px; min-height: 20px; }
            QScrollBar:horizontal { background-color: #f5f5f5; height: 12px; border: none; }
            QScrollBar::handle:horizontal { background-color: #aaa; border-radius: 6px; min-width: 20px; }
            QStatusBar { background-color: #007acc; color: white; border-top: 1px solid #ccc; }
            QStatusBar::item { border: none; }
        """)

    def _create_toolbar(self):
        toolbar = QToolBar("Main Toolbar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(20, 20))
        toolbar.setFixedHeight(40)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        self._load_model_btn = QToolButton()
        self._load_model_btn.setText("Load")
        self._load_model_btn.setToolTip("Load Model")
        self._load_model_btn.clicked.connect(self._on_load_model)
        toolbar.addWidget(self._load_model_btn)

        toolbar.addSeparator()

        self._build_crm_btn = QToolButton()
        self._build_crm_btn.setText("CRM")
        self._build_crm_btn.setToolTip("Build CRM")
        self._build_crm_btn.clicked.connect(self._on_build_crm)
        toolbar.addWidget(self._build_crm_btn)

        self._train_btn = QToolButton()
        self._train_btn.setText("Train")
        self._train_btn.setToolTip("Train")
        self._train_btn.clicked.connect(self._on_train)
        toolbar.addWidget(self._train_btn)

        toolbar.addSeparator()

        self._sidebar_toggle_left = QToolButton()
        self._sidebar_toggle_left.setText("Left")
        self._sidebar_toggle_left.setToolTip("Toggle Left Sidebar")
        self._sidebar_toggle_left.clicked.connect(self.toggle_left_sidebar)
        toolbar.addWidget(self._sidebar_toggle_left)

        self._sidebar_toggle_right = QToolButton()
        self._sidebar_toggle_right.setText("Right")
        self._sidebar_toggle_right.setToolTip("Toggle Right Sidebar")
        self._sidebar_toggle_right.clicked.connect(self.toggle_right_sidebar)
        toolbar.addWidget(self._sidebar_toggle_right)

        toolbar.addWidget(QLabel("  "))

        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Search...")
        self._search_box.setFixedWidth(250)
        toolbar.addWidget(self._search_box)

    def _create_central_area(self):
        central = QWidget()
        self.setCentralWidget(central)

        central_layout = QVBoxLayout(central)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)

        self._main_tabs = QTabWidget()
        self._main_tabs.setDocumentMode(True)
        self._main_tabs.setMovable(True)

        self._viz_tab = ModelVisualizationWidget()
        self._main_tabs.addTab(self._viz_tab, "Model Visualization")

        self._graph_tab = self._create_graph_tab()
        self._main_tabs.addTab(self._graph_tab, "Attribution Graph")

        self._features_tab = self._create_features_tab()
        self._main_tabs.addTab(self._features_tab, "Features")

        central_layout.addWidget(self._main_tabs)

    def _create_graph_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)

        self._graph_prompt_edit = QLineEdit()
        self._graph_prompt_edit.setPlaceholderText("Enter prompt to analyze...")
        self._graph_prompt_edit.setText("The quick brown fox jumps over the lazy dog")
        toolbar_layout.addWidget(self._graph_prompt_edit)

        self._graph_build_crm_btn2 = QPushButton("Build Graph")
        self._graph_build_crm_btn2.clicked.connect(self._on_build_attribution_graph)
        toolbar_layout.addWidget(self._graph_build_crm_btn2)

        self._graph_find_circuits_btn = QPushButton("Find Circuits")
        self._graph_find_circuits_btn.clicked.connect(self._on_find_circuits)
        self._graph_find_circuits_btn.setEnabled(False)
        toolbar_layout.addWidget(self._graph_find_circuits_btn)

        toolbar_layout.addStretch()

        layout.addWidget(toolbar)

        self._graph_scene = QGraphicsScene()
        self._graph_view = QGraphicsView(self._graph_scene)
        self._graph_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._graph_view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self._graph_view.setBackgroundBrush(QBrush(QColor("#1e1e1e")))
        layout.addWidget(self._graph_view)

        self._graph_info_label = QLabel("Enter a prompt and click 'Build Graph' to visualize feature attribution")
        self._graph_info_label.setStyleSheet("color: #808080; padding: 8px;")
        self._graph_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._graph_info_label)

        return tab

    def _create_features_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(8, 8, 8, 8)

        toolbar = QWidget()
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(0, 0, 0, 0)

        self._feat_prompt_edit = QLineEdit()
        self._feat_prompt_edit.setPlaceholderText("Enter prompt to analyze...")
        self._feat_prompt_edit.setText("The quick brown fox")
        toolbar_layout.addWidget(self._feat_prompt_edit)

        self._feat_layer_combo = QComboBox()
        self._feat_layer_combo.setMinimumWidth(100)
        toolbar_layout.addWidget(QLabel("Layer:"))
        toolbar_layout.addWidget(self._feat_layer_combo)

        self._feat_extract_btn = QPushButton("Extract Features")
        self._feat_extract_btn.clicked.connect(self._on_extract_features_ui)
        toolbar_layout.addWidget(self._feat_extract_btn)

        self._feat_circuit_btn = QPushButton("Analyze Circuits")
        self._feat_circuit_btn.clicked.connect(self._on_analyze_feature_circuits)
        self._feat_circuit_btn.setEnabled(False)
        toolbar_layout.addWidget(self._feat_circuit_btn)

        toolbar_layout.addStretch()

        layout.addWidget(toolbar)

        self._features_table = QTableWidget()
        self._features_table.setColumnCount(6)
        self._features_table.setHorizontalHeaderLabels(["Feature ID", "Layer", "Activation", "Norm", "Top Token", "Context"])
        self._features_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self._features_table.setStyleSheet("""
            QTableWidget { background-color: #1e1e1e; gridline-color: #3c3c3c; }
            QTableWidget::item { padding: 4px; }
            QHeaderView::section { background-color: #2d2d2d; padding: 6px; border: 1px solid #3c3c3c; }
        """)
        self._features_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._features_table)

        self._feat_details_text = QTextEdit()
        self._feat_details_text.setMaximumHeight(150)
        self._feat_details_text.setReadOnly(True)
        self._feat_details_text.setStyleSheet("QTextEdit { background-color: #252526; color: #cccccc; border: 1px solid #3c3c3c; }")
        layout.addWidget(self._feat_details_text)

        return tab

    def _create_left_sidebar(self):
        self._left_dock = QDockWidget("Explorer", self)
        self._left_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea)
        self._left_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self._left_dock.setFixedWidth(280)

        left_widget = QWidget()
        left_widget.setStyleSheet("background-color: #252526;")
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        self._left_tabs = QTabWidget()
        self._left_tabs.setDocumentMode(True)

        self.model_tree = QTreeWidget()
        self.model_tree.setHeaderLabel("Model Layers")
        self.model_tree.setStyleSheet("QTreeWidget { background-color: #252526; border: none; padding: 4px; }")
        self.model_tree.itemClicked.connect(self._on_layer_clicked)
        self._left_tabs.addTab(self.model_tree, "Layers")

        self.features_list = QListWidget()
        self.features_list.setStyleSheet("QListWidget { background-color: #252526; border: none; padding: 4px; }")
        self._left_tabs.addTab(self.features_list, "Features")

        left_layout.addWidget(self._left_tabs)

        search_box = QLineEdit()
        search_box.setPlaceholderText("Search...")
        search_box.setStyleSheet("QLineEdit { background-color: #3c3c3c; border: 1px solid #3c3c3c; border-radius: 4px; padding: 6px 8px; margin: 4px; }")
        left_layout.addWidget(search_box)

        self._left_dock.setWidget(left_widget)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self._left_dock)

    def _create_right_sidebar(self):
        self._right_dock = QDockWidget("Inspector", self)
        self._right_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self._right_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self._right_dock.setFixedWidth(320)

        right_widget = QScrollArea()
        right_widget.setWidgetResizable(True)
        right_widget.setStyleSheet("background-color: #252526; border: none;")
        right_content = QWidget()
        right_content.setStyleSheet("background-color: #252526;")
        right_layout = QVBoxLayout(right_content)
        right_layout.setContentsMargins(8, 8, 8, 8)
        right_layout.setSpacing(8)

        self._right_tabs = QTabWidget()
        self._right_tabs.setDocumentMode(True)

        model_section = QWidget()
        model_layout = QFormLayout()

        self._model_combo = QComboBox()
        self._model_combo.addItems(["gpt2", "gpt2-medium", "distilgpt2", "Qwen/Qwen3-0.6B"])
        model_layout.addRow("Model:", self._model_combo)

        self._device_combo = QComboBox()
        self._device_combo.addItems(["auto", "mps", "cuda", "cpu"])
        model_layout.addRow("Device:", self._device_combo)

        self._dtype_combo = QComboBox()
        self._dtype_combo.addItems(["float16", "float32", "bfloat16"])
        model_layout.addRow("Dtype:", self._dtype_combo)

        self._load_btn = QPushButton("Load Model")
        self._load_btn.clicked.connect(self._on_load_model)
        model_layout.addRow("", self._load_btn)

        self._model_status = QLabel("Not loaded")
        self._model_status.setStyleSheet("color: #808080;")
        model_layout.addRow("Status:", self._model_status)

        model_section.setLayout(model_layout)
        self._right_tabs.addTab(model_section, "Model")

        training_section = QWidget()
        training_layout = QFormLayout()

        self._dict_size_combo = QComboBox()
        self._dict_size_combo.addItems(["8192", "16384", "32768", "65536"])
        self._dict_size_combo.setEditable(True)
        training_layout.addRow("Dictionary Size:", self._dict_size_combo)

        self._top_k_combo = QComboBox()
        self._top_k_combo.addItems(["32", "64", "128", "256"])
        training_layout.addRow("Top K:", self._top_k_combo)

        self._batch_size_combo = QComboBox()
        self._batch_size_combo.addItems(["2", "4", "8", "16"])
        training_layout.addRow("Batch Size:", self._batch_size_combo)

        self._lr_edit = QLineEdit("1e-4")
        training_layout.addRow("Learning Rate:", self._lr_edit)

        self._epochs_spin = QSpinBox()
        self._epochs_spin.setRange(1, 100)
        self._epochs_spin.setValue(10)
        training_layout.addRow("Epochs:", self._epochs_spin)

        self._build_crm_btn2 = QPushButton("Build CRM")
        self._build_crm_btn2.clicked.connect(self._on_build_crm)
        training_layout.addRow("", self._build_crm_btn2)

        self._train_btn2 = QPushButton("Train")
        self._train_btn2.clicked.connect(self._on_train)
        self._train_btn2.setEnabled(False)
        training_layout.addRow("", self._train_btn2)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(8)
        training_layout.addRow("Progress:", self._progress_bar)

        training_section.setLayout(training_layout)
        self._right_tabs.addTab(training_section, "Training")

        analysis_section = QWidget()
        analysis_layout = QVBoxLayout()

        prompt_group = QGroupBox("Prompt")
        prompt_layout = QVBoxLayout()

        self._prompt_edit = QTextEdit()
        self._prompt_edit.setPlaceholderText("Enter prompt for analysis...")
        self._prompt_edit.setMaximumHeight(80)
        prompt_layout.addWidget(self._prompt_edit)

        analyze_btn_row = QHBoxLayout()
        self._analyze_btn = QPushButton("Analyze")
        self._analyze_btn.clicked.connect(self._on_extract_features)
        analyze_btn_row.addWidget(self._analyze_btn)
        analyze_btn_row.addStretch()
        prompt_layout.addLayout(analyze_btn_row)

        prompt_group.setLayout(prompt_layout)
        analysis_layout.addWidget(prompt_group)

        dataset_group = QGroupBox("Dataset")
        dataset_layout = QFormLayout()

        self._dataset_combo = QComboBox()
        self._dataset_combo.addItems(["custom", "c4"])
        dataset_layout.addRow("Source:", self._dataset_combo)

        self._token_limit_combo = QComboBox()
        self._token_limit_combo.addItems(["10000", "50000", "100000"])
        self._token_limit_combo.setEditable(True)
        dataset_layout.addRow("Token Limit:", self._token_limit_combo)

        self._load_dataset_btn2 = QPushButton("Load Dataset")
        self._load_dataset_btn2.clicked.connect(self._on_load_dataset)
        dataset_layout.addRow("", self._load_dataset_btn2)

        self._dataset_status = QLabel("Not loaded")
        self._dataset_status.setStyleSheet("color: #808080;")
        dataset_layout.addRow("Status:", self._dataset_status)

        dataset_group.setLayout(dataset_layout)
        analysis_layout.addWidget(dataset_group)

        agents_group = QGroupBox("Agents")
        agents_layout = QFormLayout()

        self._api_key_edit = QLineEdit()
        self._api_key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._api_key_edit.setPlaceholderText("API Key...")
        agents_layout.addRow("API Key:", self._api_key_edit)

        self._agent_model_combo = QComboBox()
        self._agent_model_combo.addItems(["minimax-minimax"])
        agents_layout.addRow("Model:", self._agent_model_combo)

        self._init_agents_btn = QPushButton("Initialize")
        self._init_agents_btn.clicked.connect(self._on_init_agents)
        agents_layout.addRow("", self._init_agents_btn)

        self._agent_status = QLabel("Not initialized")
        self._agent_status.setStyleSheet("color: #808080;")
        agents_layout.addRow("Status:", self._agent_status)

        agents_group.setLayout(agents_layout)
        analysis_layout.addWidget(agents_group)

        analysis_section.setLayout(analysis_layout)
        self._right_tabs.addTab(analysis_section, "Analysis")

        right_layout.addWidget(self._right_tabs)
        right_widget.setWidget(right_content)

        self._right_dock.setWidget(right_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self._right_dock)

    def _create_bottom_pane(self):
        self._bottom_dock = QDockWidget("Console", self)
        self._bottom_dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)
        self._bottom_dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetClosable | QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self._bottom_dock.setFixedHeight(200)

        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)

        self._bottom_tabs = QTabWidget()
        self._bottom_tabs.setDocumentMode(True)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("QTextEdit { background-color: #1e1e1e; color: #cccccc; font-family: 'SF Mono', monospace; font-size: 11px; border: none; padding: 8px; }")
        self._bottom_tabs.addTab(self.log_text, "Console")

        output_group = QWidget()
        output_layout = QVBoxLayout(output_group)
        output_layout.setContentsMargins(0, 0, 0, 0)

        self.output_table = QTableWidget()
        self.output_table.setColumnCount(5)
        self.output_table.setHorizontalHeaderLabels(["ID", "Layer", "Type", "Activation", "Norm"])
        self.output_table.horizontalHeader().setStretchLastSection(True)
        self.output_table.setStyleSheet("QTableWidget { background-color: #1e1e1e; border: none; }")
        output_layout.addWidget(self.output_table)

        self._bottom_tabs.addTab(output_group, "Features")

        bottom_layout.addWidget(self._bottom_tabs)

        self._bottom_dock.setWidget(bottom_widget)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self._bottom_dock)

    def _create_status_bar(self):
        self._status_bar = QStatusBar()
        self._status_bar.setStyleSheet("QStatusBar { background-color: #007acc; color: white; padding: 4px 8px; }")
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Ready")

        self._status_label = QLabel()
        self._status_label.setStyleSheet("color: white;")
        self._status_bar.addPermanentWidget(self._status_label)

    def toggle_left_sidebar(self):
        self._left_sidebar_visible = not self._left_sidebar_visible
        if self._left_sidebar_visible:
            self._left_dock.show()
            self._sidebar_toggle_left.setText("Left")
        else:
            self._left_dock.hide()
            self._sidebar_toggle_left.setText("Right")

    def toggle_right_sidebar(self):
        self._right_sidebar_visible = not self._right_sidebar_visible
        if self._right_sidebar_visible:
            self._right_dock.show()
            self._sidebar_toggle_right.setText("Right")
        else:
            self._right_dock.hide()
            self._sidebar_toggle_right.setText("Left")

    def toggle_bottom_pane(self):
        self._bottom_pane_visible = not self._bottom_pane_visible
        if self._bottom_pane_visible:
            self._bottom_dock.show()
        else:
            self._bottom_dock.hide()

    def _on_open_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Model", "", "Model Files (*.bin *.safetensors *.gguf);;All Files (*)")
        if file_path:
            self.log_text.append(f"Opening model from: {file_path}")

    def _on_open_dataset_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Dataset", "", "Dataset Files (*.jsonl *.csv *.txt);;All Files (*)")
        if file_path:
            self.log_text.append(f"Loading dataset from: {file_path}")

    def _on_preferences(self):
        self.log_text.append("Opening preferences...")

    def _on_load_model(self):
        config = ModelConfig(
            name=self._model_combo.currentText(),
            device=self._device_combo.currentText(),
            dtype=self._dtype_combo.currentText(),
        )

        self._load_btn.setEnabled(False)
        self._model_status.setText("Loading...")
        self._status_bar.showMessage(f"Loading model: {config.name}")

        def load():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.model_manager.load_model(config))

        def on_done():
            try:
                self._load_btn.setEnabled(True)
                info = self.model_manager.get_model_info()
                self._model_status.setText(f"Loaded ({info.get('num_parameters', 0) // 1_000_000}M params)")
                self._update_model_tree()
                num_layers = getattr(self.model_manager.model.config, "n_layer", None) or \
                            getattr(self.model_manager.model.config, "num_hidden_layers", 12)
                self._main_tabs.setCurrentIndex(0)
                self._viz_tab.update_model(self.model_manager, list(range(num_layers)))
                self._status_bar.showMessage(f"Model loaded: {config.name}")
                self._status_label.setText(f"Model: {config.name}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.log_text.append(f"Error in on_done: {e}")

        def on_error(err):
            self._load_btn.setEnabled(True)
            self._model_status.setText(f"Error: {err}")
            self.log_text.append(f"Error loading model: {err}")

        self._run_async(load, on_done, on_error)

    def _on_unload_model(self):
        self.model_manager.unload()
        self._model_status.setText("Not loaded")
        self._update_model_tree()
        self._status_bar.showMessage("Model unloaded")

    def _update_model_tree(self):
        self.model_tree.clear()
        root = QTreeWidgetItem(["Model"])
        root.setExpanded(True)

        for layer in self.model_manager.layer_info:
            layer_item = QTreeWidgetItem([f"Layer {layer.idx}: {layer.type}"])
            layer_item.setData(0, Qt.ItemDataRole.UserRole, layer.idx)
            root.addChild(layer_item)

        self.model_tree.addTopLevelItem(root)

    def _on_layer_clicked(self, item, column):
        layer_idx = item.data(0, Qt.ItemDataRole.UserRole)
        if layer_idx is not None:
            self._status_bar.showMessage(f"Selected layer {layer_idx}")

    def _on_load_dataset(self):
        source = self._dataset_combo.currentText()
        custom_path = None

        if source == "custom":
            custom_path = Path(__file__).parent.parent / "data" / "train.txt"
            if not custom_path.exists():
                custom_path = Path.cwd() / "data" / "train.txt"

        config = DatasetConfig(
            source=source,
            token_limit=int(self._token_limit_combo.currentText()),
            custom_path=custom_path,
        )

        self._load_dataset_btn2.setEnabled(False)
        self._dataset_status.setText("Loading...")
        self._status_bar.showMessage("Loading dataset...")

        def load():
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.dataset_manager.load(config, self.model_manager.tokenizer))

        def on_done():
            self._load_dataset_btn2.setEnabled(True)
            stats = self.dataset_manager.get_stats()
            self._dataset_status.setText(f"Loaded ({stats['num_texts']} texts)")
            self._status_bar.showMessage("Dataset loaded")

        def on_error(err):
            self._load_dataset_btn2.setEnabled(True)
            self._dataset_status.setText(f"Error: {err}")

        self._run_async(load, on_done, on_error)

    def _on_build_crm(self):
        if not self.model_manager.is_loaded:
            self.log_text.append("Model not loaded")
            return

        self._build_crm_btn2.setEnabled(False)
        self._status_bar.showMessage("Building CRM...")

        def build():
            dataset_loaded = False
            if not self.dataset_manager or not hasattr(self.dataset_manager, '_texts') or not self.dataset_manager._texts:
                custom_path = Path(__file__).parent.parent / "data" / "train.txt"
                if not custom_path.exists():
                    custom_path = Path.cwd() / "data" / "train.txt"

                config = DatasetConfig(
                    source="custom",
                    token_limit=10000,
                    custom_path=custom_path,
                    format="txt",
                )

                import asyncio
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.dataset_manager.load(config, self.model_manager.tokenizer))
                dataset_loaded = True
            return dataset_loaded

        def on_done(dataset_loaded):
            config = TranscoderConfig(
                dictionary_size=int(self._dict_size_combo.currentText()),
                top_k=int(self._top_k_combo.currentText()),
            )

            model_config = self.model_manager.model.config
            num_layers = getattr(model_config, "n_layer", None) or getattr(model_config, "num_hidden_layers", None)

            if num_layers is None:
                self.log_text.append("Cannot determine number of layers")
                self._build_crm_btn2.setEnabled(True)
                return

            self.trainer = CRMTrainer(self.model_manager, TrainingConfig())
            self.trainer.build_transcoders(list(range(num_layers)), config)

            self._train_btn2.setEnabled(True)
            self._feat_layer_combo.clear()
            for i in range(num_layers):
                self._feat_layer_combo.addItem(f"Layer {i}")

            if dataset_loaded:
                self.log_text.append(f"Dataset loaded ({self.dataset_manager.get_stats()['num_texts']} texts)")
            self.log_text.append(f"Built {num_layers} transcoders")
            self._status_bar.showMessage("CRM built")
            self._build_crm_btn2.setEnabled(True)

        def on_error(err):
            self._build_crm_btn2.setEnabled(True)
            self.log_text.append(f"Build error: {err}")

        self._run_async(build, on_done, on_error)

    def _on_train(self):
        if not self.trainer:
            self.log_text.append("CRM not built - build transcoders first")
            return

        if not self.dataset_manager or not hasattr(self.dataset_manager, '_texts') or not self.dataset_manager._texts:
            self.log_text.append("No dataset loaded - please load a dataset first")
            return

        try:
            dataloader = self.dataset_manager.create_dataloader(
                batch_size=int(self._batch_size_combo.currentText()),
                max_length=512,
            )
        except RuntimeError as e:
            self.log_text.append(f"Dataset error: {e}")
            return

        self._train_btn2.setEnabled(False)
        self._status_bar.showMessage("Training...")

        layer_idx = 0

        def train():
            return self.trainer.train_transcoder(dataloader, layer_idx=layer_idx, progress_callback=None)

        def on_done(result):
            self._train_btn2.setEnabled(True)
            self._progress_bar.setValue(100)
            self.log_text.append(f"Training done: {result}")
            self._status_bar.showMessage("Training complete")

        def on_error(err):
            self._train_btn2.setEnabled(True)
            self._progress_bar.setValue(0)
            self.log_text.append(f"Training error: {err}")
            self._status_bar.showMessage("Training failed")

        self._run_async(train, on_done, on_error)

    def _on_extract_features(self):
        if not self.model_manager.is_loaded:
            self.log_text.append("Model not loaded")
            return

        if not self.trainer or not self.trainer.transcoders:
            self.log_text.append("Transcoder not trained")
            return

        prompt = self._prompt_edit.toPlainText() or "The quick brown fox"
        layer_idx = 0

        transcoder = self.trainer.get_transcoder(layer_idx)
        if transcoder is None:
            self.log_text.append(f"No transcoder for layer {layer_idx}")
            return

        def extract():
            extractor = FeatureExtractor(self.model_manager, transcoder, layer_idx)
            features = extractor.extract_features(prompt, top_k=20)
            return extractor, features

        def on_done(result):
            extractor, features = result
            self._current_extractor = extractor
            self._current_features = features

            self.output_table.setRowCount(len(features))
            for i, f in enumerate(features):
                self.output_table.setItem(i, 0, QTableWidgetItem(str(f.idx)))
                self.output_table.setItem(i, 1, QTableWidgetItem(str(f.layer)))
                self.output_table.setItem(i, 2, QTableWidgetItem("feature"))
                self.output_table.setItem(i, 3, QTableWidgetItem(f"{f.activation:.4f}"))
                self.output_table.setItem(i, 4, QTableWidgetItem(f"{f.norm:.4f}"))

            self.log_text.append(f"Extracted {len(features)} features")
            self._status_bar.showMessage(f"Extracted {len(features)} features")

        def on_error(err):
            import traceback
            self.log_text.append(f"Error extracting features: {err}")
            self.log_text.append(traceback.format_exc())

        self._run_async(extract, on_done, on_error)

    def _on_build_attribution_graph(self):
        if not self.model_manager.is_loaded:
            self.log_text.append("Model not loaded")
            return

        if not self.trainer or not self.trainer.transcoders:
            self.log_text.append("CRM not built - build transcoders first")
            return

        prompt = self._graph_prompt_edit.text() or "The quick brown fox"
        self._graph_build_crm_btn2.setEnabled(False)
        self._graph_info_label.setText("Building attribution graph...")

        def build():
            graph_builder = GraphBuilder(
                self.model_manager,
                self.trainer.transcoders,
                self.trainer.layer_indices,
            )
            graph = graph_builder.build_attribution_graph(prompt, threshold=0.01)
            return graph_builder, graph

        def on_done(result):
            self._graph_build_crm_btn2.setEnabled(True)
            builder, graph = result

            self._graph_builder = builder
            self._current_graph = graph

            stats = builder.get_graph_stats(graph)
            self._graph_info_label.setText(
                f"Graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges, density: {stats['density']:.4f}"
            )
            self._graph_find_circuits_btn.setEnabled(True)

            self._render_graph(graph)
            self.log_text.append(f"Built attribution graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")

        def on_error(err):
            self._graph_build_crm_btn2.setEnabled(True)
            self._graph_info_label.setText(f"Error: {err}")
            self.log_text.append(f"Error building graph: {err}")

        self._run_async(build, on_done, on_error)

    def _render_graph(self, graph):
        self._graph_scene.clear()

        if not graph.nodes():
            return

        import networkx as nx
        try:
            pos = nx.spring_layout(graph, k=2, iterations=50, seed=42)
        except Exception:
            pos = {n: (i % 10 * 100, i // 10 * 100) for i, n in enumerate(graph.nodes())}

        node_items = {}
        colors = {
            0: QColor(46, 204, 113),
            1: QColor(52, 152, 219),
            2: QColor(155, 89, 182),
            3: QColor(241, 196, 15),
            4: QColor(230, 126, 34),
        }

        for _i, node in enumerate(graph.nodes()):
            layer = graph.nodes[node].get("layer", 0)
            x, y = pos[node]
            x = x * 150 + 400
            y = y * 150 + 300

            color = colors.get(layer % 5, QColor(100, 100, 100))

            rect = self._graph_scene.addEllipse(x - 15, y - 15, 30, 30)
            rect.setBrush(QBrush(color))
            rect.setPen(QPen(color.darker(150), 2))

            label = self._graph_scene.addText(node[:12])
            label.setDefaultTextColor(QColor(220, 220, 230))
            label.setFont(QFont("SF Mono", 6))
            label.setPos(x - 10, y - 8)
            label.setZValue(10)

            node_items[node] = (x, y)

        for src, dst in graph.edges():
            if src in node_items and dst in node_items:
                x1, y1 = node_items[src]
                x2, y2 = node_items[dst]
                weight = graph[src][dst].get("weight", 1.0)

                line = self._graph_scene.addLine(x1 + 15, y1, x2 - 15, y2)
                line.setPen(QPen(QColor(100, 100, 120), max(0.5, abs(weight))))

        self._graph_view.fitInView(self._graph_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def _on_find_circuits(self):
        if not self.model_manager.is_loaded:
            self.log_text.append("Model not loaded")
            return

        if not self.trainer or not self.trainer.transcoders:
            self.log_text.append("CRM not built")
            return

        prompt = self._graph_prompt_edit.text() or "The quick brown fox"
        self._graph_find_circuits_btn.setEnabled(False)
        self._graph_info_label.setText("Finding global circuits...")

        def find():
            analyzer = GlobalCircuitAnalyzer(
                self.model_manager,
                self.trainer.transcoders,
                self.trainer.lorsas if hasattr(self.trainer, 'loras') else None,
                self.trainer.layer_indices if hasattr(self.trainer, 'layer_indices') else None,
            )
            circuits = analyzer.analyze_all_circuits(prompt)
            return circuits

        def on_done(circuits):
            self._graph_find_circuits_btn.setEnabled(True)

            if circuits:
                self._graph_info_label.setText(f"Found {len(circuits)} circuit types")
                for circuit_type, circuit_list in circuits.items():
                    self.log_text.append(f"  {circuit_type}: {len(circuit_list)} circuits")
            else:
                self._graph_info_label.setText("No significant circuits found")

        def on_error(err):
            self._graph_find_circuits_btn.setEnabled(True)
            self.log_text.append(f"Error finding circuits: {err}")

        self._run_async(find, on_done, on_error)

    def _on_extract_features_ui(self):
        if not self.model_manager.is_loaded:
            self.log_text.append("Model not loaded")
            return

        if not self.trainer or not self.trainer.transcoders:
            self.log_text.append("CRM not built - build transcoders first")
            return

        if self._feat_layer_combo.count() == 0:
            self.log_text.append("No layers available - build CRM first")
            return

        try:
            prompt = self._feat_prompt_edit.text() or "The quick brown fox"
            layer_idx = int(self._feat_layer_combo.currentText().replace("Layer ", ""))
        except (ValueError, AttributeError) as e:
            self.log_text.append(f"Invalid layer selection: {e}")
            return

        transcoder = self.trainer.get_transcoder(layer_idx)
        if transcoder is None:
            self.log_text.append(f"No transcoder for layer {layer_idx}")
            return

        self._feat_extract_btn.setEnabled(False)
        self.log_text.append(f"Extracting features from layer {layer_idx}...")

        def extract():
            model = self.model_manager.model
            model.eval()
            transcoder.eval()

            extractor = FeatureExtractor(self.model_manager, transcoder, layer_idx)
            features = extractor.extract_features(prompt, top_k=50)

            for feat in features[:10]:
                contexts = extractor.get_top_contexts(prompt, feat.idx, k=5)
                feat.top_contexts = contexts

            return extractor, features

        def on_done(result):
            extractor, features = result

            self._current_extractor = extractor
            self._current_features = features
            self._feat_circuit_btn.setEnabled(True)
            self._feat_extract_btn.setEnabled(True)

            self._features_table.setRowCount(len(features))
            for i, f in enumerate(features):
                self._features_table.setItem(i, 0, QTableWidgetItem(str(f.idx)))
                self._features_table.setItem(i, 1, QTableWidgetItem(str(f.layer)))
                self._features_table.setItem(i, 2, QTableWidgetItem(f"{f.activation:.4f}"))
                self._features_table.setItem(i, 3, QTableWidgetItem(f"{f.norm:.4f}"))

                context_str = ""
                if f.top_contexts:
                    top_ctx = f.top_contexts[0]
                    context_str = f"{top_ctx.get('token', '')}@{top_ctx.get('position', 0)}"
                self._features_table.setItem(i, 4, QTableWidgetItem(context_str))
                self._features_table.setItem(i, 5, QTableWidgetItem(str(len(f.top_contexts)) if f.top_contexts else "0"))

            self.log_text.append(f"Extracted {len(features)} features from layer {layer_idx}")

        def on_error(err):
            import traceback
            self._feat_extract_btn.setEnabled(True)
            self.log_text.append(f"Error extracting features: {err}")
            self.log_text.append(traceback.format_exc())

        self._run_async(extract, on_done, on_error)

    def _on_analyze_feature_circuits(self):
        if not hasattr(self, '_current_extractor') or not self._current_features:
            self.log_text.append("Extract features first")
            return

        row = self._features_table.currentRow()
        if row < 0:
            self.log_text.append("Select a feature from the table")
            return

        feature = self._current_features[row]
        layer_idx = feature.layer
        feature_idx = feature.idx

        self._feat_circuit_btn.setEnabled(False)
        self.log_text.append(f"Analyzing circuit for feature {feature_idx} in layer {layer_idx}...")

        def analyze():
            analyzer = GlobalCircuitAnalyzer(
                self.model_manager,
                self.trainer.transcoders,
                self.trainer.lorsas if hasattr(self.trainer, 'loras') else None,
                self.trainer.layer_indices if hasattr(self.trainer, 'layer_indices') else None,
            )
            circuit_info = analyzer.compute_feature_circuits(layer_idx, feature_idx)
            return circuit_info

        def on_done(circuit_info):
            self._feat_circuit_btn.setEnabled(True)

            details = f"""Feature Circuit Analysis
========================
Layer: {circuit_info['layer_idx']}
Feature ID: {circuit_info['feature_idx']}
Decoder Norm: {circuit_info['norm']:.4f}
Encoder Norm: {circuit_info['encoder_norm']:.4f}
QK Circuit: {'Available' if circuit_info['qk_circuit'] else 'Not available'}
OV Circuit: {'Available' if circuit_info['ov_circuit'] else 'Not available'}
"""
            self._feat_details_text.setText(details)
            self.log_text.append(f"Analyzed circuit for feature {feature_idx} in layer {layer_idx}")

        def on_error(err):
            import traceback
            self._feat_circuit_btn.setEnabled(True)
            self.log_text.append(f"Error analyzing circuit: {err}")
            self.log_text.append(traceback.format_exc())

        self._run_async(analyze, on_done, on_error)

    def _on_init_agents(self):
        if not self._api_key_edit.text():
            self.log_text.append("Please enter API key")
            return

        config = AgentConfig(
            api_key=self._api_key_edit.text(),
            model=self._agent_model_combo.currentText(),
        )

        self.agent_manager = AgentManager(config)
        self._agent_status.setText("Initialized")
        self._init_agents_btn.setEnabled(False)
        self.log_text.append("Agents initialized")

    def _run_async(self, work_fn, done_fn, error_fn):
        class CallbackHelper(QObject):
            done_signal = Signal(object)
            error_signal = Signal(str)

        helper = CallbackHelper()
        self._async_helpers.append(helper)

        if done_fn:
            helper.done_signal.connect(done_fn, Qt.ConnectionType.QueuedConnection)
        if error_fn:
            helper.error_signal.connect(error_fn, Qt.ConnectionType.QueuedConnection)

        def worker():
            try:
                result = work_fn()
                helper.done_signal.emit(result)
            except Exception as e:
                if error_fn:
                    helper.error_signal.emit(str(e))

        import threading
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


def main():
    import sys
    app = QApplication(sys.argv)
    window = StudioMainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

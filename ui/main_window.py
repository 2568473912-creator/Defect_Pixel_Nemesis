import os
import sys
from pathlib import Path
import cv2
import numpy as np
import pyqtgraph as pg

# 🟢 1. 修复 'PyQt6' 未解析引用
# 你的代码中有 "v_splitter = PyQt6.QtWidgets.QSplitter(...)" 这种写法
# 所以必须导入 PyQt6 顶层包，或者建议直接用 QSplitter
import PyQt6.QtWidgets
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QSplitter, QGroupBox, QComboBox, QSpinBox, QApplication,
    QMessageBox, QFrame, QListWidget, QTableView, QHeaderView,
    QDialog,  # 🟢 修复 'QDialog' 未解析 (用于 QDialog.DialogCode.Accepted)
    QProgressBar, QTextEdit # 如果有遗漏也补上
)
from PyQt6.QtCore import Qt, QSettings, QSortFilterProxyModel, pyqtSignal
from PyQt6.QtGui import QIcon

# 🟢 2. 修复 'BatchWorker' 未解析引用
# 必须从核心层导入这些 Worker
from core.workers import SingleWorker, BatchWorker

# 🟢 3. 导入其他自定义模块 (确保这些文件已按上一条回答创建)
from utils.helpers import BASE_DIR, ExportHandler
from core.algorithm import CoreAlgorithm
from ui.widgets import (
    Surface3DViewer, InteractiveHistogram, ZoomableGraphicsView, DefectTableModel
)
from ui.dialogs import (
    SingleExportDialog, BatchProcessDialog, BatchCropDialog
)
# 1. 放入 CyberApp 类
# ==========================================
# 🖥️ UI (交互升级版)
# ==========================================
class CyberApp(QMainWindow):
    # [新增 1] 打开文件夹并加载文件列表
    def open_single_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not d: return

        self.current_single_dir = Path(d)
        self.file_list.clear()

        # 扫描常见图片格式
        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        files = []
        for ext in extensions:
            files.extend(list(self.current_single_dir.glob(ext)))
            # 也要支持大写后缀
            files.extend(list(self.current_single_dir.glob(ext.upper())))

        # 排序并添加到列表
        files = sorted(list(set(files)))  # 去重并排序

        if not files:
            self.file_list.addItem("No images found.")
            return

        for f in files:
            self.file_list.addItem(f.name)

        # 自动选中第一个并触发分析 (可选)
        # self.file_list.setCurrentRow(0)
        # self.on_file_list_clicked(self.file_list.item(0))

    # [新增 2] 列表点击回调
    def on_file_list_clicked(self, item):
        if not self.current_single_dir: return
        if item.text() == "No images found.": return

        filename = item.text()
        full_path = self.current_single_dir / filename

        # 调用核心分析逻辑
        self.trigger_analysis(str(full_path))

    # [新增 3] 重新分析当前图片 (用于参数调整后手动刷新)
    def re_analyze_current(self):
        if self.current_file_path and Path(self.current_file_path).exists():
            self.trigger_analysis(self.current_file_path)

    # [重构] 将原来的 run_single_analysis 拆分，核心逻辑提炼为 trigger_analysis
    def trigger_analysis(self, path):
        self.current_file_path = path

        # 设置忙碌光标
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)

        self.btn_load.setEnabled(False)
        self.btn_load.setText("PROCESSING...")
        if hasattr(self, 'file_list'):
            self.file_list.setEnabled(False)

        params = self.get_params()
        self.worker = SingleWorker(path, params)

        # 信号连接到刚刚修复的 wrapper
        self.worker.result_signal.connect(self.on_single_finished_wrapper)

        self.worker.start()

    # [包装器] 分析完成后，除了原来的逻辑，还要恢复列表可点状态
    def on_single_finished_wrapper(self, vis_raw, vis_grid, data, img_raw):
        # 必须传递所有 3 个参数：原图、网格图、数据
        # 恢复光标
        QApplication.restoreOverrideCursor()
        self.on_single_finished(vis_raw, vis_grid, data, img_raw)

        # 恢复 UI 状态
        self.btn_load.setText("🔄 RE-ANALYZE CURRENT")
        self.btn_load.setEnabled(True)
        # 确保列表恢复可用 (使用绝对路径防止报错)
        if hasattr(self, 'file_list'):
            self.file_list.setEnabled(True)


    def __init__(self):
        super().__init__()
        self.setWindowTitle("Defect Pixel Nemesis // V3.0 by Klay Wei")
        self.resize(1600, 900)


        # [新增] 3D 窗口实例 (初始隐藏)
        self.win_3d = Surface3DViewer()
        self.apply_theme()

        # [新增] 记录鼠标在图片上的最后已知位置
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        self.current_data_cache = []
        self.cursor_lines = []  # <---【新增】用于存储当前的十字准星


        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        # self.layout = QVBoxLayout(self.main_widget)
        # 直接调用单图界面初始化 (稍后我们会修改这个函数，让它不再创建Tab)
        self.init_single_mode()
        # 👇👇👇 [新增代码] 配置初始化 👇👇👇


        # 1. 指定使用 INI 格式，且路径为当前目录下的 config.ini
        # 这样用户可以直接用记事本打开修改参数
        ini_path = os.path.join(BASE_DIR, "config.ini")
        self.settings = QSettings(ini_path, QSettings.Format.IniFormat)

        # 启动时自动加载上次的参数
        self.load_settings()
        # 👆👆👆 [新增代码结束] 👆👆👆
        # 移除 init_batch_mode() 的调用

        self.setAcceptDrops(True)  # 允许拖放
        from PyQt6.QtCore import QTimer
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.setInterval(600)  # 600毫秒无操作后自动刷新
        self.debounce_timer.timeout.connect(self.re_analyze_current)

    # 🟢 [新增] 窗口关闭事件：自动保存参数到 config.ini
    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    # 🟢 [新增] 保存参数逻辑
    def save_settings(self):
        # 1. 保存通用参数
        self.settings.setValue("params/mode_idx", self.combo_mode.currentIndex())
        self.settings.setValue("params/ch_idx", self.cb_ch.currentIndex())
        self.settings.setValue("params/fs_idx", self.cb_fs.currentIndex())

        # 2. 保存暗场参数 (Dark)
        self.settings.setValue("params/dark/thresh", self.sb_thresh_abs.value())
        self.settings.setValue("params/dark/ch_dist", self.sb_ch_dist_dark.value())
        self.settings.setValue("params/dark/g_dist", self.sb_g_dist_dark.value())

        # 3. 保存亮场参数 (Bright)
        self.settings.setValue("params/bright/pct", self.sb_thresh_pct.value())
        self.settings.setValue("params/bright/ch_dist", self.sb_ch_dist_bright.value())
        self.settings.setValue("params/bright/g_dist", self.sb_g_dist_bright.value())

        # 4. 保存规格 (Specs)
        self.settings.setValue("specs/max_pts", self.sb_spec_pts.value())
        self.settings.setValue("specs/max_cls", self.sb_spec_cls.value())

        # 5. 保存上次打开的文件夹 (非常实用!)
        if self.current_single_dir:
            self.settings.setValue("paths/last_dir", str(self.current_single_dir))

    # 🟢 [新增] 导入参数逻辑
    def load_settings(self):
        # 辅助函数：安全读取 int，读不到就用默认值
        def get_int(key, default):
            try:
                val = self.settings.value(key, default)
                return int(val)
            except:
                return default

        # 1. 恢复通用参数
        self.combo_mode.setCurrentIndex(get_int("params/mode_idx", 0))  # 默认 Dark
        self.cb_ch.setCurrentIndex(get_int("params/ch_idx", 1))  # 默认 16ch
        self.cb_fs.setCurrentIndex(get_int("params/fs_idx", 1))  # 默认 5x5

        # 2. 恢复暗场参数
        self.sb_thresh_abs.setValue(get_int("params/dark/thresh", 50))
        self.sb_ch_dist_dark.setValue(get_int("params/dark/ch_dist", 3))
        self.sb_g_dist_dark.setValue(get_int("params/dark/g_dist", 5))

        # 3. 恢复亮场参数
        self.sb_thresh_pct.setValue(get_int("params/bright/pct", 30))
        self.sb_ch_dist_bright.setValue(get_int("params/bright/ch_dist", 3))
        self.sb_g_dist_bright.setValue(get_int("params/bright/g_dist", 5))

        # 4. 恢复规格
        self.sb_spec_pts.setValue(get_int("specs/max_pts", 100))
        self.sb_spec_cls.setValue(get_int("specs/max_cls", 0))

        # 5. 恢复上次路径 (自动跳转)
        last_dir = self.settings.value("paths/last_dir", "")
        if last_dir and os.path.exists(last_dir):
            self.current_single_dir = Path(last_dir)
            # 自动刷新文件列表
            self.file_list.clear()
            extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
            files = []
            for ext in extensions:
                files.extend(list(self.current_single_dir.glob(ext)))
                files.extend(list(self.current_single_dir.glob(ext.upper())))
            files = sorted(list(set(files)))
            if files:
                for f in files: self.file_list.addItem(f.name)
            else:
                self.file_list.addItem("No images found.")

        # 手动触发一次界面刷新 (确保 Dark/Bright 面板显示正确)
        self.toggle_params()
    def update_cursor_display(self, x, y, val):
        """
        接收鼠标移动信号，更新界面显示 + 3D地形
        🟢 [修改] 不再使用 val_ignored (它是8bit的)，而是直接去 cache_raw_img 查 16bit 值
        """
        # 1. 优先去读原始数据 (cache_raw_img)
        # [新增] 记录实时位置，供批量截图使用
        self.last_mouse_x = x
        self.last_mouse_y = y

        final_val = "N/A"
        val_view = val
        if hasattr(self, 'cache_raw_img') and self.cache_raw_img is not None:
            h, w = self.cache_raw_img.shape[:2]

            # 边界检查
            if 0 <= x < w and 0 <= y < h:
                # 读取原始值 (uint16)
                raw_pixel = self.cache_raw_img[y, x]

                # 处理多通道 (取最大值，保证抓到坏点)
                if self.cache_raw_img.ndim == 3:
                    raw_val = np.max(raw_pixel)
                else:
                    raw_val = raw_pixel

                # 🟢 [核心修改] 如果是 16-bit 数据，除以 256 显示
                if self.cache_raw_img.dtype == np.uint16:
                    final_val = int(raw_val / 256)
                else:
                    final_val = int(raw_val)
        else:
            # 如果没有原图缓存，被迫使用视图传来的值 (通常是缩略图的值，不太准)
            # 这里的 val_from_view 是字符串，尝试转一下
            try:
                # 去掉可能存在的括号 []
                val_str = str(val_view).replace('[', '').replace(']', '')
                final_val = int(float(val_str.split(',')[0]))  # 简单取第一个通道
            except:
                final_val = val_view

        # 2. 更新文字标签
        self.lbl_cursor_info.setText(f"📍 X: {x:<4} Y: {y:<4} 💡 Val: {final_val}")
        # [修改] 3D 地形实时更新逻辑
        if self.win_3d.isVisible():

            # 1. 优先使用纯净原图 (避免红框干扰)
            if hasattr(self, 'cache_clean_img') and self.cache_clean_img is not None:
                source_img = self.cache_clean_img
                # 如果当前是 Grid 视图，3D图的坐标会对应不上纯净原图
                # 这是一个逻辑冲突：Grid视图是拼贴的，原图是整张的。
                # 如果用户在 Grid 视图下看，我们其实很难对应回原图的 ROI，除非反算坐标。
                # 简单处理：
                # 如果在 Raw View -> 用 cache_clean_img (完美去除红框)
                # 如果在 Grid View -> 只能用 Grid 图 (带框就带框吧，因为 Grid 本身就是处理过的)

                if self.combo_view.currentIndex() == 1:  # Grid View
                    # Grid 模式下，很难找到对应的纯净图，暂时还是用屏幕显示的图
                    # 但通常 Grid 模式主要看通道差异，红框干扰较少
                    if hasattr(self, 'zoom_img') and self.zoom_img.cv_img_ref is not None:
                        source_img = self.zoom_img.cv_img_ref
                else:
                    # Raw 模式 (绝大多数情况) -> 用纯净图
                    source_img = self.cache_clean_img

            # 降级方案：如果没有纯净图，就用当前显示的
            elif hasattr(self, 'zoom_img') and self.zoom_img.cv_img_ref is not None:
                source_img = self.zoom_img.cv_img_ref
            else:
                return

            # 2. 确定 ROI 大小
            roi_size = 50
            half = roi_size // 2
            h, w = source_img.shape[:2]

            # 3. 计算边界
            x_start = max(0, x - half)
            y_start = max(0, y - half)
            x_end = min(w, x + half)
            y_end = min(h, y + half)

            # 4. 截取 ROI
            roi = source_img[y_start:y_end, x_start:x_end]

            # 5. 转灰度 (如果是彩色源)
            if len(roi.shape) == 3:
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            # 6. 发送给 3D 窗口
            if roi.shape[0] > 0 and roi.shape[1] > 0:
                self.win_3d.update_surface(roi)
    def toggle_params(self):
        idx = self.combo_mode.currentIndex()
        if idx == 0: # Dark
            self.container_dark.show()
            self.container_bright.hide()
            # 显示直方图红线，并同步当前值
            self.hist_widget.thresh_line.show()
            self.hist_widget.set_line_pos(self.sb_thresh_abs.value())
            self.hist_widget.setTitle("Gray Distribution (Drag red line to set Threshold)")
        else: # Bright
            self.container_dark.hide()
            self.container_bright.show()
            # 亮场用的是对比度百分比，绝对阈值线意义不大，隐藏避免误导
            self.hist_widget.thresh_line.hide()
            self.hist_widget.setTitle("Gray Distribution (Reference Only)")
    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI'; }
            QTabWidget::pane { border: 1px solid #333; }
            QTabBar::tab { background: #1e1e1e; color: #888; padding: 10px 20px; }
            QTabBar::tab:selected { background: #00e676; color: #000; font-weight: bold; }
            QGroupBox { border: 1px solid #333; margin-top: 10px; font-weight: bold; color: #00e676; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #2d2d2d; border: 1px solid #444; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #00e676; color: #000; }
            QLineEdit, QSpinBox, QComboBox { background-color: #1a1a1a; border: 1px solid #333; padding: 5px; color: #fff; }
            QTableWidget { gridline-color: #333; background-color: #1a1a1a; selection-background-color: #00e676; selection-color: #000;}
            QHeaderView::section { background-color: #252525; padding: 5px; border: none; font-weight: bold; color: #00e676; }
            QLabel#DetailLabel { font-size: 16px; font-weight: bold; color: #ffd740; border: 1px solid #ffd740; padding: 10px; border-radius: 5px;}
            /* 新增 Pass/Fail 标签样式 */
            QLabel#ResultLabel { font-size: 24px; font-weight: bold; border-radius: 8px; padding: 5px; }
            QLabel#ResultPass { background-color: rgba(0, 230, 118, 0.2); color: #00e676; border: 2px solid #00e676; }
            QLabel#ResultFail { background-color: rgba(255, 23, 68, 0.2); color: #ff1744; border: 2px solid #ff1744; }
        """)
    # =================================================================
    # 请完全替换 CyberApp 类中的 init_single_mode 方法
    # =================================================================
    def init_single_mode(self):
        # 1. 容器与分割器
        root_layout = QHBoxLayout(self.main_widget)  # <--- 关键修改：挂载到 self.main_widget
        root_layout.setContentsMargins(0, 0, 0, 0)

        h_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_layout.addWidget(h_splitter)

        # 2. 左侧面板
        self.left_panel = QFrame();
        self.left_panel.setMinimumWidth(380)
        lp_layout = QVBoxLayout(self.left_panel);
        lp_layout.setContentsMargins(0, 0, 0, 0)
        v_splitter = PyQt6.QtWidgets.QSplitter(Qt.Orientation.Vertical);
        v_splitter.setHandleWidth(4)
        v_splitter.setStyleSheet(
            "QSplitter::handle { background-color: #2a2a2a; } QSplitter::handle:hover { background-color: #00e676; }")
        lp_layout.addWidget(v_splitter)

        # --- 2.1 文件浏览器 ---
        top_widget = QWidget();
        top_layout = QVBoxLayout(top_widget);
        top_layout.setContentsMargins(10, 10, 10, 5)
        grp_files = QGroupBox("FILE BROWSER");
        f_layout = QVBoxLayout(grp_files)
        self.btn_sel_single_dir = QPushButton("📂 Open Folder");
        self.btn_sel_single_dir.clicked.connect(self.open_single_folder);
        self.btn_sel_single_dir.setStyleSheet("background-color: #333; border: 1px solid #555; padding: 6px;")
        f_layout.addWidget(self.btn_sel_single_dir)

        self.file_list = PyQt6.QtWidgets.QListWidget()
        self.file_list.setStyleSheet("""
            QListWidget { background: #1a1a1a; border: 1px solid #333; color: #ccc; outline: none; }
            QListWidget::item { padding: 4px; }
            QListWidget::item:selected { background: #00e676; color: black; }
            QListWidget::item:hover { background: #333; }
        """)
        self.file_list.itemClicked.connect(self.on_file_list_clicked);
        f_layout.addWidget(self.file_list)

        # 👇👇👇 [新增代码开始] 👇👇👇
        # 创建一个水平布局来放两个功能按钮
        h_funcs = QHBoxLayout()

        self.btn_open_batch = QPushButton("⚡ BATCH PROCESS")
        self.btn_open_batch.setStyleSheet("background-color: #6200ea; font-weight: bold;")
        self.btn_open_batch.clicked.connect(self.open_batch_dialog)
        h_funcs.addWidget(self.btn_open_batch)

        self.btn_crop_tool = QPushButton("✂️ BATCH CROP")
        self.btn_crop_tool.setStyleSheet("background-color: #0091ea; font-weight: bold;")
        self.btn_crop_tool.clicked.connect(self.open_crop_dialog)  # 绑定新函数
        h_funcs.addWidget(self.btn_crop_tool)

        f_layout.addLayout(h_funcs)
        # 👆👆👆 [新增代码结束] 👆👆👆


        top_layout.addWidget(grp_files);
        v_splitter.addWidget(top_widget)

        # --- 2.2 参数与控制 ---
        mid_widget = QWidget();
        mid_layout = QVBoxLayout(mid_widget);
        mid_layout.setContentsMargins(10, 5, 10, 5);
        mid_layout.setSpacing(8)
        self.btn_toggle_param = QPushButton("▼ PARAMETERS & HISTOGRAM");
        self.btn_toggle_param.setCheckable(True);
        self.btn_toggle_param.setChecked(True)
        self.btn_toggle_param.setStyleSheet(
            "QPushButton { text-align: left; font-weight: bold; border: none; background: transparent; color: #888; padding: 5px; } QPushButton:checked { color: #00e676; }")
        self.btn_toggle_param.toggled.connect(self.on_param_toggle);
        mid_layout.addWidget(self.btn_toggle_param)

        self.grp_param = QGroupBox();
        self.grp_param.setStyleSheet("QGroupBox { border: 1px solid #333; margin-top: 0px; padding-top: 5px; }")
        p_layout = QVBoxLayout(self.grp_param);
        p_layout.setSpacing(6);
        p_layout.setContentsMargins(5, 5, 5, 5)

        p_layout.addWidget(QLabel("ANALYSIS MODE:"));
        self.combo_mode = QComboBox();
        self.combo_mode.addItems(["🌑 Dark Field (White Pixel)", "☀️ Bright Field (Contrast)"]);
        self.combo_mode.currentIndexChanged.connect(self.toggle_params);
        p_layout.addWidget(self.combo_mode)
        self.hist_widget = InteractiveHistogram();
        self.hist_widget.setFixedHeight(100);
        self.hist_widget.threshold_changed_signal.connect(self.on_hist_line_changed);
        p_layout.addWidget(self.hist_widget)

        h1 = QHBoxLayout();
        h1.addWidget(QLabel("CH:"));
        self.cb_ch = QComboBox();
        self.cb_ch.addItems(["4", "16", "64"]);
        self.cb_ch.setCurrentIndex(1);
        h1.addWidget(self.cb_ch)
        h1.addWidget(QLabel("Filter:"));
        self.cb_fs = QComboBox();
        self.cb_fs.addItems(["3", "5", "7"]);
        self.cb_fs.setCurrentIndex(1);
        h1.addWidget(self.cb_fs);
        p_layout.addLayout(h1)

        self.container_dark = QWidget();
        lay_dark = QVBoxLayout(self.container_dark);
        lay_dark.setContentsMargins(0, 0, 0, 0)

        h_d1 = QHBoxLayout();
        h_d1.addWidget(QLabel("Abs Thresh:"));
        self.sb_thresh_abs = QSpinBox();
        self.sb_thresh_abs.setRange(0, 255);
        self.sb_thresh_abs.setValue(50);
        self.sb_thresh_abs.valueChanged.connect(self.on_spinbox_changed);
        h_d1.addWidget(self.sb_thresh_abs);
        lay_dark.addLayout(h_d1)

        h_d2 = QHBoxLayout();

        # 2.1 同通道距离
        lbl_cd = QLabel("Ch Dist:")
        lbl_cd.setToolTip("同通道同色像素判定Cluster的距离 (建议: 3)")
        h_d2.addWidget(lbl_cd);
        self.sb_ch_dist_dark = QSpinBox();  # 改名以便区分
        self.sb_ch_dist_dark.setRange(1, 20);
        self.sb_ch_dist_dark.setValue(3);  # 默认为3 (严格)
        h_d2.addWidget(self.sb_ch_dist_dark);

        # 2.2 全局距离
        lbl_gd = QLabel("Global:")
        lbl_gd.setToolTip("不同通道合并判定Cluster的距离 (建议: 5)")
        h_d2.addWidget(lbl_gd);
        self.sb_g_dist_dark = QSpinBox();  # 改名以便区分
        self.sb_g_dist_dark.setRange(1, 20);
        self.sb_g_dist_dark.setValue(5);  # 默认为5 (标准)
        h_d2.addWidget(self.sb_g_dist_dark);

        lay_dark.addLayout(h_d2);
        p_layout.addWidget(self.container_dark)

        self.container_bright = QWidget();
        lay_bright = QVBoxLayout(self.container_bright);
        lay_bright.setContentsMargins(0, 0, 0, 0)

        # 行1: 对比度阈值
        h_b1 = QHBoxLayout();
        h_b1.addWidget(QLabel("Contrast %:"));
        self.sb_thresh_pct = QSpinBox();
        self.sb_thresh_pct.setRange(1, 100);
        self.sb_thresh_pct.setValue(30);
        self.sb_thresh_pct.setToolTip("相对背景的对比度百分比 (1-100)")
        h_b1.addWidget(self.sb_thresh_pct);
        lay_bright.addLayout(h_b1)

        # 行2: 聚类距离设置
        h_b2 = QHBoxLayout();

        # 2.1 同通道距离
        h_b2.addWidget(QLabel("Ch Dist:"));
        self.sb_ch_dist_bright = QSpinBox();
        self.sb_ch_dist_bright.setRange(1, 20);
        self.sb_ch_dist_bright.setValue(3);
        h_b2.addWidget(self.sb_ch_dist_bright);

        # 2.2 全局距离
        h_b2.addWidget(QLabel("Global:"));
        self.sb_g_dist_bright = QSpinBox();
        self.sb_g_dist_bright.setRange(1, 20);
        self.sb_g_dist_bright.setValue(5);
        h_b2.addWidget(self.sb_g_dist_bright);

        lay_bright.addLayout(h_b2);
        p_layout.addWidget(self.container_bright);
        self.container_bright.hide()
        mid_layout.addWidget(self.grp_param)

        h_spec = QHBoxLayout();
        h_spec.addWidget(QLabel("Max Pts:"));
        self.sb_spec_pts = QSpinBox();
        self.sb_spec_pts.setRange(0, 99999);
        self.sb_spec_pts.setValue(100);
        h_spec.addWidget(self.sb_spec_pts)
        h_spec.addWidget(QLabel("Max Cls:"));
        self.sb_spec_cls = QSpinBox();
        self.sb_spec_cls.setRange(0, 999);
        self.sb_spec_cls.setValue(0);
        h_spec.addWidget(self.sb_spec_cls);
        mid_layout.addLayout(h_spec)

        self.btn_load = QPushButton("🔄 RE-ANALYZE");
        self.btn_load.clicked.connect(self.re_analyze_current);
        self.btn_load.setMinimumHeight(40);
        mid_layout.addWidget(self.btn_load)

        # === 结果栏 (重点修改区域) ===
        h_res_det = QHBoxLayout()
        h_res_det.setSpacing(5)  # 两个框之间的间距

        # 左边：Pass/Fail (加宽)
        self.lbl_result = QLabel("READY")
        self.lbl_result.setObjectName("ResultLabel")
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_result.setFixedWidth(200)  # [修改] 宽度改为 200
        self.lbl_result.setStyleSheet("""
            background-color: #1a1a1a; 
            color: #666; 
            border: 2px solid #444; 
            border-radius: 6px; 
            font-weight: bold; font-size: 11pt;
        """)
        h_res_det.addWidget(self.lbl_result)

        # 右边：详情 (样式升级为盒子)
        self.lbl_detail = QLabel("Wait Selection")
        self.lbl_detail.setAlignment(Qt.AlignmentFlag.AlignCenter)  # 居中对齐看起来更整齐
        self.lbl_detail.setStyleSheet("""
            background-color: #1a1a1a; 
            color: #ccc; 
            border: 2px solid #444; 
            border-radius: 6px; 
            font-size: 10pt;
            padding: 2px;
        """)
        h_res_det.addWidget(self.lbl_detail, stretch=1)

        mid_layout.addLayout(h_res_det);
        v_splitter.addWidget(mid_widget)

        # ==========================================================
        # 🟢 [修复版] 2.3 结果表格与导出按钮 (防消失/防黑屏)
        # ==========================================================
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(10, 0, 10, 10)
        bottom_layout.setSpacing(5)

        # 1. 顶部工具栏 (导出按钮)
        tool_widget = QWidget()
        # 给工具栏一个固定高度，防止它变得无限大或无限小
        tool_widget.setFixedHeight(40)
        h_tool = QHBoxLayout(tool_widget)
        h_tool.setContentsMargins(0, 0, 0, 0)

        h_tool.addWidget(QLabel("📋 Defect List"))
        h_tool.addStretch()

        self.btn_export_single = QPushButton("💾 Export")
        self.btn_export_single.setFixedSize(100, 30)  # 给按钮固定大小
        self.btn_export_single.setStyleSheet("""
                    QPushButton { background-color: #0091ea; color: white; font-weight: bold; border-radius: 4px; }
                    QPushButton:hover { background-color: #40c4ff; }
                """)

        # ⚠️ 务必确认 self.export_current_data 存在，否则会闪退/黑屏
        # 这里的 try-except 是为了防止因函数不存在而导致的界面全黑
        try:
            self.btn_export_single.clicked.connect(self.export_current_data)
        except AttributeError:
            print("⚠️ 警告: export_current_data 函数未找到，按钮将不起作用")
            self.btn_export_single.setEnabled(False)

        h_tool.addWidget(self.btn_export_single)
        bottom_layout.addWidget(tool_widget)

        # 2. 表格本体 (🚀 升级为 QTableView + Model)
        self.table = QTableView()
        self.table.setAlternatingRowColors(False)
        self.table.setMinimumHeight(200)

        # 设置样式 (保持原有的暗黑风格)
        self.table.setStyleSheet("""
                    QTableView {
                        background-color: #0f0f0f;
                        color: #e0e0e0;
                        gridline-color: #333;
                        border: 1px solid #444;
                        selection-background-color: #00e676;
                        selection-color: #000000;
                    }
                    QHeaderView::section {
                        background-color: #222;
                        color: #aaa;
                        padding: 4px;
                        border: 1px solid #333;
                        font-weight: bold;
                    }
                    QTableCornerButton::section {
                        background-color: #222;
                        border: 1px solid #333;
                    }
                """)
        # --- 初始化 Model 与 Proxy (排序代理) ---
        self.model = DefectTableModel([])  # 初始为空
        self.proxy_model = QSortFilterProxyModel()
        self.proxy_model.setSourceModel(self.model)

        # 绑定 Model 到 View
        self.table.setModel(self.proxy_model)
        self.table.setSortingEnabled(True)  # 启用表头排序

        # 优化列宽显示
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setStretchLastSection(True)

        # 🟢 [修改] 信号连接：clicked 信号传回的是 QModelIndex
        self.table.clicked.connect(self.on_table_click)
        self.table.selectionModel().currentChanged.connect(self.on_table_selection_change)
        bottom_layout.addWidget(self.table)

        # 3. 将底部容器加入分割器
        # 🟢 [关键] 设置 Collapsible 为 False，禁止用户把它拖没了
        v_splitter.addWidget(bottom_widget)
        v_splitter.setCollapsible(2, False)

        # 4. 设置分割器比例 (上:中:下) -> 确保给底部留了位置
        v_splitter.setSizes([150, 400, 300])

        # 5. 最后确认将左侧面板加入主分割器
        h_splitter.addWidget(self.left_panel)


        # 3. 右侧面板 (保持不变)
        right_widget = QWidget();
        right_layout = QVBoxLayout(right_widget);
        right_layout.setContentsMargins(10, 10, 10, 10)
        h_info = QHBoxLayout();
        self.combo_view = QComboBox();
        self.combo_view.addItems(["🖼️ Raw Analysis View", "🔲 Channel Grid View"]);
        self.combo_view.setMinimumWidth(180);
        self.combo_view.currentIndexChanged.connect(self.toggle_view_image);
        h_info.addWidget(self.combo_view);
        # [新增] 3D 视图开关按钮
        self.btn_3d = QPushButton("⛰️ 3D View")
        self.btn_3d.setCheckable(True)  # 这是个开关
        self.btn_3d.setStyleSheet("""
                    QPushButton { background: #333; color: #ccc; border: 1px solid #555; padding: 4px 10px; border-radius: 4px; }
                    QPushButton:checked { background: #6200ea; color: white; border: 1px solid #7c4dff; }
                    QPushButton:hover { background: #444; }
                """)
        self.btn_3d.clicked.connect(self.toggle_3d_window)
        h_info.addWidget(self.btn_3d)

        h_info.addStretch();
        h_info.addWidget(QLabel("Shortcuts: [WASD] Pan | [Space] View | [Ctrl+→] Next"));
        h_info.addStretch();
        self.lbl_cursor_info = QLabel("X: --  Y: --  Val: --");
        self.lbl_cursor_info.setStyleSheet(
            "color: #00e676; font-weight: bold; font-family: Consolas; background: #222; padding: 2px 8px; border-radius: 4px;");
        h_info.addWidget(self.lbl_cursor_info);
        right_layout.addLayout(h_info)
        self.zoom_img = ZoomableGraphicsView();
        self.zoom_img.mouse_moved_signal.connect(self.update_cursor_display);
        right_layout.addWidget(self.zoom_img, stretch=2)
        self.graph = pg.PlotWidget(background='#0f0f0f');
        self.graph.showGrid(x=True, y=True, alpha=0.3);
        plot_item = self.graph.getPlotItem();
        plot_item.invertY(True);
        plot_item.showAxis('bottom', False);
        plot_item.showAxis('top', True);
        plot_item.showAxis('left', True);
        right_layout.addWidget(self.graph, stretch=2)
        h_splitter.addWidget(right_widget);
        h_splitter.setSizes([450, 900])
        self.current_single_dir = None;
        self.current_file_path = None

        # ==========================================================

    def on_table_selection_change(self, current, previous):
        if not current.isValid(): return
        # 直接复用点击逻辑
        self.on_table_click(current)

    def get_params(self):
        # 1. 安全检查：如果控件已经被销毁，返回默认值，避免闪退
        try:
            # 尝试访问 C++ 对象
            if not self.combo_mode or not self.combo_mode.isVisible():
                # 这里只是简单检查，核心是下面的 currentIndex 可能会抛错
                pass
            current_idx = self.combo_mode.currentIndex()
        except RuntimeError:
            print("⚠️ 警告：UI控件丢失，使用默认参数")
            return {"mode": "Dark", "ch": 16, "fs": 5, "thresh": 50, "ch_dist": 3, "g_dist": 5}

        # 2. 正常获取逻辑
        mode = "Dark" if current_idx == 0 else "Bright"

        common = {
            "ch": int(self.cb_ch.currentText()),
            "fs": int(self.cb_fs.currentText()),
            "mode": mode
        }

        if mode == "Dark":
            return {**common,
                    "thresh": self.sb_thresh_abs.value(),
                    # [修改] 获取暗场的两个距离参数
                    "ch_dist": self.sb_ch_dist_dark.value(),
                    "g_dist": self.sb_g_dist_dark.value()}
        else:
            return {**common,
                    "thresh": self.sb_thresh_pct.value(),
                    # [修改] 获取亮场的两个距离参数
                    "ch_dist": self.sb_ch_dist_bright.value(),
                    "g_dist": self.sb_g_dist_bright.value()}
    def sel_dir(self, l):
        # 1. 打开文件夹选择框
        d = QFileDialog.getExistingDirectory(self, "Select Directory")
        # 2. 如果用户选了路径（没点取消），就更新标签文本
        if d:
            l.setText(d)

    # def run_single_analysis(self):
    #     p, _ = QFileDialog.getOpenFileName(self, "Img", "", "Img (*.png *.jpg *.tif)")
    #     if not p: return
    #     self.worker = SingleWorker(p, self.get_params())
    #     self.worker.result_signal.connect(self.on_single_finished)
    #     self.worker.start()

    def on_single_finished(self, vis_raw, vis_grid, data, img_raw):
        # 1. 缓存数据和图片
        self.current_data_cache = data
        self.cache_vis_raw = vis_raw  # 缓存带框原图
        self.cache_vis_grid = vis_grid  # 缓存网格图

        # 🟢 [关键!] 缓存原始数据 (这就是你要的“完全不缩略”的数据)
        self.cache_raw_img = img_raw
        # 为了 3D 图也没红框干扰，3D 也可以用这个
        self.cache_clean_img = img_raw
        # [新增] 尝试重新读取一份纯净的原图用于 3D 显示
        # 因为 vis_raw 已经被 OpenCV 画上了红框，3D 地形图会把红框也当成像素高度显示出来

        # 2. 安全检查
        if vis_raw is None: return
        h, w = vis_raw.shape[:2]  # 使用原图尺寸作为基准

        # 3. [修改] 更新图片显示
        # 根据当前 Combox 的选择来决定显示哪张图
        self.toggle_view_image(self.combo_view.currentIndex())

        # 4. 获取参数
        params = self.get_params()
        global_dist = params['g_dist']
        max_pts = self.sb_spec_pts.value()
        max_cls = self.sb_spec_cls.value()

        # 5. 统计与判定 (逻辑不变)
        stats = CoreAlgorithm.get_stats(data, (h, w), global_dist)
        total_pts = stats['total_pts']
        total_cls = stats['white_cls'] + stats['black_cls']
        is_pass = (total_pts <= max_pts) and (total_cls <= max_cls)

        # 6. 更新结果标签 (逻辑不变)
        info_str = (f"(Pixels: {total_pts}, Groups: {total_cls})\n"
                    f"⚪W_Pix: {stats['white_pts']} (Grp:{stats['white_cls']})\n"
                    f"⚫B_Pix: {stats['black_pts']} (Grp:{stats['black_cls']})")

        if is_pass:
            self.lbl_result.setText(f"🟢 PASS\n{info_str}")
            self.lbl_result.setStyleSheet(
                """background-color: rgba(0, 230, 118, 0.15); color: #00e676; border: 2px solid #00e676; border-radius: 6px; font-weight: bold; font-size: 11pt;""")
        else:
            self.lbl_result.setText(f"🔴 FAIL\n{info_str}")
            self.lbl_result.setStyleSheet(
                """background-color: rgba(255, 23, 68, 0.15); color: #ff1744; border: 2px solid #ff1744; border-radius: 6px; font-weight: bold; font-size: 11pt;""")

        # ==========================================================
        # 📊 表格填充逻辑 (🚀 极速版)
        # ==========================================================
        self.current_data_cache = data

        # 1. 更新表格 (瞬间完成，无需循环)
        self.model.update_data(data)

        # 2. 收集 Graph 绘图用的数据 (这一步还是需要的，但它是纯数据处理，很快)
        self.cursor_lines = []
        spots_cluster, spots_bright, spots_dark = [], [], []

        for r, d in enumerate(data):
            # 收集 Graph 绘图用的数据
            pt_data = {'pos': (d['gx'], d['gy']), 'data': r}
            pol = d.get('polarity', 'Bright')

            if "Cluster" in d['final_type']:
                spots_cluster.append(pt_data)
            elif pol == 'Dark':
                spots_dark.append(pt_data)
            else:
                spots_bright.append(pt_data)

        # 3. 绘制散点图 (逻辑不变)
        self.graph.clear()  # 记得先清空旧图

        if spots_bright:
            s1 = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush(0, 255, 0, 200), symbol='o')
            s1.addPoints(spots_bright)
            self.graph.addItem(s1)

        if spots_dark:
            s2 = pg.ScatterPlotItem(size=8, pen=None, brush=pg.mkBrush(30, 144, 255, 200), symbol='o')
            s2.addPoints(spots_dark)
            self.graph.addItem(s2)

        if spots_cluster:
            s3 = pg.ScatterPlotItem(size=14, pen=pg.mkPen('w', width=1), brush=pg.mkBrush(255, 50, 50, 180),
                                    symbol='s')
            s3.addPoints(spots_cluster)
            self.graph.addItem(s3)

        # 4. 设置图表范围
        if vis_raw is not None:
            h, w = vis_raw.shape[:2]
            self.graph.setXRange(0, float(w))
            self.graph.setYRange(0, float(h))

        # 5. 刷新直方图
        if hasattr(self, 'hist_widget'):
            self.hist_widget.update_data(self.cache_raw_img)

    def toggle_view_image(self, index):
        # 检查缓存是否存在
        if not hasattr(self, 'cache_vis_raw') or self.cache_vis_raw is None:
            return

        if index == 0:
            # 显示原始分析图
            self.zoom_img.set_image(self.cache_vis_raw)
            # 启用表格点击联动
            self.table.setEnabled(True)
        else:
            if (not hasattr(self, 'cache_vis_grid') or self.cache_vis_grid is None) and hasattr(self, 'cache_raw_img'):
                # 给个简单的加载提示（可选）
                self.btn_load.setText("GENERATING GRID...")
                QApplication.processEvents()  # 强制刷新界面显示文字

                # 调用核心算法生成
                channels = self.get_params()['ch']
                self.cache_vis_grid = CoreAlgorithm.generate_channel_grid(self.cache_raw_img, channels)

                self.btn_load.setText("🔄 RE-ANALYZE")  # 恢复文字

                # 显示通道网格图
            if self.cache_vis_grid is not None:
                self.zoom_img.set_image(self.cache_vis_grid)

            # Grid 模式下禁用表格联动（防止坐标错位）
            # self.table.setEnabled(False)

    def on_table_click(self, index):
        """
        [修改版] 适配 QTableView + QSortFilterProxyModel
        """
        if not index.isValid(): return

        # 1. 索引映射：View(排序后) -> Source(原始数据顺序)
        source_index = self.proxy_model.mapToSource(index)
        real_row = source_index.row()

        # 2. 获取数据
        if not hasattr(self, 'current_data_cache') or real_row >= len(self.current_data_cache):
            return

        data = self.current_data_cache[real_row]

        # 3. 更新详情标签 (逻辑不变)
        info_text = (f"TYPE: <b>{data['final_type']}</b> &nbsp;|&nbsp; "
                     f"LOC: <span style='color:#00e676'>({data['gx']}, {data['gy']})</span> &nbsp;|&nbsp; "
                     f"VAL: <span style='color:#ff9800'>{data['val']}</span>")
        self.lbl_detail.setText(info_text)

        # 4. 散点图十字光标 (逻辑不变)
        if not hasattr(self, 'cursor_lines'): self.cursor_lines = []
        plot_item = self.graph.getPlotItem()
        for line in self.cursor_lines:
            try:
                plot_item.removeItem(line)
            except:
                pass
        self.cursor_lines.clear()

        pen_style = pg.mkPen(color='#ffd740', width=2, style=Qt.PenStyle.DashLine)
        v_line = pg.InfiniteLine(pos=data['gx'], angle=90, pen=pen_style)
        h_line = pg.InfiniteLine(pos=data['gy'], angle=0, pen=pen_style)
        plot_item.addItem(v_line)
        plot_item.addItem(h_line)
        self.cursor_lines.extend([v_line, h_line])

        # 5. 图片高亮 (逻辑不变)
        is_grid_view = (self.combo_view.currentIndex() == 1)
        target_x, target_y = data['gx'], data['gy']

        if is_grid_view and hasattr(self, 'cache_vis_raw'):
            # 坐标映射算法 (Raw -> Grid)
            h_raw, w_raw = self.cache_vis_raw.shape[:2]
            channels = self.get_params()['ch']
            step = int(np.sqrt(channels))
            sub_h = h_raw // step
            sub_w = w_raw // step

            grid_row_idx = data['gy'] % step
            grid_col_idx = data['gx'] % step
            local_y = data['gy'] // step
            local_x = data['gx'] // step

            if local_y < sub_h and local_x < sub_w:
                target_x = grid_col_idx * sub_w + local_x
                target_y = grid_row_idx * sub_h + local_y

        self.zoom_img.highlight_defect(target_x, target_y, size=30)

    def run_batch(self):
        # 简单的校验
        if "None" in [self.lbl_in.text(), self.lbl_out.text()]:
            self.log.append("❌ Please select Input and Output folders first.")
            return

        self.log.clear()
        self.btn_run_batch.setEnabled(False)

        # 1. 获取规格参数 (Max Points, Max Clusters)
        # 确保您的界面里有 self.sb_spec_pts 和 self.sb_spec_cls 这两个控件
        specs = (self.sb_spec_pts.value(), self.sb_spec_cls.value())

        # 2. 传递 specs 给 Worker
        self.bw = BatchWorker(self.lbl_in.text(), self.lbl_out.text(), self.get_params(), specs)

        self.bw.progress_signal.connect(lambda v, m: (self.pbar.setValue(v), self.log.append(m)))
        self.bw.finished_signal.connect(lambda: self.btn_run_batch.setEnabled(True))
        self.bw.start()

    # ==========================================================
    # ⌨️ 键盘快捷键响应 (V13 新增)
    # ==========================================================
    def keyPressEvent(self, event):
        """
        重写键盘按下事件，实现快捷键逻辑
        """
        # 1. 图片平移 (W/A/S/D) - 步长 50 像素
        step = 50
        if event.key() == Qt.Key.Key_W:
            self.zoom_img.pan_view(0, -step)
        elif event.key() == Qt.Key.Key_S:
            self.zoom_img.pan_view(0, step)
        elif event.key() == Qt.Key.Key_A:
            self.zoom_img.pan_view(-step, 0)
        elif event.key() == Qt.Key.Key_D:
            self.zoom_img.pan_view(step, 0)

        # 2. 视图切换 (空格键 Space)
        elif event.key() == Qt.Key.Key_Space:
            if hasattr(self, 'combo_view'):
                # 在 0 和 1 之间循环切换
                current_idx = self.combo_view.currentIndex()
                new_idx = 1 - current_idx
                self.combo_view.setCurrentIndex(new_idx)

        # 3. 文件翻页 (Ctrl + 左/右方向键)
        # 必须按住 Ctrl，防止误触
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_Right:
                self.switch_image(1)  # 下一张
            elif event.key() == Qt.Key.Key_Left:
                self.switch_image(-1)  # 上一张

        # 务必调用父类方法，否则其他标准快捷键可能失效
        super().keyPressEvent(event)

    def switch_image(self, direction):
        """
        切换文件列表中的图片
        direction: 1 为下一张, -1 为上一张
        """
        # 检查列表是否可用
        if not hasattr(self, 'file_list') or self.file_list.count() == 0:
            return

        # 获取当前行号
        current_row = self.file_list.currentRow()

        # 计算新行号
        new_row = current_row + direction

        # 边界检查
        if 0 <= new_row < self.file_list.count():
            # 1. 选中新行
            self.file_list.setCurrentRow(new_row)

            # 2. 获取该行 Item 并触发点击逻辑
            item = self.file_list.item(new_row)
            self.on_file_list_clicked(item)

            # 3. 滚动列表确保选中项可见
            self.file_list.scrollToItem(item)
        else:
            # (可选) 到底了提示一下，或者循环到开头
            print("End of file list reached.")

    # ==========================================
    # 📊 直方图与参数联动槽函数
    # ==========================================
    def on_hist_line_changed(self, val):
        """直方图线被拖拽 -> 更新 SpinBox"""
        # 只有在暗场模式下，直方图阈值才有直接物理意义
        if self.combo_mode.currentIndex() == 0:  # Dark Mode
            # blockSignals 防止死循环 (SpinBox变->又触发线变)
            self.sb_thresh_abs.blockSignals(True)
            self.sb_thresh_abs.setValue(val)
            self.sb_thresh_abs.blockSignals(False)
            # 🟢 [新增] 拖动直方图线松手后，也自动刷新
            if hasattr(self, 'debounce_timer'):
                self.debounce_timer.start()
            # 可选：如果想拖动时实时重新分析，可以在这里调用 self.trigger_analysis
            # 但考虑到性能，建议还是手动点刷新，或者加个防抖

    def on_spinbox_changed(self, val):
        """SpinBox 数值改变 -> 更新直方图线位置"""
        self.hist_widget.set_line_pos(val)
        # 🟢 [新增] 启动防抖计时器
        if hasattr(self, 'debounce_timer'):
            self.debounce_timer.start()

    # [新增] 参数面板折叠逻辑
    def on_param_toggle(self, checked):
        # 1. 控制参数 GroupBox 的显示/隐藏
        self.grp_param.setVisible(checked)

        # 2. 更新按钮文字 (指示箭头方向)
        if checked:
            self.btn_toggle_param.setText("▼ PARAMETERS & HISTOGRAM")
        else:
            self.btn_toggle_param.setText("▶ PARAMETERS (Hidden)")

    def toggle_3d_window(self, checked):
        if checked:
            self.win_3d.show()
        else:
            self.win_3d.hide()
# [新增] 打开批量处理弹窗
    def open_batch_dialog(self):
        # 传递当前界面的参数给弹窗，这样不用重新设置一遍
        BatchProcessDialog(self.get_params(), self).exec()

    # [新增] 打开批量截图工具
    # [优化] 打开批量截图工具，自动填入当前鼠标位置
    # [优化] 打开批量截图工具，自动填入当前鼠标位置 (Center)
    # [优化] 打开批量截图工具，自动填入当前【视图中心】坐标
    # [优化] 打开批量截图工具 (智能联动版)
    # 替换原来的 open_crop_dialog
    def open_crop_dialog(self):
        default_size = 10

        # 1. 默认中心点 (基于当前视图)
        if hasattr(self, 'zoom_img'):
            view_center = self.zoom_img.viewport().rect().center()
            scene_pos = self.zoom_img.mapToScene(view_center)
            cx, cy = int(scene_pos.x()), int(scene_pos.y())
        else:
            cx, cy = 0, 0
        default_rect = (cx, cy, default_size, default_size)

        # 🟢 [修改] 提取 Cluster 数据 (带类型)
        cluster_data = []
        if hasattr(self, 'current_data_cache') and self.current_data_cache:
            for d in self.current_data_cache:
                ftype = d.get('final_type', '')
                # 只要是 Cluster，就把它的信息存下来
                if "Cluster" in ftype:
                    cluster_data.append({
                        'x': d['gx'],
                        'y': d['gy'],
                        'type': ftype  # 关键：保留类型信息
                    })

        # 3. 实例化弹窗 (传入 initial_data)
        dlg = BatchCropDialog(self, default_rect, initial_data=cluster_data)

        # 4. 路径填充
        if self.current_single_dir:
            dlg.edt_in.setText(str(self.current_single_dir))
            out_path = self.current_single_dir / "crop_output"
            dlg.edt_out.setText(str(out_path))

        dlg.exec()
    def export_current_data(self):
        # 1. 基础检查
        if not hasattr(self, 'current_data_cache') or not self.current_data_cache:
            QMessageBox.warning(self, "Warning", "No analysis data available to export!")
            return

        # 2. 获取原图
        source_img = None
        if hasattr(self, 'cache_clean_img') and self.cache_clean_img is not None:
            source_img = self.cache_clean_img
        elif hasattr(self, 'zoom_img') and self.zoom_img.cv_img_ref is not None:
            source_img = self.zoom_img.cv_img_ref

        if source_img is None:
            QMessageBox.warning(self, "Error", "Source image lost. Please re-analyze.")
            return
        # =========================================================
        # 🟢 [新增] 弹出设置对话框
        # =========================================================
        dlg = SingleExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return  # 用户点了取消

        # 获取用户设置的参数
        snap_radius, snap_size, save_details = dlg.get_settings()
        # =========================================================
        # 3. 选择路径
        default_name = Path(self.current_file_path).stem if self.current_file_path else "Analysis_Report"
        save_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not save_dir: return

        try:
            self.btn_export_single.setText("Saving...")
            self.btn_export_single.setEnabled(False)
            QApplication.processEvents()

            # =========================================================
            # 🟢 [新增] 准备统计数据 (Stats) 和 规格 (Specs)
            # =========================================================
            # A. 获取当前参数 (为了拿到 g_dist)
            params = self.get_params()
            h, w = source_img.shape[:2]

            # B. 计算详细统计 (白点数、黑点数、团簇数等)
            # 这里的 params['g_dist'] 必须确保和分析时使用的一致
            stats = CoreAlgorithm.get_stats(self.current_data_cache, (h, w), params['g_dist'])

            # C. 获取界面上的规格设置 (Max Pts, Max Cls)
            max_pts = self.sb_spec_pts.value()
            max_cls = self.sb_spec_cls.value()
            specs = (max_pts, max_cls)
            # =========================================================

            # 4. 执行导出
            # 你可以在这里写死，或者也添加一个 QDialog 来询问用户
            # 这里暂时使用默认值 (Radius=5, Size=60)
            excel_file = ExportHandler.save_report(
                data=self.current_data_cache,
                original_img=source_img,
                filename_stem=default_name,
                output_dir=save_dir,
                stats=stats,
                specs=specs,
                save_details=True,
                snap_params=(5, 60)  # 🟢 显式传入默认值，或者你想要的值
            )

            QMessageBox.information(self, "Success", f"Report saved to:\n{excel_file}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")
        finally:
            self.btn_export_single.setText("💾 Export Report")
            self.btn_export_single.setEnabled(True)

    # 🟢 [新增] 拖拽进入事件
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    # 🟢 [新增] 拖拽放下事件
    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if not urls: return

        path = Path(urls[0].toLocalFile())

        if path.is_dir():
            # 如果拖入的是文件夹 -> 打开文件夹
            self.current_single_dir = path
            self.file_list.clear()
            # ... (复用 open_single_folder 的加载逻辑，建议提取为 load_files_from_dir 函数) ...
            # 为了简单，这里直接触发按钮逻辑需要稍微重构，或者直接手动加载一遍：
            self.edt_in.setText(str(path)) if hasattr(self, 'edt_in') else None  # 如果有路径框
            # 简单方式：模拟加载逻辑
            self.settings.setValue("paths/last_dir", str(path))
            self.load_settings()  # 重新加载会刷新列表

        elif path.is_file() and path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']:
            # 如果拖入的是图片 -> 分析该图
            # 如果该图不在当前列表里，最好也顺便加载同级目录
            parent_dir = path.parent
            if self.current_single_dir != parent_dir:
                self.current_single_dir = parent_dir
                self.settings.setValue("paths/last_dir", str(parent_dir))
                self.load_settings()  # 刷新列表

            # 触发分析
            self.trigger_analysis(str(path))
    pass
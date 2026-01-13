from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QGroupBox, QGridLayout,
    QSpinBox, QCheckBox, QPushButton, QLineEdit, QFileDialog,
    QProgressBar, QTextEdit, QTabWidget, QWidget, QFileDialog
)
from PyQt6.QtCore import Qt

from core.workers import BatchWorker, BatchCropWorker # 引用 Worker

# 1. 放入 SingleExportDialog 类
class SingleExportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EXPORT SETTINGS")
        self.resize(400, 300)
        self.setStyleSheet("background-color: #1a1a1a; color: #ccc;")

        layout = QVBoxLayout(self)

        # 1. 标题
        lbl_title = QLabel("📝 Report Configuration")
        lbl_title.setStyleSheet("color: #00e676; font-weight: bold; font-size: 12pt; margin-bottom: 10px;")
        lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl_title)

        # 2. 截图设置组
        grp_snap = QGroupBox("📸 Snapshot Settings")
        grp_snap.setStyleSheet(
            "QGroupBox { border: 1px solid #555; margin-top: 10px; font-weight: bold; color: #00e676; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        snap_layout = QGridLayout(grp_snap)

        # Radius
        snap_layout.addWidget(QLabel("Capture Radius:"), 0, 0)
        self.sb_radius = QSpinBox()
        self.sb_radius.setRange(1, 100)
        self.sb_radius.setValue(5)  # 默认值
        self.sb_radius.setSuffix(" px")
        self.sb_radius.setToolTip("Half-width of the area cropped from original image.\n(5px radius = 11x11px area)")
        self.sb_radius.setStyleSheet("border: 1px solid #444; padding: 4px; color: #fff;")
        snap_layout.addWidget(self.sb_radius, 0, 1)

        # Output Size
        snap_layout.addWidget(QLabel("Output Size:"), 1, 0)
        self.sb_size = QSpinBox()
        self.sb_size.setRange(10, 1000)
        self.sb_size.setValue(60)  # 默认值
        self.sb_size.setSuffix(" px")
        self.sb_size.setToolTip("Final resolution of the image in Excel.")
        self.sb_size.setStyleSheet("border: 1px solid #444; padding: 4px; color: #fff;")
        snap_layout.addWidget(self.sb_size, 1, 1)

        layout.addWidget(grp_snap)

        # 3. 选项组
        grp_opt = QGroupBox("📄 Content Options")
        grp_opt.setStyleSheet(
            "QGroupBox { border: 1px solid #555; margin-top: 10px; font-weight: bold; color: #00e676; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        opt_layout = QVBoxLayout(grp_opt)

        self.chk_details = QCheckBox("Generate Detail Sheet + Images")
        self.chk_details.setChecked(True)
        self.chk_details.setStyleSheet("color: #ddd;")
        opt_layout.addWidget(self.chk_details)

        layout.addWidget(grp_opt)

        layout.addStretch()

        # 4. 按钮
        h_btn = QHBoxLayout()
        btn_cancel = QPushButton("Cancel")
        btn_cancel.setStyleSheet("background: #333; color: #aaa; border: 1px solid #555; padding: 6px;")
        btn_cancel.clicked.connect(self.reject)

        btn_ok = QPushButton("✅ Export")
        btn_ok.setStyleSheet("background: #00e676; color: #000; font-weight: bold; border: none; padding: 6px;")
        btn_ok.clicked.connect(self.accept)

        h_btn.addWidget(btn_cancel)
        h_btn.addWidget(btn_ok)
        layout.addLayout(h_btn)

    def get_settings(self):
        """返回 (radius, size, is_details)"""
        return (self.sb_radius.value(), self.sb_size.value(), self.chk_details.isChecked())
    pass

# 2. 放入 BatchProcessDialog 类
class BatchProcessDialog(QDialog):
    def __init__(self, current_params, parent=None):
        super().__init__(parent)
        self.setWindowTitle("BATCH PROCESSOR")
        self.resize(600, 450)
        # 简单的暗色样式
        self.setStyleSheet("background-color: #1a1a1a; color: #ccc;")

        self.params = current_params  # 接收主界面的参数
        self.worker = None

        layout = QVBoxLayout(self)

        # 1. 输入路径
        layout.addWidget(QLabel("📂 Input Folder:"))
        h1 = QHBoxLayout()
        self.edt_input = QLineEdit()
        self.edt_input.setStyleSheet("border: 1px solid #444; padding: 5px; color: #fff;")
        self.btn_input = QPushButton("Browse")
        self.btn_input.clicked.connect(self.browse_input)
        h1.addWidget(self.edt_input); h1.addWidget(self.btn_input)
        layout.addLayout(h1)

        # 2. 输出路径
        layout.addWidget(QLabel("💾 Output Folder:"))
        h2 = QHBoxLayout()
        self.edt_output = QLineEdit()
        self.edt_output.setStyleSheet("border: 1px solid #444; padding: 5px; color: #fff;")
        self.btn_output = QPushButton("Browse")
        self.btn_output.clicked.connect(self.browse_output)
        h2.addWidget(self.edt_output); h2.addWidget(self.btn_output)
        layout.addLayout(h2)

        # 3. 文件名过滤
        layout.addWidget(QLabel("🔍 Filename Filter (Optional):"))
        self.edt_filter = QLineEdit()
        self.edt_filter.setPlaceholderText("e.g. 'sensor_01' (Leave empty for all)")
        self.edt_filter.setStyleSheet("border: 1px solid #444; padding: 5px; color: #00e676;")
        layout.addWidget(self.edt_filter)

        # ============================================================
        # 🟢 [新增] 截图参数设置区域
        # ============================================================
        self.grp_snap = QGroupBox("📸 Cluster Snapshot Settings")
        self.grp_snap.setStyleSheet(
            "QGroupBox { border: 1px solid #555; margin-top: 10px; font-weight: bold; color: #00e676; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        snap_layout = QHBoxLayout(self.grp_snap)

        # 参数 A: 抓取半径 (决定截取原图多大的区域)
        # 半径=5 意味着 直径=10 (即 10x10 的原图区域)
        snap_layout.addWidget(QLabel("Capture Radius (px):"))
        self.sb_radius = QSpinBox()
        self.sb_radius.setRange(1, 100)
        self.sb_radius.setValue(5)  # 默认值 (原 half=5)
        self.sb_radius.setToolTip(
            "Radius around the defect center to crop from original image.\n(e.g. 5 means 11x11 area)")
        snap_layout.addWidget(self.sb_radius)

        # 参数 B: 输出尺寸 (决定最终保存的图片多大，即放大倍数)
        # 如果半径5(11px) -> 输出60px，大约放大 5.5 倍
        snap_layout.addWidget(QLabel("Final Image Size (px):"))
        self.sb_out_size = QSpinBox()
        self.sb_out_size.setRange(10, 500)
        self.sb_out_size.setValue(60)  # 默认值 (原 60x60)
        self.sb_out_size.setToolTip("The resolution of the saved snapshot image.")
        self.sb_out_size.setSuffix(" px")
        snap_layout.addWidget(self.sb_out_size)

        layout.addWidget(self.grp_snap)
        # ============================================================
        # 4. 进度条
        self.pbar = QProgressBar()
        self.pbar.setStyleSheet("QProgressBar { border: 1px solid #444; text-align: center; } QProgressBar::chunk { background-color: #6200ea; }")
        layout.addWidget(self.pbar)

        # 5. 日志区域
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setStyleSheet("background: #000; border: 1px solid #333; font-family: Consolas;")
        layout.addWidget(self.txt_log)

        # 6. 开始按钮
        self.btn_start = QPushButton("▶ START BATCH")
        self.btn_start.setMinimumHeight(40)
        self.btn_start.setStyleSheet("background: #00e676; color: #000; font-weight: bold;")
        self.btn_start.clicked.connect(self.start_batch)
        layout.addWidget(self.btn_start)
        # ... 在布局代码里添加 ...
        self.chk_export_details = QCheckBox("生成详细报告 (Sheet2 + 截图)")
        self.chk_export_details.setChecked(True)  # 默认勾选
        layout.addWidget(self.chk_export_details)  # 把它加到你的布局里

    def browse_input(self):
        d = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if d: self.edt_input.setText(d)

    def browse_output(self):
        d = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if d: self.edt_output.setText(d)

    def start_batch(self):
        inp = self.edt_input.text()
        out = self.edt_output.text()
        if not inp or not out:
            self.txt_log.append("❌ Please select both Input and Output folders.")
            return

        self.btn_start.setEnabled(False)
        self.txt_log.clear()

        # 获取规格参数 (需要在 params 里有这些 key，如果没有就用默认值)
        # 注意：这里假设您的 params 字典里可能没有 specs，所以我们安全获取
        max_pts = self.params.get('max_defect_count', 100)
        max_cls = self.params.get('max_cluster_size', 0)
        specs = (max_pts, max_cls)

        # 🟢 [修改] 获取截图参数
        snap_radius = self.sb_radius.value()
        snap_size = self.sb_out_size.value()
        snap_params = (snap_radius, snap_size)

        # 启动后台线程 (BatchWorker 必须已定义)
        self.worker = BatchWorker(inp, out, self.edt_filter.text(), self.params, specs)
        self.worker.log_signal.connect(self.txt_log.append)
        self.worker.progress_signal.connect(self.update_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.export_details = self.chk_export_details.isChecked()
        self.worker.start()

    def update_progress(self, curr, total):
        self.pbar.setMaximum(total)
        self.pbar.setValue(curr)

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.txt_log.append("--- DONE ---")

    pass

# 3. 放入 BatchCropDialog 类
# ==========================================
# ✂️ 批量截图工具 (UI升级版：中心点逻辑 + 单图缩放)
# ==========================================
# 请替换整个 BatchCropDialog 类
class BatchCropDialog(QDialog):
    # 🟢 [修改] 接收 initial_data (包含坐标和类型) 而不是简单的坐标列表
    def __init__(self, parent=None, default_rect=None, initial_data=None):
        super().__init__(parent)
        self.setWindowTitle("BATCH CROP TOOLKIT (Linked & Filtered)")
        self.resize(550, 700)  # 高度增加以容纳过滤器
        self.setStyleSheet("background-color: #1a1a1a; color: #ccc;")

        self.worker = None
        self.coord_list = []  # 最终用于截图的坐标 [(x, y)...]
        self.linked_data = initial_data if initial_data else []  # 备份原始数据 [{'x':1, 'y':2, 'type':'...'}, ...]

        layout = QVBoxLayout(self)

        # === 1. 公共路径设置 (保持不变) ===
        grp_io = QGroupBox("Path Settings")
        grp_io.setStyleSheet(
            "QGroupBox { border: 1px solid #444; margin-top: 10px; font-weight: bold; color: #00e676; } QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }")
        l_io = QVBoxLayout(grp_io)

        h1 = QHBoxLayout()
        self.edt_in = QLineEdit();
        self.edt_in.setPlaceholderText("Input Folder...")
        self.edt_in.setStyleSheet("border: 1px solid #444; padding: 5px; color: #fff;")
        btn_in = QPushButton("📂");
        btn_in.clicked.connect(self.browse_in)
        h1.addWidget(self.edt_in);
        h1.addWidget(btn_in);
        l_io.addLayout(h1)

        h2 = QHBoxLayout()
        self.edt_out = QLineEdit();
        self.edt_out.setPlaceholderText("Output Folder...")
        self.edt_out.setStyleSheet("border: 1px solid #444; padding: 5px; color: #fff;")
        btn_out = QPushButton("📂");
        btn_out.clicked.connect(self.browse_out)
        h2.addWidget(self.edt_out);
        h2.addWidget(btn_out);
        l_io.addLayout(h2)

        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Filter:"))
        self.edt_filter = QLineEdit();
        self.edt_filter.setPlaceholderText("e.g. 'sensor_A' (Optional)")
        self.edt_filter.setStyleSheet("border: 1px solid #444; padding: 5px; color: #00e676;")
        h3.addWidget(self.edt_filter);
        l_io.addLayout(h3)
        layout.addWidget(grp_io)

        # === 2. Tabs ===
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #444; }
            QTabBar::tab { background: #333; color: #aaa; padding: 8px 20px; }
            QTabBar::tab:selected { background: #00e676; color: #000; font-weight: bold; }
        """)

        self.tab_single = QWidget()
        self._init_tab_single(default_rect)
        self.tabs.addTab(self.tab_single, "🔲 Single Region")

        self.tab_coords = QWidget()
        self._init_tab_coords()
        self.tabs.addTab(self.tab_coords, "📍 Batch Clusters")

        layout.addWidget(self.tabs)

        # === 3. 进度与日志 ===
        self.pbar = QProgressBar()
        self.pbar.setStyleSheet(
            "QProgressBar { border: 1px solid #444; text-align: center; } QProgressBar::chunk { background-color: #00e676; }")
        layout.addWidget(self.pbar)
        self.txt_log = QTextEdit();
        self.txt_log.setReadOnly(True);
        self.txt_log.setStyleSheet("background: #000; font-family: Consolas; font-size: 9pt; border: 1px solid #333;")
        layout.addWidget(self.txt_log)
        self.btn_run = QPushButton("▶ START PROCESS");
        self.btn_run.setMinimumHeight(45);
        self.btn_run.setStyleSheet("background-color: #6200ea; color: white; font-weight: bold; font-size: 11pt;")
        self.btn_run.clicked.connect(self.start_process)
        layout.addWidget(self.btn_run)

        # 🟢 [新增] 初始化逻辑：如果有联动数据，自动应用筛选
        if self.linked_data:
            self.tabs.setCurrentIndex(1)
            self.sb_size.setValue(60)  # 默认 Cluster 大小
            self.apply_type_filter()  # 初始筛选
            self.txt_log.append(f"🔗 Linked Analysis Data: Found {len(self.linked_data)} total clusters.")

    # 🟢 [新增] 筛选逻辑函数
    def apply_type_filter(self):
        # 如果是 CSV 模式，不执行此筛选
        if not self.linked_data: return

        filtered = []
        allow_channel = self.chk_ch_cls.isChecked()
        allow_spatial = self.chk_sp_cls.isChecked()

        for item in self.linked_data:
            ctype = item['type']
            # 根据类型关键字筛选
            if allow_channel and "Channel" in ctype:
                filtered.append((item['x'], item['y']))
            elif allow_spatial and "Spatial" in ctype:
                filtered.append((item['x'], item['y']))

        self.coord_list = filtered

        # 更新 UI 状态
        count = len(filtered)
        self.lbl_csv_status.setText(f"🎯 Selected {count} pts")

        if count > 0:
            self.lbl_csv_status.setStyleSheet("color: #00e676; font-weight: bold;")
        else:
            self.lbl_csv_status.setStyleSheet("color: #ff5252;")

    def _init_tab_coords(self):
        layout = QVBoxLayout(self.tab_coords)

        # 1. 导入 CSV
        h_csv = QHBoxLayout()
        self.lbl_csv_status = QLabel("No Data")
        self.lbl_csv_status.setStyleSheet("color: #888;")
        btn_load_csv = QPushButton("📄 Load CSV")
        btn_load_csv.setStyleSheet("background: #333; padding: 5px; border: 1px solid #555;")
        btn_load_csv.clicked.connect(self.load_csv)
        h_csv.addWidget(btn_load_csv);
        h_csv.addWidget(self.lbl_csv_status)
        layout.addLayout(h_csv)

        # ==========================================================
        # 🟢 [新增] 类型过滤器 (Type Filter)
        # ==========================================================
        grp_filter = QGroupBox("Cluster Type Filter (Linked Mode)")
        grp_filter.setStyleSheet("border: 1px solid #555; color: #00e676; margin-top: 5px;")
        h_flt = QHBoxLayout(grp_filter)

        self.chk_ch_cls = QCheckBox("Channel Cluster")
        self.chk_ch_cls.setChecked(True)
        self.chk_ch_cls.setToolTip("Include clusters detected within a single channel.")
        self.chk_ch_cls.setStyleSheet("color: #ccc;")
        self.chk_ch_cls.toggled.connect(self.apply_type_filter)

        self.chk_sp_cls = QCheckBox("Spatial Cluster")
        self.chk_sp_cls.setChecked(True)
        self.chk_sp_cls.setToolTip("Include clusters aggregated from multiple channels (spatial proximity).")
        self.chk_sp_cls.setStyleSheet("color: #ccc;")
        self.chk_sp_cls.toggled.connect(self.apply_type_filter)

        h_flt.addWidget(self.chk_ch_cls)
        h_flt.addWidget(self.chk_sp_cls)
        layout.addWidget(grp_filter)
        # ==========================================================

        # 2. 截图参数
        grp_param = QGroupBox("Snapshot Parameters")
        grp_param.setStyleSheet("border: 1px solid #555; color: #00e676; margin-top: 5px;")
        g_layout = QGridLayout(grp_param)
        g_layout.addWidget(QLabel("Capture Radius:"), 0, 0)
        self.sb_radius = QSpinBox();
        self.sb_radius.setRange(1, 100);
        self.sb_radius.setValue(5)
        self.sb_radius.setSuffix(" px");
        g_layout.addWidget(self.sb_radius, 0, 1)
        g_layout.addWidget(QLabel("Output Size:"), 1, 0)
        self.sb_size = QSpinBox();
        self.sb_size.setRange(10, 1000);
        self.sb_size.setValue(60)
        self.sb_size.setSuffix(" px");
        g_layout.addWidget(self.sb_size, 1, 1)
        layout.addWidget(grp_param)

        layout.addStretch()

    # 其他辅助函数
    def _init_tab_single(self, default_rect):
        layout = QVBoxLayout(self.tab_single)
        self.chk_full = QCheckBox("Export Full Image")
        self.chk_full.toggled.connect(self.toggle_single_coords)
        layout.addWidget(self.chk_full)
        self.grp_rect = QGroupBox("1. ROI (Center Based)")
        gl = QGridLayout(self.grp_rect)
        self.sb_x = QSpinBox();
        self.sb_x.setRange(0, 99999);
        self.sb_x.setPrefix("X: ")
        self.sb_y = QSpinBox();
        self.sb_y.setRange(0, 99999);
        self.sb_y.setPrefix("Y: ")
        self.sb_w = QSpinBox();
        self.sb_w.setRange(1, 99999);
        self.sb_w.setPrefix("W: ")
        self.sb_h = QSpinBox();
        self.sb_h.setRange(1, 99999);
        self.sb_h.setPrefix("H: ")
        if default_rect:
            self.sb_x.setValue(default_rect[0]);
            self.sb_y.setValue(default_rect[1])
            self.sb_w.setValue(default_rect[2]);
            self.sb_h.setValue(default_rect[3])
        gl.addWidget(self.sb_x, 0, 0);
        gl.addWidget(self.sb_y, 0, 1)
        gl.addWidget(self.sb_w, 1, 0);
        gl.addWidget(self.sb_h, 1, 1)
        layout.addWidget(self.grp_rect)
        self.grp_resize = QGroupBox("2. Resize")
        v_r = QVBoxLayout(self.grp_resize)
        self.chk_resize = QCheckBox("Enable Resize");
        self.chk_resize.toggled.connect(self.toggle_resize)
        v_r.addWidget(self.chk_resize)
        h_r = QHBoxLayout();
        self.sb_out_w = QSpinBox();
        self.sb_out_w.setRange(1, 9999);
        self.sb_out_w.setValue(60)
        self.sb_out_h = QSpinBox();
        self.sb_out_h.setRange(1, 9999);
        self.sb_out_h.setValue(60)
        h_r.addWidget(QLabel("Size:"));
        h_r.addWidget(self.sb_out_w);
        h_r.addWidget(self.sb_out_h)
        self.wid_res = QWidget();
        self.wid_res.setLayout(h_r);
        self.wid_res.setEnabled(False)
        v_r.addWidget(self.wid_res);
        layout.addWidget(self.grp_resize);
        layout.addStretch()

    def browse_in(self):
        d = QFileDialog.getExistingDirectory(self);

        if d: self.edt_in.setText(d)

    def browse_out(self):
        d = QFileDialog.getExistingDirectory(self);

        if d: self.edt_out.setText(d)

    def toggle_single_coords(self, c):
        self.grp_rect.setEnabled(not c); self.grp_resize.setEnabled(not c)

    def toggle_resize(self, c):
        self.wid_res.setEnabled(c)

    def load_csv(self):
        f, _ = QFileDialog.getOpenFileName(self, "CSV", "", "*.csv")
        if f:
            # 加载 CSV 时清空 Linked 数据，避免混淆
            self.linked_data = []
            try:
                c = []
                with open(f, 'r') as cf:
                    import csv
                    for r in csv.reader(cf):
                        if r:
                            try:
                                c.append((int(float(r[0])), int(float(r[1]))))
                            except:
                                pass
                self.coord_list = c
                self.lbl_csv_status.setText(f"Loaded {len(c)} pts (CSV)")
                # CSV 模式下禁用过滤器（因为不知道类型）
                self.chk_ch_cls.setEnabled(False)
                self.chk_sp_cls.setEnabled(False)
            except:
                pass

    def start_process(self):
        inp = self.edt_in.text();
        out = self.edt_out.text()
        if not inp or not out: return
        mode = {}
        if self.tabs.currentIndex() == 0:
            mode['mode'] = 'single'
            mode['is_full'] = self.chk_full.isChecked()
            cx, cy, w, h = self.sb_x.value(), self.sb_y.value(), self.sb_w.value(), self.sb_h.value()
            mode['rect'] = (cx - w // 2, cy - h // 2, w, h)
            mode['resize_enabled'] = self.chk_resize.isChecked()
            mode['resize_target'] = (self.sb_out_w.value(), self.sb_out_h.value())
        else:
            if not self.coord_list: return
            mode['mode'] = 'coords'
            mode['coord_list'] = self.coord_list
            mode['snap_params'] = (self.sb_radius.value(), self.sb_size.value())

        self.btn_run.setEnabled(False)
        self.worker = BatchCropWorker(inp, out, self.edt_filter.text(), mode)
        self.worker.log_signal.connect(self.txt_log.append)
        self.worker.progress_signal.connect(lambda c, t: self.pbar.setValue(int(c / t * 100)))
        self.worker.finished_signal.connect(lambda: self.btn_run.setEnabled(True))
        self.worker.start()

    pass
import os
import sys
import traceback
from pathlib import Path
import cv2
import numpy as np
import pyqtgraph as pg

# PyQt6 Imports
import PyQt6.QtWidgets
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QSplitter, QGroupBox, QComboBox, QSpinBox, QApplication,
    QMessageBox, QFrame, QListWidget, QTableView, QHeaderView,
    QDialog, QProgressBar, QTextEdit, QGraphicsRectItem
)
from PyQt6.QtCore import Qt, QSettings, QSortFilterProxyModel, pyqtSignal, QTimer
from PyQt6.QtGui import QIcon, QBrush, QColor, QRadialGradient, QPen

# Core Imports
from core.workers import SingleWorker, BatchWorker
from core.algorithm import CoreAlgorithm

# UI & Utils Imports
from utils.helpers import BASE_DIR, EXE_DIR, ExportHandler
from ui.widgets import Surface3DViewer, InteractiveHistogram, ZoomableGraphicsView, DefectTableModel
from ui.dialogs import SingleExportDialog, BatchProcessDialog, BatchCropDialog
from utils.logger import log


# ==========================================
# 🟢 全局异常钩子
# ==========================================
def exception_hook(exctype, value, traceback_obj):
    """捕获未处理的异常，防止程序闪退"""
    err_msg = "".join(traceback.format_exception(exctype, value, traceback_obj))
    log.critical(f"Uncaught Exception:\n{err_msg}")
    sys.__excepthook__(exctype, value, traceback_obj)


# ==========================================
# 🖥️ 主窗口类
# ==========================================
class CyberApp(QMainWindow):
    def __init__(self):
        super().__init__()
        log.info("Application Initializing...")
        self.setWindowTitle("Defect Pixel Nemesis // V5.0 Final")
        self.resize(1600, 900)

        # 状态变量
        self.current_single_dir = None
        self.current_file_path = None
        self.current_data_cache = []
        self.cursor_lines = []
        self.last_mouse_x = 0
        self.last_mouse_y = 0

        # 缓存
        self.cache_vis_raw = None
        self.cache_vis_grid = None
        self.cache_vis_clean = None
        self.cache_raw_img = None
        self.cache_clean_img = None

        # 3D 窗口
        self.win_3d = Surface3DViewer()

        # UI 初始化
        self.apply_theme()
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.init_single_mode()

        # 配置加载
        ini_path = os.path.join(EXE_DIR, "config.ini")
        self.settings = QSettings(ini_path, QSettings.Format.IniFormat)
        self.load_settings()

        # 拖放支持
        self.setAcceptDrops(True)

        # 防抖定时器

    # ==========================================
    # 🟢 核心业务逻辑
    # ==========================================
    def open_single_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if not d: return

        self.current_single_dir = Path(d)
        self.file_list.clear()

        extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.raw', '*.bin']
        files = []
        for ext in extensions:
            files.extend(list(self.current_single_dir.glob(ext)))
            files.extend(list(self.current_single_dir.glob(ext.upper())))

        files = sorted(list(set(files)))
        if not files:
            self.file_list.addItem("No images found.")
            return

        for f in files:
            self.file_list.addItem(f.name)

    def on_file_list_clicked(self, item):
        if not self.current_single_dir: return
        if item.text() == "No images found.": return

        filename = item.text()
        full_path = self.current_single_dir / filename
        self.trigger_analysis(str(full_path))

    def re_analyze_current(self):
        if self.current_file_path and Path(self.current_file_path).exists():
            self.trigger_analysis(self.current_file_path)

    def trigger_analysis(self, path):
        f_name = Path(path).name.lower()
        if "dark" in f_name:
            if self.combo_mode.currentIndex() != 0:
                self.combo_mode.setCurrentIndex(0)
        elif "mid" in f_name or "bright" in f_name:
            if self.combo_mode.currentIndex() != 1:
                self.combo_mode.setCurrentIndex(1)

        QApplication.processEvents()
        self.current_file_path = path

        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        self.btn_load.setEnabled(False)
        self.btn_load.setText("PROCESSING...")
        if hasattr(self, 'file_list'):
            self.file_list.setEnabled(False)

        params = self.get_params()
        self.worker = SingleWorker(path, params)
        self.worker.result_signal.connect(self.on_single_finished_wrapper)
        self.worker.error_occurred.connect(self.on_analysis_error)
        self.worker.start()

    def on_analysis_error(self, err_msg):
        QApplication.restoreOverrideCursor()
        self.btn_load.setText("🔄 RE-ANALYZE")
        self.btn_load.setEnabled(True)
        if hasattr(self, 'file_list'):
            self.file_list.setEnabled(True)
        log.error(f"Analysis Error: {err_msg}")
        QMessageBox.critical(self, "Processing Failed", err_msg)

    def on_single_finished_wrapper(self, vis_raw, vis_grid, data, img_raw):
        QApplication.restoreOverrideCursor()
        self.on_single_finished(vis_raw, vis_grid, data, img_raw)
        self.btn_load.setText("🔄 RE-ANALYZE CURRENT")
        self.btn_load.setEnabled(True)
        if hasattr(self, 'file_list'):
            self.file_list.setEnabled(True)

    def on_single_finished(self, vis_raw, vis_grid, data, img_raw):
        self.current_data_cache = data
        self.cache_vis_raw = vis_raw
        self.cache_vis_grid = vis_grid
        self.cache_raw_img = img_raw

        if img_raw is not None:
            vis_clean = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            if vis_raw is not None and len(vis_raw.shape) == 3 and len(vis_clean.shape) == 2:
                vis_clean = cv2.cvtColor(vis_clean, cv2.COLOR_GRAY2BGR)
            self.cache_vis_clean = vis_clean
            self.cache_clean_img = img_raw
        else:
            self.cache_vis_clean = None

        if vis_raw is None: return
        h, w = vis_raw.shape[:2]

        # 刷新显示 (适应窗口)
        self.toggle_view_image(self.combo_view.currentIndex(), maintain_view=False)

        # 统计
        params = self.get_params()
        stats = CoreAlgorithm.get_stats(data, (h, w), params['g_dist'])

        max_pts = self.sb_spec_pts.value()
        max_cls = self.sb_spec_cls.value()
        total_pts = stats['total_pts']
        total_cls = stats['white_cls'] + stats['black_cls']
        is_pass = (total_pts <= max_pts) and (total_cls <= max_cls)

        info_str = (f"(Pixels: {total_pts}, Groups: {total_cls})\n"
                    f"⚪W_Pix: {stats['white_pts']} (Grp:{stats['white_cls']})\n"
                    f"⚫B_Pix: {stats['black_pts']} (Grp:{stats['black_cls']})")

        if is_pass:
            self.lbl_result.setText(f"🟢 PASS\n{info_str}")
            self.lbl_result.setStyleSheet(
                "background-color: rgba(0, 230, 118, 0.15); color: #00e676; border: 2px solid #00e676; border-radius: 6px; font-weight: bold; font-size: 11pt;")
        else:
            self.lbl_result.setText(f"🔴 FAIL\n{info_str}")
            self.lbl_result.setStyleSheet(
                "background-color: rgba(255, 23, 68, 0.15); color: #ff1744; border: 2px solid #ff1744; border-radius: 6px; font-weight: bold; font-size: 11pt;")

        self.model.update_data(data)
        self.update_graph()

        if hasattr(self, 'hist_widget'):
            self.hist_widget.update_data(self.cache_raw_img)

    # ==========================================
    # 🟢 坐标转换与绘图逻辑
    # ==========================================
    def transform_coordinates(self, gx, gy, ch_idx):
        if self.combo_view.currentIndex() != 1:
            return gx, gy

        if not hasattr(self, 'cache_raw_img') or self.cache_raw_img is None:
            return gx, gy

        h_raw, w_raw = self.cache_raw_img.shape[:2]
        params = self.get_params()
        channels = params.get('ch', 1)

        if channels <= 1: return gx, gy

        step = int(np.sqrt(channels))
        sub_w = w_raw // step
        sub_h = h_raw // step

        idx_zero = ch_idx - 1
        row = idx_zero // step
        col = idx_zero % step

        off_x = col * sub_w
        off_y = row * sub_h
        local_x = gx // step
        local_y = gy // step

        return off_x + local_x, off_y + local_y

    def create_glow_brush(self, color_hex, alpha_center=200, alpha_edge=0):
        base_col = QColor(color_hex)
        grad = QRadialGradient(0.5, 0.5, 0.5)
        grad.setCoordinateMode(QRadialGradient.CoordinateMode.ObjectBoundingMode)
        c_center = QColor(base_col);
        c_center.setAlpha(alpha_center)
        c_edge = QColor(base_col);
        c_edge.setAlpha(alpha_edge)
        grad.setColorAt(0.0, c_center)
        grad.setColorAt(0.7, c_edge)
        grad.setColorAt(1.0, c_edge)
        return QBrush(grad)

    def update_graph(self):
        """刷新右下角散点图"""
        self.graph.clear()

        # 重建 FOV 框 (绿色透明)
        if hasattr(self, 'fov_box'):
            self.graph.addItem(self.fov_box)

        plot_item = self.graph.getPlotItem()
        # 🟢 [修复] 强制 X 轴显示在顶部
        plot_item.showAxis('top', True)
        plot_item.showAxis('bottom', False)

        if plot_item.legend:
            try:
                if plot_item.legend.scene(): plot_item.legend.scene().removeItem(plot_item.legend)
            except:
                pass
            plot_item.legend = None
        self.legend = self.graph.addLegend(offset=(10, 10))
        self.legend.setScale(0.8)
        self.legend.setBrush(pg.mkBrush((0, 0, 0, 150)))

        if not hasattr(self, 'current_data_cache') or self.current_data_cache is None or len(
                self.current_data_cache) == 0:
            return

        spots_bright, spots_dark = [], []
        spots_cls_ch, spots_cls_sp = [], []

        for r, d in enumerate(self.current_data_cache):
            def get_v(k):
                return d[k] if hasattr(d, 'dtype') else d.get(k)

            try:
                gx, gy = int(get_v('gx')), int(get_v('gy'))
                ch = int(get_v('ch'))
            except:
                continue

            px, py = self.transform_coordinates(gx, gy, ch)
            pt_data = {'pos': (px, py), 'data': r}

            raw_ftype = get_v('final_type')
            ftype = 0
            try:
                ftype = int(raw_ftype)
            except:
                s = str(raw_ftype)
                if "Channel" in s:
                    ftype = 1
                elif "Spatial" in s:
                    ftype = 2

            raw_pol = get_v('polarity')
            pol = 0
            try:
                pol = int(raw_pol)
            except:
                if "Dark" in str(raw_pol) or "Black" in str(raw_pol): pol = 1

            if ftype == 1:
                spots_cls_ch.append(pt_data)
            elif ftype == 2:
                spots_cls_sp.append(pt_data)
            elif pol == 1:
                spots_dark.append(pt_data)
            else:
                spots_bright.append(pt_data)

        if spots_bright:
            s = pg.ScatterPlotItem(size=12, pen=None, brush=self.create_glow_brush("#00e676"), symbol='o',
                                   name='Bright', hoverable=True)
            s.addPoints(spots_bright)
            s.sigClicked.connect(self.on_scatter_clicked)
            self.graph.addItem(s)

        if spots_dark:
            s = pg.ScatterPlotItem(size=12, pen=None, brush=self.create_glow_brush("#2979ff"), symbol='o', name='Dark',
                                   hoverable=True)
            s.addPoints(spots_dark)
            s.sigClicked.connect(self.on_scatter_clicked)
            self.graph.addItem(s)

        if spots_cls_ch:
            s = pg.ScatterPlotItem(size=18, pen=None, brush=self.create_glow_brush("#ffea00"), symbol='d',
                                   name='Ch-Cluster', hoverable=True)
            s.addPoints(spots_cls_ch)
            s.sigClicked.connect(self.on_scatter_clicked)
            self.graph.addItem(s)

        if spots_cls_sp:
            s = pg.ScatterPlotItem(size=18, pen=None, brush=self.create_glow_brush("#ff1744"), symbol='d',
                                   name='Sp-Cluster', hoverable=True)
            s.addPoints(spots_cls_sp)
            s.sigClicked.connect(self.on_scatter_clicked)
            self.graph.addItem(s)

        # 🟢 在 update_graph 方法末尾找到此处：
        img_for_size = self.cache_vis_grid if self.combo_view.currentIndex() == 1 else self.cache_vis_raw
        if img_for_size is not None:
            h, w = img_for_size.shape[:2]
            # 关键：设置 padding=0 强制坐标轴严格贴合图片边缘
            self.graph.setXRange(0, float(w), padding=0)
            self.graph.setYRange(0, float(h), padding=0)

    # ==========================================
    # 🟢 视图交互
    # ==========================================
    def toggle_view_image(self, index, maintain_view=False):
        if not hasattr(self, 'cache_vis_raw') or self.cache_vis_raw is None: return

        if index == 0:  # Raw View
            show_markers = self.chk_show_overlay.isChecked() if hasattr(self, 'chk_show_overlay') else True
            img_to_show = self.cache_vis_raw if show_markers else self.cache_vis_clean
            self.zoom_img.set_image(img_to_show, maintain_view=maintain_view)
            self.table.setEnabled(True)
        else:  # Grid View
            if (not hasattr(self, 'cache_vis_grid') or self.cache_vis_grid is None) and hasattr(self, 'cache_raw_img'):
                self.btn_load.setText("GENERATING GRID...")
                QApplication.processEvents()
                channels = self.get_params()['ch']
                self.cache_vis_grid = CoreAlgorithm.generate_channel_grid(self.cache_raw_img, channels)
                self.btn_load.setText("🔄 RE-ANALYZE")

            if self.cache_vis_grid is not None:
                self.zoom_img.set_image(self.cache_vis_grid, maintain_view=maintain_view)

        self.update_graph()

    def update_cursor_display(self, x, y, val):
        self.last_mouse_x = x
        self.last_mouse_y = y

        final_val = "N/A"
        if hasattr(self, 'cache_raw_img') and self.cache_raw_img is not None:
            h, w = self.cache_raw_img.shape[:2]
            if 0 <= x < w and 0 <= y < h:
                raw_pixel = self.cache_raw_img[y, x]
                if self.cache_raw_img.ndim == 3:
                    raw_val = np.max(raw_pixel)
                else:
                    raw_val = raw_pixel

                if self.cache_raw_img.dtype == np.uint16:
                    final_val = int(raw_val / 256)
                else:
                    final_val = int(raw_val)
        else:
            final_val = val

        self.lbl_cursor_info.setText(f"📍 X: {x:<4} Y: {y:<4} 💡 Val: {final_val}")

        if self.win_3d.isVisible():
            source_img = None
            if self.combo_view.currentIndex() == 1:
                if hasattr(self, 'zoom_img') and self.zoom_img.cv_img_ref is not None:
                    source_img = self.zoom_img.cv_img_ref
            else:
                if hasattr(self, 'cache_clean_img') and self.cache_clean_img is not None:
                    source_img = self.cache_clean_img

            if source_img is not None:
                roi_size = 50
                half = roi_size // 2
                h, w = source_img.shape[:2]
                x_s, x_e = max(0, x - half), min(w, x + half)
                y_s, y_e = max(0, y - half), min(h, y + half)
                roi = source_img[y_s:y_e, x_s:x_e]
                if len(roi.shape) == 3:
                    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                if roi.size > 0:
                    self.win_3d.update_surface(roi)

    def on_table_click(self, index):
        if not index.isValid(): return
        source_index = self.proxy_model.mapToSource(index)
        real_row = source_index.row()

        if not hasattr(self, 'current_data_cache') or real_row >= len(self.current_data_cache): return
        data = self.current_data_cache[real_row]

        def get_v(k):
            return data[k] if hasattr(data, 'dtype') else data.get(k)

        gx, gy = int(get_v('gx')), int(get_v('gy'))
        val = int(get_v('val'))

        raw_ftype = get_v('final_type')
        ftype_str = "Single"
        try:
            ft = int(raw_ftype)
            if ft == 1:
                ftype_str = "Channel_Cluster"
            elif ft == 2:
                ftype_str = "Spatial_Cluster"
        except:
            ftype_str = str(raw_ftype)

        info_text = (f"TYPE: <b>{ftype_str}</b> &nbsp;|&nbsp; "
                     f"LOC: <span style='color:#00e676'>({gx}, {gy})</span> &nbsp;|&nbsp; "
                     f"VAL: <span style='color:#ff9800'>{val}</span>")
        self.lbl_detail.setText(info_text)

        if not hasattr(self, 'cursor_lines'): self.cursor_lines = []
        plot_item = self.graph.getPlotItem()
        for line in self.cursor_lines:
            try:
                plot_item.removeItem(line)
            except:
                pass
        self.cursor_lines.clear()

        ch = int(get_v('ch'))
        target_x, target_y = self.transform_coordinates(gx, gy, ch)

        pen = pg.mkPen(color='#ffd740', width=2, style=Qt.PenStyle.DashLine)
        v_line = pg.InfiniteLine(pos=target_x, angle=90, pen=pen)
        h_line = pg.InfiniteLine(pos=target_y, angle=0, pen=pen)
        plot_item.addItem(v_line)
        plot_item.addItem(h_line)
        self.cursor_lines.extend([v_line, h_line])

        self.zoom_img.highlight_defect(target_x, target_y, size=30)

    def on_scatter_clicked(self, plot_item, points):
        if len(points) == 0: return
        p = points[0]
        x, y = int(p.pos().x()), int(p.pos().y())

        self.zoom_img.highlight_defect(x, y)

        row_idx = p.data()
        val = "N/A"
        if row_idx is not None and row_idx < len(self.current_data_cache):
            item_data = self.current_data_cache[row_idx]
            val = item_data['val'] if hasattr(item_data, 'dtype') else item_data.get('val')
            if hasattr(self, 'proxy_model'):
                source_idx = self.model.index(row_idx, 0)
                proxy_idx = self.proxy_model.mapFromSource(source_idx)
                if proxy_idx.isValid():
                    self.table.selectRow(proxy_idx.row())
                    self.table.scrollTo(proxy_idx)

        self.lbl_cursor_info.setText(f"📍 X: {x:<4} Y: {y:<4} 💡 Val: {val}")
        if self.win_3d.isVisible():
            self.update_cursor_display(x, y, val)

    def on_table_selection_change(self, current, previous):
        if current.isValid(): self.on_table_click(current)

    # ==========================================
    # 🟢 参数与配置
    # ==========================================
    def get_params(self):
        try:
            if not self.combo_mode: return {}
            idx = self.combo_mode.currentIndex()
        except:
            return {}

        mode = "Dark" if idx == 0 else "Bright"
        common = {
            "ch": int(self.cb_ch.currentText()),
            "fs": int(self.cb_fs.currentText()),
            "mode": mode
        }
        if mode == "Dark":
            return {**common, "thresh": self.sb_thresh_abs.value(), "ch_dist": self.sb_ch_dist_dark.value(),
                    "g_dist": self.sb_g_dist_dark.value()}
        else:
            return {**common, "thresh": self.sb_thresh_pct.value(), "ch_dist": self.sb_ch_dist_bright.value(),
                    "g_dist": self.sb_g_dist_bright.value()}

    def toggle_params(self):
        idx = self.combo_mode.currentIndex()
        if idx == 0:
            self.container_dark.show()
            self.container_bright.hide()
            self.hist_widget.thresh_line.show()
            self.hist_widget.set_line_pos(self.sb_thresh_abs.value())
            self.hist_widget.setTitle("Gray Distribution (Drag to set Threshold)")
        else:
            self.container_dark.hide()
            self.container_bright.show()
            self.hist_widget.thresh_line.hide()
            self.hist_widget.setTitle("Gray Distribution (Ref Only)")




    def on_param_toggle(self, checked):
        self.grp_param.setVisible(checked)
        self.btn_toggle_param.setText("▼ PARAMETERS & HISTOGRAM" if checked else "▶ PARAMETERS (Hidden)")

    def toggle_3d_window(self, checked):
        self.win_3d.setVisible(checked)

    def update_fov_box(self, rect):
        """接收主视图的可视区域并同步到小地图，修正坐标缩放偏差"""
        if not hasattr(self, 'fov_box'): return

        # 1. 动态获取当前正在显示的图片的实际像素尺寸 (修正 1/4 面积问题的关键)
        mw, mh = 0, 0

        # 如果当前是 Grid View，使用 Grid 图的宽高
        if self.combo_view.currentIndex() == 1 and self.cache_vis_grid is not None:
            mh, mw = self.cache_vis_grid.shape[:2]
        # 如果是 Raw View，使用 Raw 图的宽高
        elif self.cache_vis_raw is not None:
            mh, mw = self.cache_vis_raw.shape[:2]

        # 安全检查：如果图片还没加载，不执行
        if mw == 0 or mh == 0: return

        # 2. 坐标限制 (Clamping)：确保框体左上角不小于 0，且不超过图片最大边缘
        x = max(0, min(mw, rect.x()))
        y = max(0, min(mh, rect.y()))

        # 3. 尺寸限制 (Clamping)：确保框体右侧和下方不会超出图片边缘
        # rect.width() 是当前屏幕能看到的“主图像素宽度”
        w = min(rect.width(), mw - x)
        h = min(rect.height(), mh - y)

        # 4. 极端情况处理：如果主视图缩小到比全图还大（边缘留白），强制框体填满
        if x + w > mw: w = mw - x
        if y + h > mh: h = mh - y

        # 5. 更新矩形位置
        self.fov_box.setRect(x, y, w, h)
    # ==========================================
    # 🟢 弹窗与导出
    # ==========================================
    def open_batch_dialog(self):
        BatchProcessDialog(self.get_params(), self).exec()

    def open_crop_dialog(self):
        default_size = 10
        cx, cy = 0, 0
        if hasattr(self, 'zoom_img'):
            c = self.zoom_img.mapToScene(self.zoom_img.viewport().rect().center())
            cx, cy = int(c.x()), int(c.y())

        cluster_data = []
        if hasattr(self, 'current_data_cache') and self.current_data_cache is not None:
            for d in self.current_data_cache:
                try:
                    raw_ftype = d['final_type'] if hasattr(d, 'dtype') else d.get('final_type')
                    is_cluster = False
                    if isinstance(raw_ftype, int):
                        is_cluster = (raw_ftype != 0)
                    else:
                        is_cluster = "Cluster" in str(raw_ftype)

                    if is_cluster:
                        gx = int(d['gx'] if hasattr(d, 'dtype') else d.get('gx'))
                        gy = int(d['gy'] if hasattr(d, 'dtype') else d.get('gy'))
                        cluster_data.append({'x': gx, 'y': gy, 'type': raw_ftype})
                except:
                    pass

        dlg = BatchCropDialog(self, (cx, cy, default_size, default_size), initial_data=cluster_data)
        if self.current_single_dir:
            dlg.edt_in.setText(str(self.current_single_dir))
            dlg.edt_out.setText(str(self.current_single_dir / "crop_output"))
        dlg.exec()

    def export_current_data(self):
        if not hasattr(self, 'current_data_cache') or self.current_data_cache is None or len(
                self.current_data_cache) == 0:
            QMessageBox.warning(self, "Warning", "No data to export.")
            return

        source_img = self.cache_clean_img if hasattr(self,
                                                     'cache_clean_img') and self.cache_clean_img is not None else None
        if source_img is None and hasattr(self, 'zoom_img'): source_img = self.zoom_img.cv_img_ref

        if source_img is None:
            QMessageBox.warning(self, "Error", "Source image lost.")
            return

        dlg = SingleExportDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted: return

        snap_radius, snap_size, save_details = dlg.get_settings()
        default_name = Path(self.current_file_path).stem if self.current_file_path else "Report"
        save_dir = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if not save_dir: return

        try:
            self.btn_export_single.setText("Saving...")
            self.btn_export_single.setEnabled(False)
            QApplication.processEvents()

            params = self.get_params()
            h, w = source_img.shape[:2]
            stats = CoreAlgorithm.get_stats(self.current_data_cache, (h, w), params['g_dist'])
            specs = (self.sb_spec_pts.value(), self.sb_spec_cls.value())

            f = ExportHandler.save_report(
                self.current_data_cache, source_img, default_name, save_dir,
                stats, specs, save_details, (snap_radius, snap_size)
            )
            QMessageBox.information(self, "Success", f"Saved to:\n{f}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export Failed:\n{e}")
        finally:
            self.btn_export_single.setText("💾 Export")
            self.btn_export_single.setEnabled(True)

    # ==========================================
    # 🟢 杂项事件
    # ==========================================
    def closeEvent(self, event):
        self.save_settings()
        super().closeEvent(event)

    def save_settings(self):
        self.settings.setValue("params/mode_idx", self.combo_mode.currentIndex())
        self.settings.setValue("params/ch_idx", self.cb_ch.currentIndex())
        self.settings.setValue("params/fs_idx", self.cb_fs.currentIndex())
        self.settings.setValue("params/dark/thresh", self.sb_thresh_abs.value())
        self.settings.setValue("params/dark/ch_dist", self.sb_ch_dist_dark.value())
        self.settings.setValue("params/dark/g_dist", self.sb_g_dist_dark.value())
        self.settings.setValue("params/bright/pct", self.sb_thresh_pct.value())
        self.settings.setValue("params/bright/ch_dist", self.sb_ch_dist_bright.value())
        self.settings.setValue("params/bright/g_dist", self.sb_g_dist_bright.value())
        self.settings.setValue("specs/max_pts", self.sb_spec_pts.value())
        self.settings.setValue("specs/max_cls", self.sb_spec_cls.value())
        if self.current_single_dir:
            self.settings.setValue("paths/last_dir", str(self.current_single_dir))

    def load_settings(self):
        def g(k, d):
            return int(self.settings.value(k, d))

        self.combo_mode.setCurrentIndex(g("params/mode_idx", 0))
        self.cb_ch.setCurrentIndex(g("params/ch_idx", 1))
        self.cb_fs.setCurrentIndex(g("params/fs_idx", 1))
        self.sb_thresh_abs.setValue(g("params/dark/thresh", 50))
        self.sb_ch_dist_dark.setValue(g("params/dark/ch_dist", 3))
        self.sb_g_dist_dark.setValue(g("params/dark/g_dist", 5))
        self.sb_thresh_pct.setValue(g("params/bright/pct", 30))
        self.sb_ch_dist_bright.setValue(g("params/bright/ch_dist", 3))
        self.sb_g_dist_bright.setValue(g("params/bright/g_dist", 5))
        self.sb_spec_pts.setValue(g("specs/max_pts", 100))
        self.sb_spec_cls.setValue(g("specs/max_cls", 0))
        ld = self.settings.value("paths/last_dir", "")
        if ld and os.path.exists(ld):
            self.current_single_dir = Path(ld)
            self.file_list.clear()
            exts = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff', '*.raw', '*.bin']
            files = []
            for e in exts: files.extend(
                list(self.current_single_dir.glob(e)) + list(self.current_single_dir.glob(e.upper())))
            files = sorted(list(set(files)))
            for f in files: self.file_list.addItem(f.name)
        self.toggle_params()

    def keyPressEvent(self, event):
        step = 50
        if event.key() == Qt.Key.Key_W:
            self.zoom_img.pan_view(0, -step)
        elif event.key() == Qt.Key.Key_S:
            self.zoom_img.pan_view(0, step)
        elif event.key() == Qt.Key.Key_A:
            self.zoom_img.pan_view(-step, 0)
        elif event.key() == Qt.Key.Key_D:
            self.zoom_img.pan_view(step, 0)
        elif event.key() == Qt.Key.Key_Space:
            if hasattr(self, 'combo_view'):
                self.combo_view.setCurrentIndex(1 - self.combo_view.currentIndex())
        elif event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.key() == Qt.Key.Key_Right:
                self.switch_image(1)
            elif event.key() == Qt.Key.Key_Left:
                self.switch_image(-1)
        super().keyPressEvent(event)

    def switch_image(self, d):
        if not hasattr(self, 'file_list') or self.file_list.count() == 0: return
        curr = self.file_list.currentRow()
        new_r = curr + d
        if 0 <= new_r < self.file_list.count():
            self.file_list.setCurrentRow(new_r)
            self.on_file_list_clicked(self.file_list.item(new_r))
            self.file_list.scrollToItem(self.file_list.item(new_r))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if not urls: return
        p = Path(urls[0].toLocalFile())
        if p.is_dir():
            self.current_single_dir = p
            self.settings.setValue("paths/last_dir", str(p))
            self.load_settings()
        elif p.is_file():
            pd = p.parent
            if self.current_single_dir != pd:
                self.current_single_dir = pd
                self.settings.setValue("paths/last_dir", str(pd))
                self.load_settings()

            items = self.file_list.findItems(p.name, Qt.MatchFlag.MatchExactly)
            if items:
                r = self.file_list.row(items[0])
                self.file_list.setCurrentRow(r)
                self.on_file_list_clicked(items[0])
            else:
                self.trigger_analysis(str(p))

    def apply_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #121212; color: #e0e0e0; font-family: 'Segoe UI'; font-size: 10pt; }
            QPushButton { background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #2d2d2d, stop:1 #1a1a1a); border: 1px solid #444; border-bottom: 2px solid #333; color: #ccc; padding: 6px 12px; border-radius: 4px; font-weight: bold; }
            QPushButton:hover { border: 1px solid #00e676; color: #00e676; }
            QPushButton:pressed { background-color: #00e676; color: #000; }
            QLineEdit, QSpinBox, QComboBox { background-color: #0f0f0f; border: 1px solid #333; padding: 5px; color: #00e676; font-family: 'Consolas'; border-radius: 3px; }
            QGroupBox { border: 1px solid #333; margin-top: 20px; font-weight: bold; color: #888; border-radius: 4px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; background-color: #121212; color: #00e676; }
            QListWidget, QTableView { background-color: #0a0a0a; border: 1px solid #333; outline: none; }
            QListWidget::item:selected, QTableView::item:selected { background-color: rgba(0, 230, 118, 0.2); border: 1px solid #00e676; color: #fff; }
            QHeaderView::section { background-color: #1a1a1a; color: #888; padding: 6px; border: none; border-bottom: 2px solid #333; }
            QSplitter::handle { background-color: #222; }
            QSplitter::handle:hover { background-color: #00e676; }
        """)

    def init_single_mode(self):
        layout = QHBoxLayout(self.main_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)

        left_frame = QFrame()
        left_frame.setMinimumWidth(380)
        v_layout = QVBoxLayout(left_frame)
        v_splitter = QSplitter(Qt.Orientation.Vertical)
        v_layout.addWidget(v_splitter)

        f_wid = QWidget();
        f_lay = QVBoxLayout(f_wid)
        grp = QGroupBox("FILE BROWSER")
        l = QVBoxLayout(grp)
        self.btn_sel_single_dir = QPushButton("📂 Open Folder");
        self.btn_sel_single_dir.clicked.connect(self.open_single_folder);
        l.addWidget(self.btn_sel_single_dir)
        self.file_list = QListWidget();
        self.file_list.itemClicked.connect(self.on_file_list_clicked);
        l.addWidget(self.file_list)
        hb = QHBoxLayout()
        self.btn_open_batch = QPushButton("⚡ BATCH");
        self.btn_open_batch.clicked.connect(self.open_batch_dialog);
        hb.addWidget(self.btn_open_batch)
        self.btn_crop_tool = QPushButton("✂️ CROP");
        self.btn_crop_tool.clicked.connect(self.open_crop_dialog);
        hb.addWidget(self.btn_crop_tool)
        l.addLayout(hb);
        f_lay.addWidget(grp);
        v_splitter.addWidget(f_wid)

        p_wid = QWidget();
        p_lay = QVBoxLayout(p_wid)
        self.btn_toggle_param = QPushButton("▼ PARAMETERS");
        self.btn_toggle_param.setCheckable(True);
        self.btn_toggle_param.setChecked(True);
        self.btn_toggle_param.toggled.connect(self.on_param_toggle);
        p_lay.addWidget(self.btn_toggle_param)
        self.grp_param = QGroupBox()
        gp = QVBoxLayout(self.grp_param)
        self.combo_mode = QComboBox();
        self.combo_mode.addItems(["🌑 Dark Field", "☀️ Bright Field"]);
        self.combo_mode.currentIndexChanged.connect(self.toggle_params)
        gp.addWidget(QLabel("MODE:"));
        gp.addWidget(self.combo_mode)
        self.hist_widget = InteractiveHistogram();
        self.hist_widget.setFixedHeight(100);

        gp.addWidget(self.hist_widget)
        h1 = QHBoxLayout();
        self.cb_ch = QComboBox();
        self.cb_ch.addItems(["4", "16", "64"]);
        self.cb_ch.setCurrentIndex(1);
        self.cb_fs = QComboBox();
        self.cb_fs.addItems(["3", "5", "7"]);
        self.cb_fs.setCurrentIndex(1);
        h1.addWidget(QLabel("CH:"));
        h1.addWidget(self.cb_ch);
        h1.addWidget(QLabel("Filter:"));
        h1.addWidget(self.cb_fs);
        gp.addLayout(h1)
        self.container_dark = QWidget();
        dl = QVBoxLayout(self.container_dark);
        dl.setContentsMargins(0, 0, 0, 0)
        dh1 = QHBoxLayout();
        self.sb_thresh_abs = QSpinBox();
        self.sb_thresh_abs.setRange(0, 255);
        dh1.addWidget(QLabel("Thresh:"));
        dh1.addWidget(self.sb_thresh_abs);
        dl.addLayout(dh1)
        dh2 = QHBoxLayout();
        self.sb_ch_dist_dark = QSpinBox();
        self.sb_g_dist_dark = QSpinBox();
        dh2.addWidget(QLabel("Ch Dist:"));
        dh2.addWidget(self.sb_ch_dist_dark);
        dh2.addWidget(QLabel("Global:"));
        dh2.addWidget(self.sb_g_dist_dark);
        dl.addLayout(dh2)
        gp.addWidget(self.container_dark)
        self.container_bright = QWidget();
        bl = QVBoxLayout(self.container_bright);
        bl.setContentsMargins(0, 0, 0, 0)
        bh1 = QHBoxLayout();
        self.sb_thresh_pct = QSpinBox();
        self.sb_thresh_pct.setRange(1, 100);
        bh1.addWidget(QLabel("Contrast %:"));
        bh1.addWidget(self.sb_thresh_pct);
        bl.addLayout(bh1)
        bh2 = QHBoxLayout();
        self.sb_ch_dist_bright = QSpinBox();
        self.sb_g_dist_bright = QSpinBox();
        bh2.addWidget(QLabel("Ch Dist:"));
        bh2.addWidget(self.sb_ch_dist_bright);
        bh2.addWidget(QLabel("Global:"));
        bh2.addWidget(self.sb_g_dist_bright);
        bl.addLayout(bh2)
        gp.addWidget(self.container_bright)
        p_lay.addWidget(self.grp_param)
        hs = QHBoxLayout();
        self.sb_spec_pts = QSpinBox();
        self.sb_spec_pts.setRange(0, 99999);
        self.sb_spec_cls = QSpinBox();
        self.sb_spec_cls.setRange(0, 999);
        hs.addWidget(QLabel("Max Pts:"));
        hs.addWidget(self.sb_spec_pts);
        hs.addWidget(QLabel("Max Cls:"));
        hs.addWidget(self.sb_spec_cls);
        p_lay.addLayout(hs)
        self.btn_load = QPushButton("🔄 RE-ANALYZE");
        self.btn_load.clicked.connect(self.re_analyze_current);
        p_lay.addWidget(self.btn_load)
        hr = QHBoxLayout();
        self.lbl_result = QLabel("READY");
        self.lbl_result.setAlignment(Qt.AlignmentFlag.AlignCenter);
        self.lbl_result.setStyleSheet(
            "font-size: 16pt; font-weight: bold; border: 2px dashed #333; border-radius: 6px;");
        hr.addWidget(self.lbl_result)
        self.lbl_detail = QLabel("Wait Selection");
        self.lbl_detail.setStyleSheet("background: #0f0f0f; border: 1px solid #333; padding: 5px;");
        hr.addWidget(self.lbl_detail, stretch=1);
        p_lay.addLayout(hr)
        v_splitter.addWidget(p_wid)

        b_wid = QWidget();
        bl = QVBoxLayout(b_wid)
        ht = QHBoxLayout();
        ht.addWidget(QLabel("📋 Defect List"));
        ht.addStretch();
        self.btn_export_single = QPushButton("💾 Export");
        self.btn_export_single.clicked.connect(self.export_current_data);
        ht.addWidget(self.btn_export_single);
        bl.addLayout(ht)
        self.table = QTableView();
        self.model = DefectTableModel([]);
        self.proxy_model = QSortFilterProxyModel();
        self.proxy_model.setSourceModel(self.model);
        self.table.setModel(self.proxy_model);
        self.table.setSortingEnabled(True);
        self.table.clicked.connect(self.on_table_click);
        self.table.selectionModel().currentChanged.connect(self.on_table_selection_change);
        bl.addWidget(self.table)
        v_splitter.addWidget(b_wid)

        right_frame = QWidget();
        r_lay = QVBoxLayout(right_frame)
        hi = QHBoxLayout();
        self.combo_view = QComboBox();
        self.combo_view.addItems(["🖼️ Raw View", "🔲 Grid View"]);
        self.combo_view.currentIndexChanged.connect(lambda i: self.toggle_view_image(i));
        hi.addWidget(self.combo_view)
        self.btn_3d = QPushButton("⛰️ 3D");
        self.btn_3d.setCheckable(True);
        self.btn_3d.clicked.connect(self.toggle_3d_window);
        hi.addWidget(self.btn_3d)

        self.chk_show_overlay = PyQt6.QtWidgets.QCheckBox("Show Markers");
        self.chk_show_overlay.setChecked(True)
        self.chk_show_overlay.toggled.connect(
            lambda _: self.toggle_view_image(self.combo_view.currentIndex(), maintain_view=True));
        hi.addWidget(self.chk_show_overlay)

        hi.addStretch();
        hi.addWidget(QLabel("Shortcuts: [WASD] Pan | [Space] View | [Ctrl+Arr] File"));
        self.lbl_cursor_info = QLabel("X:--- Y:---");
        self.lbl_cursor_info.setStyleSheet(
            "font-family: Consolas; color: #00e676; background: black; padding: 4px; border: 1px solid #333;");
        hi.addWidget(self.lbl_cursor_info);
        r_lay.addLayout(hi)
        self.zoom_img = ZoomableGraphicsView();
        self.zoom_img.mouse_moved_signal.connect(self.update_cursor_display);
        self.zoom_img.view_changed_signal.connect(self.update_fov_box);
        r_lay.addWidget(self.zoom_img, stretch=2)

        # 🟢 [修改] 坐标轴位置 & 初始化 FOV 框
        self.graph = pg.PlotWidget(background='#0f0f0f');
        self.graph.setMouseEnabled(x=True, y=True)  # 禁用鼠标
        self.graph.hideButtons()

        plot_item = self.graph.getPlotItem()
        plot_item.invertY(True)
        plot_item.showAxis('top', True)  # X轴上移
        plot_item.showAxis('bottom', False)  # 隐藏下X轴

        # 初始化绿色透明框
        self.fov_box = QGraphicsRectItem()
        self.fov_box.setPen(QPen(QColor("#00e676"), 2))
        self.fov_box.setBrush(QBrush(QColor(0, 230, 118, 40)))
        self.graph.addItem(self.fov_box)

        r_lay.addWidget(self.graph, stretch=2)

        splitter.addWidget(left_frame);
        splitter.addWidget(right_frame);
        splitter.setSizes([450, 900])


if __name__ == "__main__":
    sys.excepthook = exception_hook
    app = QApplication(sys.argv)
    window = CyberApp()
    window.show()
    sys.exit(app.exec())
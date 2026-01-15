import numpy as np
import cv2
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QGraphicsView, QGraphicsScene,
    QGraphicsItem, QGraphicsPixmapItem, QTableView, QHeaderView
)
from PyQt6.QtCore import (
    Qt, pyqtSignal, QRectF, QAbstractTableModel, QModelIndex, QRect
)
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QImage, QPixmap, QFont
)
from PyQt6.QtCore import QPointF

# ==========================================
# 1. DefectTableModel (高性能数据模型)
# ==========================================
class DefectTableModel(QAbstractTableModel):
    def __init__(self, data=None):
        super().__init__()
        self._data = data or []
        self._headers = ["CH", "Type", "Polarity", "X", "Y", "Val", "Size"]

    def rowCount(self, parent=QModelIndex()):
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        return len(self._headers)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None

        row = index.row()
        col = index.column()
        item = self._data[row]

        if role == Qt.ItemDataRole.DisplayRole:
            if col == 0: return item['ch']
            if col == 1: return item['final_type']
            if col == 2: return "White" if item.get('polarity') == 'Bright' else "Black"
            if col == 3: return item['gx']
            if col == 4: return item['gy']
            if col == 5: return item['val']
            if col == 6: return item.get('size', 1)

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return None

    def update_data(self, new_data):
        self.beginResetModel()
        self._data = new_data
        self.endResetModel()

# ==========================================
# 2. MiniMapOverlay (鹰眼小地图 - 核心修复)
# ==========================================
class MiniMapOverlay(QWidget):
    def __init__(self, parent_view):
        super().__init__(parent_view)
        self.view = parent_view

        self.setFixedSize(240, 160)

        self.setStyleSheet("""
            background-color: rgba(0, 0, 0, 255);
            border: 2px solid #00e676;
            border-radius: 4px;
        """)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self.preview_pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.hide()

    # 🟢 [核心修复] 增加 scene_size 参数
    def update_data(self, full_pixmap, scene_size=None):
        if full_pixmap is None:
            self.hide()
            return

        self.show()

        w_target = self.width() - 8
        h_target = self.height() - 8

        self.preview_pixmap = full_pixmap.scaled(
            w_target, h_target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        # 🟢 [核心修复] 计算比例时使用真实场景尺寸 (scene_size)
        # 之前的 bug 是因为使用了 full_pixmap.width() (可能是缩小的预览图尺寸)，
        # 导致 scale_factor 偏大，从而导致白框偏大 (或位置偏差)。
        if scene_size and scene_size[0] > 0:
            self.scale_factor = self.preview_pixmap.width() / scene_size[0]
        elif full_pixmap.width() > 0:
            self.scale_factor = self.preview_pixmap.width() / full_pixmap.width()
        else:
            self.scale_factor = 1

        self.offset_x = (self.width() - self.preview_pixmap.width()) / 2
        self.offset_y = (self.height() - self.preview_pixmap.height()) / 2

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if not self.preview_pixmap: return

        # 1. 绘制缩略图
        painter.drawPixmap(int(self.offset_x), int(self.offset_y), self.preview_pixmap)

        # 2. 绘制视口框
        if self.view.scene():
            viewport_rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()

            mx = viewport_rect.x() * self.scale_factor + self.offset_x
            my = viewport_rect.y() * self.scale_factor + self.offset_y
            mw = viewport_rect.width() * self.scale_factor
            mh = viewport_rect.height() * self.scale_factor

            # 绘制内部视野框
            pen = QPen(QColor("#ffffff"))
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(QColor(255, 255, 255, 30))
            painter.drawRect(QRectF(mx, my, mw, mh))

            # 绘制绿色外边框
            border_pen = QPen(QColor("#00e676"))
            border_pen.setWidth(4)
            painter.setPen(border_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.rect().adjusted(2, 2, -2, -2))

    def mousePressEvent(self, event):
        self._navigate(event.position())

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._navigate(event.position())

    def _navigate(self, pos):
        if not self.preview_pixmap: return
        cx = (pos.x() - self.offset_x) / self.scale_factor
        cy = (pos.y() - self.offset_y) / self.scale_factor
        self.view.centerOn(cx, cy)

# ==========================================
# 3. Surface3DViewer (3D 地形查看器)
# ==========================================
class Surface3DViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D PIXEL INTENSITY SURFACE")
        self.resize(600, 600)
        self.setWindowFlags(Qt.WindowType.Window)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('#111111')
        layout.addWidget(self.view)

        axis = gl.GLAxisItem()
        axis.setSize(x=55, y=55, z=85)
        axis.translate(-25, -25, 0)
        self.view.addItem(axis)

        self.add_ruler_labels()

        g = gl.GLGridItem()
        g.setSize(x=60, y=60, z=0)
        g.setSpacing(x=5, y=5, z=0)
        self.view.addItem(g)

        dummy_z = np.zeros((50, 50))
        self.p1 = gl.GLSurfacePlotItem(z=dummy_z, computeNormals=True, smooth=False, shader='shaded')
        self.p1.translate(-25, -25, 0)
        self.view.addItem(self.p1)

        self.view.setCameraPosition(distance=90, elevation=30, azimuth=45)

        pos = np.array([0.0, 0.33, 0.66, 1.0])
        color = np.array([
            [0, 0, 140, 255], [0, 255, 255, 255],
            [255, 255, 0, 255], [255, 0, 0, 255]
        ], dtype=np.ubyte)
        self.colormap = pg.ColorMap(pos, color)

        lbl = QLabel("XYZ Scale: [X/Y] Pixel Offset (0-50) | [Z] Intensity Value (0-255)")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background: #000; color: #aaa; font-size: 9pt; padding: 4px; border-top: 1px solid #333;")
        layout.addWidget(lbl)

    def add_ruler_labels(self):
        def add_text(x, y, z, text):
            t = gl.GLTextItem(pos=(x, y, z), text=text, font=QFont('Arial', 8))
            self.view.addItem(t)

        for i in range(0, 51, 10):
            world_x = i - 25
            add_text(world_x, -28, 0, str(i))

        for i in range(0, 51, 10):
            world_y = i - 25
            add_text(-28, world_y, 0, str(i))

        for val in range(0, 256, 50):
            height = val / 3.0
            add_text(-28, -28, height, str(val))

    def update_surface(self, roi_data):
        if roi_data is None: return
        z = roi_data.astype(np.float32)
        h, w = z.shape
        z_display = z / 3.0
        norm = z / 255.0
        colors = self.colormap.map(norm, mode='float')
        colors = colors.reshape(h * w, 4)
        self.p1.setData(z=z_display, colors=colors)

# ==========================================
# 4. InteractiveHistogram (交互式直方图)
# ==========================================
class InteractiveHistogram(pg.PlotWidget):
    threshold_changed_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent, background='#1a1a1a')
        self.setTitle("Pixel Intensity Distribution (All Channels)", color='#ccc', size='10pt')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setMouseEnabled(x=False, y=True)
        self.hideButtons()

        left_axis = self.getPlotItem().getAxis('left')
        left_axis.setWidth(45)
        self.setLabel('bottom', 'DN Value (8-bit Mapped)', units='')
        self.setLabel('left', 'Pixel Count', units='')

        self.curve = self.plot(stepMode=True, fillLevel=0,
                               brush=pg.mkBrush(0, 230, 118, 100),
                               pen=pg.mkPen('#00e676', width=1))

        self.thresh_line = pg.InfiniteLine(pos=50, angle=90, movable=True,
                                           pen=pg.mkPen('#ff1744', width=2, style=Qt.PenStyle.DashLine),
                                           hoverPen=pg.mkPen('#ff1744', width=4))
        self.thresh_line.sigPositionChangeFinished.connect(self.on_line_dragged)
        self.addItem(self.thresh_line)

        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1, style=Qt.PenStyle.DotLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=1, style=Qt.PenStyle.DotLine))
        self.addItem(self.v_line)
        self.addItem(self.h_line)

        self.scene().sigMouseMoved.connect(self.on_mouse_move)
        self.setXRange(0, 256, padding=0)
        self.setYRange(0, 100)
        self.current_hist = None

    def update_data(self, img):
        if img is None: return
        raw_data = img.flatten()
        if raw_data.dtype == np.uint16:
            data_to_plot = (raw_data >> 8).astype(np.uint8)
        else:
            data_to_plot = raw_data.astype(np.uint8)

        hist = np.bincount(data_to_plot, minlength=256)
        if len(hist) > 256: hist = hist[:256]
        self.current_hist = hist

        if len(self.current_hist) > 1:
            valid_data = self.current_hist[1:]
            if len(valid_data) > 0:
                peak_val = np.max(valid_data)
                self.setYRange(0, float(peak_val) * 1.2)

        x = np.arange(257)
        self.curve.setData(x, self.current_hist)

    def on_mouse_move(self, pos):
        if self.sceneBoundingRect().contains(pos):
            mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
            x_val = mouse_point.x()
            if 0 <= x_val <= 255 and self.current_hist is not None:
                idx = int(x_val)
                if idx < len(self.current_hist):
                    y_val = self.current_hist[idx]
                    self.v_line.setPos(x_val)
                    self.h_line.setPos(y_val)
                    self.setTitle(
                        f"<span style='color: #ccc'>DN Value: {idx}</span>  |  "
                        f"<span style='color: #00e676'>Count: {int(y_val)}</span>",
                        size='10pt'
                    )

    def on_line_dragged(self):
        val = int(self.thresh_line.value())
        val = max(0, min(255, val))
        self.threshold_changed_signal.emit(val)

    def set_line_pos(self, val):
        self.thresh_line.setValue(val)

# ==========================================
# 5. LazyGraphicsItem (懒加载图元)
# ==========================================
class LazyGraphicsItem(QGraphicsItem):
    def __init__(self, cv_img):
        super().__init__()
        self.cv_img = cv_img
        self.h, self.w = cv_img.shape[:2]

        max_dim = 2000
        scale = min(1.0, max_dim / max(self.h, self.w))
        if scale < 1.0:
            preview_img = cv2.resize(cv_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            preview_img = cv_img

        self.preview_pixmap = self._cv2_to_qpixmap(preview_img)
        self.rect = QRectF(0, 0, self.w, self.h)

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget):
        painter.drawPixmap(self.rect, self.preview_pixmap, QRectF(self.preview_pixmap.rect()))
        lod = option.levelOfDetailFromTransform(painter.worldTransform())

        if lod > 0.5:
            exposed = option.exposedRect
            x = max(0, int(exposed.x()))
            y = max(0, int(exposed.y()))
            rw = int(exposed.width()) + 2
            rh = int(exposed.height()) + 2

            if x + rw > self.w: rw = self.w - x
            if y + rh > self.h: rh = self.h - y

            if rw > 0 and rh > 0:
                sub_img = self.cv_img[y:y + rh, x:x + rw]
                h_sub, w_sub = sub_img.shape[:2]

                if len(sub_img.shape) == 2:
                    qimg = QImage(sub_img.data, w_sub, h_sub, w_sub, QImage.Format.Format_Grayscale8)
                else:
                    sub_rgb = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
                    qimg = QImage(sub_rgb.data, w_sub, h_sub, w_sub * 3, QImage.Format.Format_RGB888)

                painter.drawImage(QRectF(x, y, w_sub, h_sub), qimg)

    def _cv2_to_qpixmap(self, img):
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)

# ==========================================
# 6. ZoomableGraphicsView (主视图 - 修复版)
# ==========================================
class ZoomableGraphicsView(QGraphicsView):
    mouse_moved_signal = pyqtSignal(int, int, str)
    view_changed_signal = pyqtSignal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#111"))
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)

        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)

        self.img_item = None
        self.cv_img_ref = None
        self.highlight_item = None
        self.minimap = MiniMapOverlay(self)

    def drawForeground(self, painter, rect):
        super().drawForeground(painter, rect)
        scene_rect = self.mapToScene(self.viewport().rect()).boundingRect()
        l, t, w, h = scene_rect.left(), scene_rect.top(), scene_rect.width(), scene_rect.height()
        scale_factor = self.transform().m11()
        if scale_factor == 0: return

        line_width = 2.0 / scale_factor
        corner_len = 20.0 / scale_factor
        margin = 15.0 / scale_factor
        text_size = 12.0 / scale_factor

        painter.save()
        neon_color = QColor("#00e676")
        pen = QPen(neon_color)
        pen.setWidthF(line_width)
        pen.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)

        # 绘制角标
        painter.drawLine(QPointF(l + margin, t + margin + corner_len), QPointF(l + margin, t + margin))
        painter.drawLine(QPointF(l + margin, t + margin), QPointF(l + margin + corner_len, t + margin))
        painter.drawLine(QPointF(l + w - margin - corner_len, t + margin), QPointF(l + w - margin, t + margin))
        painter.drawLine(QPointF(l + w - margin, t + margin), QPointF(l + w - margin, t + margin + corner_len))
        painter.drawLine(QPointF(l + margin, t + h - margin - corner_len), QPointF(l + margin, t + h - margin))
        painter.drawLine(QPointF(l + margin, t + h - margin), QPointF(l + margin + corner_len, t + h - margin))
        painter.drawLine(QPointF(l + w - margin - corner_len, t + h - margin), QPointF(l + w - margin, t + h - margin))
        painter.drawLine(QPointF(l + w - margin, t + h - margin), QPointF(l + w - margin, t + h - margin - corner_len))

        # 绘制十字
        center_x, center_y = l + w / 2, t + h / 2
        cross_len = 10.0 / scale_factor
        gap = 5.0 / scale_factor
        pen.setColor(QColor(0, 230, 118, 150))
        pen.setWidthF(1.0 / scale_factor)
        painter.setPen(pen)
        painter.drawLine(QPointF(center_x - cross_len, center_y), QPointF(center_x - gap, center_y))
        painter.drawLine(QPointF(center_x + gap, center_y), QPointF(center_x + cross_len, center_y))
        painter.drawLine(QPointF(center_x, center_y - cross_len), QPointF(center_x, center_y - gap))
        painter.drawLine(QPointF(center_x, center_y + gap), QPointF(center_x, center_y + cross_len))

        # 绘制文字
        font = QFont("Consolas")
        font.setPointSizeF(text_size * 0.8)
        painter.setFont(font)
        painter.setPen(QColor(0, 230, 118, 200))
        status_text = "SYSTEM: ONLINE  |  FOV: TARGET LOCKED"
        painter.drawText(QRectF(l, t + h - margin - text_size * 2, w, text_size * 2),
                         Qt.AlignmentFlag.AlignCenter, status_text)
        painter.restore()

    def emit_view_rect(self):
        if self.scene():
            view_rect = self.mapToScene(self.viewport().rect()).boundingRect()
            self.view_changed_signal.emit(view_rect)

    # 🟢 [核心修复] 传递 scene_size
    def set_image(self, img_cv, maintain_view=False):
        self.cv_img_ref = img_cv
        self.scene_obj.clear()
        self.highlight_item = None

        if img_cv is None:
            self.minimap.update_data(None)
            return

        h, w = img_cv.shape[:2]

        self.img_item = LazyGraphicsItem(img_cv)
        self.scene_obj.addItem(self.img_item)
        self.setSceneRect(0, 0, w, h)

        # 🟢 传入真实尺寸 (w, h) 修复小地图比例
        self.minimap.update_data(self.img_item.preview_pixmap, scene_size=(w, h))

        if not maintain_view:
            self.fitInView(self.scene_obj.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        self.emit_view_rect()

    def highlight_defect(self, x, y, size=30):
        if self.highlight_item:
            self.scene_obj.removeItem(self.highlight_item)

        pen = QPen(QColor("#2979ff"))
        pen.setWidth(2)
        rect = QRectF(x - size / 2, y - size / 2, size, size)
        self.highlight_item = self.scene_obj.addRect(rect, pen)
        self.centerOn(x, y)
        self.minimap.update()
        self.emit_view_rect()

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        factor = 1.25 if zoom_in else 1 / 1.25
        self.scale(factor, factor)
        self.viewport().update()
        self.minimap.update()
        self.emit_view_rect()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margin = 20
        if self.minimap.isVisible():
            mw, mh = self.minimap.width(), self.minimap.height()
            x = self.width() - mw - margin
            y = self.height() - mh - margin
            self.minimap.move(x, y)
        self.emit_view_rect()

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self.minimap.update()
        self.emit_view_rect()

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.emit_view_rect()

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        if self.cv_img_ref is not None:
            scene_pos = self.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            h, w = self.cv_img_ref.shape[:2]

            if 0 <= x < w and 0 <= y < h:
                if len(self.cv_img_ref.shape) == 2:
                    val = str(self.cv_img_ref[y, x])
                else:
                    val = str(self.cv_img_ref[y, x])
                self.mouse_moved_signal.emit(x, y, val)

    def pan_view(self, dx, dy):
        h_bar = self.horizontalScrollBar()
        v_bar = self.verticalScrollBar()
        h_bar.setValue(h_bar.value() + dx)
        v_bar.setValue(v_bar.value() + dy)
        self.emit_view_rect()
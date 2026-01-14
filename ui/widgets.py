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

# 1. 放入 DefectTableModel 类
# ==========================================
# 🚀 高性能数据模型 (替换 QTableWidget)
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
            # 根据列号返回对应的数据
            if col == 0: return item['ch']          # CH (int)
            if col == 1: return item['final_type']  # Type (str)
            if col == 2: return "White" if item.get('polarity') == 'Bright' else "Black"
            if col == 3: return item['gx']          # X (int)
            if col == 4: return item['gy']          # Y (int)
            if col == 5: return item['val']         # Val (int)
            if col == 6: return item.get('size', 1) # Size (int)

        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter

        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self._headers[section]
        return None

    def update_data(self, new_data):
        """核心：瞬间刷新数据"""
        self.beginResetModel()  # 通知视图：我要大换血了
        self._data = new_data
        self.endResetModel()    # 刷新完成

    pass

# 2. 放入 MiniMapOverlay 类
# ==========================================
# 🦅 组件升级：鹰眼小地图 (V22: 霓虹绿边框)
# ==========================================
class MiniMapOverlay(QWidget):
    def __init__(self, parent_view):
        super().__init__(parent_view)
        self.view = parent_view

        self.setFixedSize(240, 160)

        # [修改] 样式表：边框改为高亮绿色，背景加深
        self.setStyleSheet("""
            background-color: rgba(0, 0, 0, 255); /* 纯黑背景，防止图片干扰 */
            border: 2px solid #00e676;           /* 🟢 醒目的霓虹绿边框 */
            border-radius: 4px;
        """)
        self.setCursor(Qt.CursorShape.CrossCursor)

        self.preview_pixmap = None
        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0

        self.hide()

    def update_data(self, full_pixmap):
        if full_pixmap is None:
            self.hide()
            return

        self.show()

        # 计算内缩尺寸 (留出边框和padding)
        w_target = self.width() - 8
        h_target = self.height() - 8

        self.preview_pixmap = full_pixmap.scaled(
            w_target, h_target,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        if full_pixmap.width() > 0:
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

        # 2. 绘制视口框 (当前视野范围)
        if self.view.scene():
            viewport_rect = self.view.mapToScene(self.view.viewport().rect()).boundingRect()

            mx = viewport_rect.x() * self.scale_factor + self.offset_x
            my = viewport_rect.y() * self.scale_factor + self.offset_y
            mw = viewport_rect.width() * self.scale_factor
            mh = viewport_rect.height() * self.scale_factor

            # 绘制内部视野框 (白色细线，与外框绿色区分开)
            pen = QPen(QColor("#ffffff"))
            pen.setWidth(1)
            painter.setPen(pen)
            # 填充淡淡的白色，表示“我在这里”
            painter.setBrush(QColor(255, 255, 255, 30))
            painter.drawRect(QRectF(mx, my, mw, mh))
            # ==========================================================
            # 🟢 [新增] 绘制固定的绿色外边框
            # ==========================================================
            border_pen = QPen(QColor("#00e676"))  # 霓虹绿
            border_pen.setWidth(4)  # 边框宽度设为 4 像素，更醒目
            painter.setPen(border_pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)  # 内部不填充

            # 绘制整个控件范围的矩形框
            # adjusted(2, 2, -2, -2) 是为了让边框线完全显示在控件内部，不被切掉边缘
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
    pass

# 3. 放入 Surface3DViewer 类
# ==========================================
# ⛰️ 组件升级：3D 地形查看器 (带数字标尺版)
# ==========================================
class Surface3DViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("3D PIXEL INTENSITY SURFACE")
        self.resize(600, 600)
        self.setWindowFlags(Qt.WindowType.Window)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # 1. 创建 OpenGL 视图
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor('#111111')
        layout.addWidget(self.view)

        # 2. [保留] 添加简单的坐标轴线 (红绿蓝线)
        axis = gl.GLAxisItem()
        axis.setSize(x=55, y=55, z=85)  # 稍微长一点
        axis.translate(-25, -25, 0)  # 移动到原点
        self.view.addItem(axis)

        # 3. [新增] 添加数字标尺 (手动创建 TextItem)
        self.add_ruler_labels()

        # 4. 添加网格底板
        g = gl.GLGridItem()
        g.setSize(x=60, y=60, z=0)
        g.setSpacing(x=5, y=5, z=0)
        self.view.addItem(g)

        # 5. 创建表面绘图项 (初始数据)
        dummy_z = np.zeros((50, 50))
        self.p1 = gl.GLSurfacePlotItem(z=dummy_z, computeNormals=True, smooth=False, shader='shaded')
        self.p1.translate(-25, -25, 0)
        self.view.addItem(self.p1)

        # 设置视角
        self.view.setCameraPosition(distance=90, elevation=30, azimuth=45)

        # 6. 颜色映射
        pos = np.array([0.0, 0.33, 0.66, 1.0])
        color = np.array([
            [0, 0, 140, 255],  # 蓝
            [0, 255, 255, 255],  # 青
            [255, 255, 0, 255],  # 黄
            [255, 0, 0, 255]  # 红
        ], dtype=np.ubyte)
        self.colormap = pg.ColorMap(pos, color)

        # 底部提示
        lbl = QLabel("XYZ Scale: [X/Y] Pixel Offset (0-50) | [Z] Intensity Value (0-255)")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet("background: #000; color: #aaa; font-size: 9pt; padding: 4px; border-top: 1px solid #333;")
        layout.addWidget(lbl)

    def add_ruler_labels(self):
        """手动添加 X, Y, Z 轴的数字刻度"""

        # 辅助函数：添加文本
        def add_text(x, y, z, text):
            # GLTextItem 会自动始终朝向相机，非常适合做标签
            t = gl.GLTextItem(pos=(x, y, z), text=text, font=QFont('Arial', 8))
            self.view.addItem(t)

        # === X 轴刻度 (0 到 50) ===
        # 沿着 Y=-28 的边缘排列
        for i in range(0, 51, 10):
            # i 是 ROI 内的局部坐标
            # world_x 是世界坐标 (因为我们把地形平移了 -25)
            world_x = i - 25
            add_text(world_x, -28, 0, str(i))

        # === Y 轴刻度 (0 到 50) ===
        # 沿着 X=-28 的边缘排列
        for i in range(0, 51, 10):
            world_y = i - 25
            add_text(-28, world_y, 0, str(i))

        # === Z 轴刻度 (0 到 255) ===
        # 沿着角落 (-28, -28) 向上排列
        # 注意：显示高度需要除以 3 (因为我们渲染时 z_display = z / 3.0)
        for val in range(0, 256, 50):  # 每隔 50 显示一个刻度
            height = val / 3.0
            add_text(-28, -28, height, str(val))

    def update_surface(self, roi_data):
        """
        更新 3D 地形数据
        """
        if roi_data is None: return

        z = roi_data.astype(np.float32)
        h, w = z.shape

        # Z轴缩放系数 (必须与 add_ruler_labels 中的比例一致)
        z_display = z / 3.0

        # 颜色映射
        norm = z / 255.0
        colors = self.colormap.map(norm, mode='float')
        colors = colors.reshape(h * w, 4)

        self.p1.setData(z=z_display, colors=colors)

    pass

# 4. 放入 InteractiveHistogram 类
# ==========================================
# 📊 最终修复版：CMOS 专用交互式直方图
# ==========================================
class InteractiveHistogram(pg.PlotWidget):
    # 信号：当阈值线被拖动时，发送新的阈值 (0-255)
    threshold_changed_signal = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent, background='#1a1a1a')

        # --- 1. 界面初始化 ---
        self.setTitle("Pixel Intensity Distribution (All Channels)", color='#ccc', size='10pt')
        self.showGrid(x=True, y=True, alpha=0.3)
        # 允许 Y 轴缩放，锁定 X 轴
        self.setMouseEnabled(x=False, y=True)
        self.hideButtons()

        # 优化坐标轴显示
        left_axis = self.getPlotItem().getAxis('left')
        left_axis.setWidth(45)  # 增加宽度防止数字被遮挡
        self.setLabel('bottom', 'DN Value (8-bit Mapped)', units='')
        self.setLabel('left', 'Pixel Count', units='')

        # --- 2. 核心绘图元素 ---
        # stepMode=True 确保柱状图对齐刻度，fillLevel=0 填充底部
        self.curve = self.plot(stepMode=True, fillLevel=0,
                               brush=pg.mkBrush(0, 230, 118, 100),  # 半透明绿色填充
                               pen=pg.mkPen('#00e676', width=1))

        # --- 3. 交互式阈值线 ---
        # 使用 Qt.PenStyle.DashLine 修复兼容性报错
        self.thresh_line = pg.InfiniteLine(pos=50, angle=90, movable=True,
                                           pen=pg.mkPen('#ff1744', width=2, style=Qt.PenStyle.DashLine),
                                           hoverPen=pg.mkPen('#ff1744', width=4))

        # 绑定拖拽结束事件
        self.thresh_line.sigPositionChangeFinished.connect(self.on_line_dragged)
        self.addItem(self.thresh_line)

        # --- 4. 鼠标悬停十字光标 ---
        # 使用 'y' (黄色) 修复颜色报错，使用 Qt.PenStyle.DotLine 修复样式报错
        self.v_line = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('y', width=1, style=Qt.PenStyle.DotLine))
        self.h_line = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen('y', width=1, style=Qt.PenStyle.DotLine))
        self.addItem(self.v_line)
        self.addItem(self.h_line)

        # 监听鼠标移动
        self.scene().sigMouseMoved.connect(self.on_mouse_move)

        # 初始化视图范围 (0-256)
        self.setXRange(0, 256, padding=0)
        self.setYRange(0, 100)

        # 缓存当前数据
        self.current_hist = None

    def update_data(self, img):
        """
        核心计算函数：
        1. 拍平多通道数据 (避免 RGB 平均化导致数值变小)
        2. 智能处理 16-bit 数据映射 (还原 15, 16 预期值)
        3. 使用 bincount 精确统计
        """
        if img is None: return

        # ====================================================
        # 步骤 1: 数据拍平 (Flatten)
        # 这一步修复了 "不区分通道是错的" 的问题
        # 如果图片是 (H, W, 3)，变成 (H*W*3,)，把 R,G,B 拆开独立统计
        # ====================================================
        raw_data = img.flatten()
        # 2. 🟢 [优化] 位深映射 (使用位运算代替除法)
        if raw_data.dtype == np.uint16:
            # 16-bit 降 8-bit：右移 8 位
            data_to_plot = (raw_data >> 8).astype(np.uint8)
        else:
            data_to_plot = raw_data.astype(np.uint8)

        # ====================================================
        # 步骤 3: 精确统计 (Bincount)
        # ====================================================
        # minlength=256 保证即使最大值只有 20，数组长度也是 256
        hist = np.bincount(data_to_plot, minlength=256)

        # 截取前 256 个 (防止异常大值导致数组越界)
        if len(hist) > 256:
            hist = hist[:256]

        self.current_hist = hist

        # ====================================================
        # 步骤 4: 智能 Y 轴缩放
        # ====================================================
        if len(self.current_hist) > 1:
            # 避开下标 0 (背景黑底)，否则真正的信号会被压扁
            valid_data = self.current_hist[1:]
            if len(valid_data) > 0:
                peak_val = np.max(valid_data)
                self.setYRange(0, float(peak_val) * 1.2)

        # ====================================================
        # 步骤 5: 更新绘图
        # ====================================================
        # stepMode=True 需要 x 比 y 多一个点
        x = np.arange(257)
        self.curve.setData(x, self.current_hist)

    def on_mouse_move(self, pos):
        """鼠标移动时更新十字线和标题读数"""
        if self.sceneBoundingRect().contains(pos):
            mouse_point = self.getPlotItem().vb.mapSceneToView(pos)
            x_val = mouse_point.x()

            if 0 <= x_val <= 255 and self.current_hist is not None:
                idx = int(x_val)
                # 防止数组越界
                if idx < len(self.current_hist):
                    y_val = self.current_hist[idx]

                    # 更新线条位置
                    self.v_line.setPos(x_val)
                    self.h_line.setPos(y_val)

                    # 实时更新标题显示数值
                    self.setTitle(
                        f"<span style='color: #ccc'>DN Value: {idx}</span>  |  "
                        f"<span style='color: #00e676'>Count: {int(y_val)}</span>",
                        size='10pt'
                    )

    def on_line_dragged(self):
        """线拖动结束，发送信号"""
        val = int(self.thresh_line.value())
        val = max(0, min(255, val))
        self.threshold_changed_signal.emit(val)

    def set_line_pos(self, val):
        """外部修改 SpinBox 时同步更新线的位置"""
        self.thresh_line.setValue(val)

    pass

# 5. 放入 LazyGraphicsItem 类
class LazyGraphicsItem(QGraphicsItem):
    """
    智能懒加载图元：
    1. 缩小时显示低清预览图 (Preview)，保持流畅。
    2. 放大时动态切片渲染高清原图 (Raw)，保证坏点清晰可见。
    """

    def __init__(self, cv_img):
        super().__init__()
        self.cv_img = cv_img  # 持有原图引用 (Numpy array)
        self.h, self.w = cv_img.shape[:2]

        # 1. 生成低分辨率预览图 (限制长边 2000 像素)
        # 这张图用于由远及近的过渡，以及缩小时的显示
        max_dim = 2000
        scale = min(1.0, max_dim / max(self.h, self.w))
        if scale < 1.0:
            # INTER_AREA 对缩小图像保留特征较好，虽然还是会丢单像素，
            # 但我们在放大时会切换回原图
            preview_img = cv2.resize(cv_img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        else:
            preview_img = cv_img

        self.preview_pixmap = self._cv2_to_qpixmap(preview_img)
        self.rect = QRectF(0, 0, self.w, self.h)

    def boundingRect(self):
        return self.rect

    def paint(self, painter, option, widget):
        # 1. 始终绘制预览图作为底色 (填满整个区域)
        # 这样在快速拖动尚未加载高清图时，不会出现白屏
        painter.drawPixmap(self.rect, self.preview_pixmap, QRectF(self.preview_pixmap.rect()))

        # 2. 计算细节层次 (LOD - Level of Detail)
        # transform.m11() 近似代表水平缩放比例
        # 如果缩放比例很小 (比如看全图)，只画预览图，节省性能
        lod = option.levelOfDetailFromTransform(painter.worldTransform())

        # 阈值可调：当 1 个屏幕像素对应图片上少于 2 个像素时，开始渲染高清
        # 也就是说，当你稍微放大一点，它就会切高清图
        if lod > 0.5:
            # 3. 计算当前屏幕可见的区域 (Exposed Rect)
            exposed = option.exposedRect

            # 转换为整数坐标并做边界安全检查
            x = max(0, int(exposed.x()))
            y = max(0, int(exposed.y()))
            rw = int(exposed.width()) + 2  # 多切一点防止边缘缝隙
            rh = int(exposed.height()) + 2

            # 再次修正边界
            if x + rw > self.w: rw = self.w - x
            if y + rh > self.h: rh = self.h - y

            if rw > 0 and rh > 0:
                # 4. 【核心】实时切片 (Slicing)
                # 这一步非常快，因为只是内存视图操作
                sub_img = self.cv_img[y:y + rh, x:x + rw]

                # 5. 局部转码并绘制
                # 只转换屏幕上看到的那一小块，不会爆内存
                h_sub, w_sub = sub_img.shape[:2]

                # 转换 QImage
                if len(sub_img.shape) == 2:
                    qimg = QImage(sub_img.data, w_sub, h_sub, w_sub, QImage.Format.Format_Grayscale8)
                else:
                    # 注意：如果这里卡顿，可以考虑把原图存为 RGB 格式，省去转换
                    # 但通常局部转换极快 (ms级)
                    sub_rgb = cv2.cvtColor(sub_img, cv2.COLOR_BGR2RGB)
                    qimg = QImage(sub_rgb.data, w_sub, h_sub, w_sub * 3, QImage.Format.Format_RGB888)

                # 绘制高清切片到指定位置
                painter.drawImage(QRectF(x, y, w_sub, h_sub), qimg)

    def _cv2_to_qpixmap(self, img):
        h, w = img.shape[:2]
        if len(img.shape) == 2:
            qimg = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        else:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            qimg = QImage(img_rgb.data, w, h, w * 3, QImage.Format.Format_RGB888)
        return QPixmap.fromImage(qimg)
    pass

# 6. 放入 ZoomableGraphicsView 类
class ZoomableGraphicsView(QGraphicsView):
    mouse_moved_signal = pyqtSignal(int, int, str)
    # 🟢 [补回] 1. 定义视野变化信号 (用于雷达框联动)
    view_changed_signal = pyqtSignal(QRectF)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setBackgroundBrush(QColor("#111"))

        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)

        # 🟢 [关键修复] 2. 开启鼠标追踪，否则不按键时拿不到坐标/数值
        self.setMouseTracking(True)

        self.scene_obj = QGraphicsScene(self)
        self.setScene(self.scene_obj)

        self.img_item = None
        self.cv_img_ref = None
        self.highlight_item = None
        self.minimap = MiniMapOverlay(self)

    # 🟢 [补回] 3. 发送视野信号的辅助函数
    def emit_view_rect(self):
        if self.scene():
            # 获取当前视口在场景中的矩形范围
            view_rect = self.mapToScene(self.viewport().rect()).boundingRect()
            self.view_changed_signal.emit(view_rect)

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

        self.minimap.update_data(self.img_item.preview_pixmap)

        if not maintain_view:
            self.fitInView(self.scene_obj.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

        # 🟢 [补回] 视图改变后发送信号
        self.emit_view_rect()

    def highlight_defect(self, x, y, size=30):
        if self.highlight_item:
            self.scene_obj.removeItem(self.highlight_item)

        pen = QPen(Qt.GlobalColor.cyan)
        pen.setWidth(2)
        rect = QRectF(x - size / 2, y - size / 2, size, size)
        self.highlight_item = self.scene_obj.addRect(rect, pen)
        self.centerOn(x, y)
        self.minimap.update()
        self.emit_view_rect()  # 🟢

    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        factor = 1.25 if zoom_in else 1 / 1.25
        self.scale(factor, factor)
        self.viewport().update()
        self.minimap.update()
        self.emit_view_rect()  # 🟢

    def resizeEvent(self, event):
        super().resizeEvent(event)
        margin = 20
        if self.minimap.isVisible():
            mw, mh = self.minimap.width(), self.minimap.height()
            x = self.width() - mw - margin
            y = self.height() - mh - margin
            self.minimap.move(x, y)
        self.emit_view_rect()  # 🟢

    def scrollContentsBy(self, dx, dy):
        super().scrollContentsBy(dx, dy)
        self.minimap.update()
        self.emit_view_rect()  # 🟢

    def mouseReleaseEvent(self, event):
        super().mouseReleaseEvent(event)
        self.emit_view_rect()  # 🟢

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        # 🟢 这里的逻辑现在因为 setMouseTracking(True) 而能实时触发了
        if self.cv_img_ref is not None:
            scene_pos = self.mapToScene(event.pos())
            x, y = int(scene_pos.x()), int(scene_pos.y())
            h, w = self.cv_img_ref.shape[:2]

            if 0 <= x < w and 0 <= y < h:
                # 简单读取数值
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
        self.emit_view_rect()  # 🟢

    pass
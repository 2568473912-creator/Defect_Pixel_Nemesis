import os
import csv
import cv2
import xlsxwriter
import gc
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback  # 🟢 [新增]
from core.algorithm import CoreAlgorithm
from core.tasks import process_single_image_task
from utils.helpers import get_safe_roi
from utils.helpers import get_safe_roi, FileHandler  # 🟢 [修改] 导入 FileHandler
from utils.logger import log  # 🟢 [新增] 导入日志

# 1. 放入 SingleWorker 类
# ==========================================
# 🧵 2.1 单图分析线程 (修改版：返回双图)
# ==========================================
class SingleWorker(QThread):
    # [修改] 信号定义：改为发送 3 个对象 (原图Vis, 网格图Grid, 数据Data)
    # 使用 object 类型以兼容 numpy 数组和列表
    result_signal = pyqtSignal(object, object, object, object)
    error_occurred = pyqtSignal(str)  # 🟢 [新增] 错误信号

    def __init__(self, path, params):
        super().__init__()
        self.path = path
        self.params = params

    def run(self):
        try:
            log.info(f"Worker started for: {self.path}")

            # 🟢 [修改] 使用 FileHandler 安全读取
            # 注意：get_params() 需要返回 bit_depth 信息，如果没有，这里默认传 8 或 16
            # 假设 self.params 包含 'w', 'h', 'ch'。如果是普通图片，后两个参数可以忽略或自动推断
            # 如果是 RAW 模式，必须从 UI 传入 bit_depth。
            # 这里为了兼容您的原始代码(cv2.imread)，我们尝试用 FileHandler
            # 如果 params 里没有宽高，FileHandler 对于 RAW 会报错。
            # 兼容逻辑：先尝试用 params 参数，如果缺少，说明是普通图片流程

            # 暂时假设是普通图片或参数已齐备。
            # 如果原代码只用 cv2.imread，说明之前只处理普通图片。
            # 这里我们为了健壮性，对于普通图片仍然可以用 FileHandler (它内部封装了 imdecode)

            # 简单起见，如果 load_image_file 需要参数而 params 里没有，可以给默认值
            width = self.params.get('w', 0)
            height = self.params.get('h', 0)
            channels = self.params.get('ch', 1)  # 默认单通道避免报错
            bit_depth = 8  # 默认 8位

            img_raw = FileHandler.load_image_file(self.path, width, height, channels, bit_depth)

            if img_raw is None:
                raise ValueError("Failed to load image. Check file format or path.")

            # 1. 执行核心检测算法
            log.debug("Running CoreAlgorithm...")
            vis_raw, data = CoreAlgorithm.run_dispatch(img_raw, self.params)

            # 2. 生成通道网格图
            vis_grid = None  # 根据原代码逻辑

            # 3. 发送结果
            log.info(f"Analysis complete. Defects found: {len(data)}")
            self.result_signal.emit(vis_raw, vis_grid, data, img_raw)

        except Exception as e:
            # 🟢 [新增] 捕获所有异常
            err_msg = traceback.format_exc()
            log.error(f"SingleWorker Crashed:\n{err_msg}")
            self.error_occurred.emit(f"Analysis Error:\n{str(e)}")

    pass

# 2. 放入 BatchWorker 类
# ==========================================
# 🧵 2.2 批量处理线程 (修复版：变量定义完整)
# ==========================================
# ==========================================
# 🧵 批量处理线程 (多进程并行版 - 修复版)
# ==========================================
class BatchWorker(QThread):
    progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, in_dir, out_dir, filter_str, params, specs, snap_params=(5, 60)):
        super().__init__()
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.filter_str = filter_str
        self.params = params
        self.specs = specs
        self.snap_params = snap_params
        self.is_running = True
        self.export_details = True

    def run(self):
        try:
            log.info("Batch Processing Started")  # 🟢 [新增]
            self.out_dir.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            log.error(f"Output Dir Error: {e}")  # 🟢 [新增]
            self.log_signal.emit(f"❌ Output Dir Error: {e}")
            self.finished_signal.emit();
            return

        files = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        if self.in_dir.exists():
            for f in self.in_dir.iterdir():
                if f.is_file() and f.suffix.lower() in valid_extensions:
                    if self.filter_str and self.filter_str.lower() not in f.name.lower(): continue
                    files.append(f)

        total_files = len(files)
        if total_files == 0:
            self.log_signal.emit("❌ No matching images found.")
            self.finished_signal.emit();
            return

        self.log_signal.emit(f"🚀 Found {total_files} files. Starting Multiprocessing...")

        summary_data = []
        all_cluster_details = []

        # 自动控制并发数
        max_workers = min(os.cpu_count(), 16)

        try:  # 🟢 [新增] 大循环的 try-except
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_image_task,
                        f, self.out_dir, self.params, self.specs, self.snap_params, self.export_details
                    ): f for f in files
                }

                completed_count = 0
                for future in as_completed(futures):
                    if not self.is_running:
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    try:
                        res = future.result()
                        # ... (结果处理逻辑保持不变) ...
                        if res['status'] == 'error':
                            self.log_signal.emit(f"❌ {res['msg']}")
                            log.warning(f"Task failed: {res['msg']}")  # 🟢
                        else:
                            summary_data.append(res['summary_row'])
                            if 'cluster_details' in res:
                                all_cluster_details.extend(res['cluster_details'])
                            self._save_single_csv(res['file_stem'], res['data'])
                            log_icon = "🟢" if res['result_str'] == "PASS" else "🔴"
                            self.log_signal.emit(f"{log_icon} {res['filename']} -> {res['result_str']}")

                    except Exception as e:
                        log.error(f"Future Result Error: {e}", exc_info=True)  # 🟢
                        self.log_signal.emit(f"⚠️ Process Error: {e}")

                    completed_count += 1
                    self.progress_signal.emit(completed_count, total_files)

            if self.is_running:
                self._save_summary_excel(summary_data, all_cluster_details)

        except Exception as e:
            log.critical(f"BatchWorker Critical Error: {e}", exc_info=True)  # 🟢
            self.log_signal.emit(f"🔥 Critical Error: {e}")

        self.finished_signal.emit()

    def _save_single_csv(self, file_stem, data):
        csv_path = self.out_dir / f"{file_stem}_detail_report.csv"
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as cf:
                writer = csv.writer(cf)
                # [修改] 增加 Cluster ID 列
                writer.writerow(['CH', 'Cluster ID', 'Type', 'Polarity', 'X', 'Y', 'Val', 'Size', 'CropPath'])
                for d in data:
                    pol = "White" if d.get('polarity') == 'Bright' else "Black"
                    writer.writerow([
                        d['ch'],
                        d.get('cluster_id', 0),  # 写入 ID
                        d.get('final_type'), pol,
                        d['gx'], d['gy'], d['val'],
                        d.get('size', 1), d.get('CropPath', '')
                    ])
        except Exception as e:
            log.error(f"CSV Error: {e}")

    def _save_summary_excel(self, summary_data, all_cluster_details):
        if not summary_data: return
        try:
            excel_path = str(self.out_dir / "Batch_Report.xlsx")
            workbook = xlsxwriter.Workbook(excel_path)

            # 样式定义
            header_fmt = workbook.add_format(
                {'bold': True, 'bg_color': '#333333', 'font_color': 'white', 'border': 1, 'align': 'center',
                 'valign': 'vcenter'})
            pass_fmt = workbook.add_format(
                {'bg_color': '#C6EFCE', 'font_color': '#006100', 'align': 'center', 'border': 1, 'valign': 'vcenter'})
            fail_fmt = workbook.add_format(
                {'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'align': 'center', 'border': 1, 'valign': 'vcenter'})
            norm_fmt = workbook.add_format({'align': 'center', 'border': 1, 'valign': 'vcenter'})

            # Sheet 1: Summary
            ws1 = workbook.add_worksheet("Summary")
            headers1 = ["Filename", "Result", "Total Pixels", "Total Clusters", "White Pixels", "Black Pixels",
                        "White Clusters", "Black Clusters"]
            ws1.write_row(0, 0, headers1, header_fmt)

            summary_data.sort(key=lambda x: x['Filename'])
            for r, item in enumerate(summary_data, start=1):
                res = item["Result"]
                fmt = pass_fmt if res == "PASS" else fail_fmt
                ws1.write(r, 0, item["Filename"], norm_fmt)
                ws1.write(r, 1, res, fmt)
                ws1.write(r, 2, item["Total_Points"], norm_fmt)
                ws1.write(r, 3, item["Total_Clusters"], norm_fmt)
                ws1.write(r, 4, item["White_Points"], norm_fmt)
                ws1.write(r, 5, item["Black_Points"], norm_fmt)
                ws1.write(r, 6, item["White_Clusters"], norm_fmt)
                ws1.write(r, 7, item["Black_Clusters"], norm_fmt)
            ws1.set_column(0, 0, 30)

            # Sheet 2: Cluster Details
            if self.export_details and all_cluster_details:
                ws2 = workbook.add_worksheet("Cluster_Details")
                # [修改] 增加 Cluster ID 列
                headers2 = ["Filename", "Cluster ID", "CH", "Type", "Polarity", "X", "Y", "Val", "Size", "Snapshot"]
                ws2.write_row(0, 0, headers2, header_fmt)
                # 🟢 [新增] Excel 行数上限
                MAX_ROWS = 1048500
                all_cluster_details.sort(key=lambda x: x['Filename'])

                for r, d in enumerate(all_cluster_details, start=1):
                    # 🟢 [新增] 超限检查
                    if r > MAX_ROWS:
                        ws2.write(r, 0, "⚠️ TRUNCATED", norm_fmt)
                        self.log_signal.emit(f"⚠️ Batch Report Truncated: Exceeded {MAX_ROWS} rows.")
                        break
                    ws2.set_row(r, 65)
                    ws2.write(r, 0, d["Filename"], norm_fmt)
                    ws2.write(r, 1, d.get("Cluster ID", 0), norm_fmt)  # 写入 ID
                    ws2.write(r, 2, d["CH"], norm_fmt)
                    ws2.write(r, 3, d["Type"], norm_fmt)
                    ws2.write(r, 4, d["Polarity"], norm_fmt)
                    ws2.write(r, 5, d["X"], norm_fmt)
                    ws2.write(r, 6, d["Y"], norm_fmt)
                    ws2.write(r, 7, d["Val"], norm_fmt)
                    ws2.write(r, 8, d["Size"], norm_fmt)

                    # 插入图片 (检查路径是否存在)
                    # 因为我们刚才在 tasks.py 做了控制，同一个 Cluster 只有第一行会有 CropPath
                    if d.get("CropPath") and os.path.exists(d["CropPath"]):
                        ws2.insert_image(r, 9, d["CropPath"], {'x_offset': 5, 'y_offset': 2})

                ws2.set_column(0, 0, 25)
                ws2.set_column(9, 9, 12)  # 调整最后一列宽

            workbook.close()
            self.log_signal.emit(f"✅ Excel Saved: {excel_path}")

        except Exception as e:
            log.error(f"Excel Error: {e}")  # 🟢
            self.log_signal.emit(f"⚠️ Excel Error: {e}")

    def stop(self):
        self.is_running = False

    pass

# 3. 放入 BatchCropWorker 类
# ==========================================
# ✂️ 批量截图线程 (矩阵 Excel 版)
# ==========================================
class BatchCropWorker(QThread):
    progress_signal = pyqtSignal(int, int)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, in_dir, out_dir, filter_str, mode_config):
        super().__init__()
        self.in_dir = Path(in_dir)
        self.out_dir = Path(out_dir)
        self.filter_str = filter_str
        self.config = mode_config
        self.is_running = True

        # 🟢 [新增] 用于存储 Excel 矩阵数据
        # 结构: {(coord_index, file_index): "image_path_string"}
        self.matrix_data = {}
        self.processed_files = []  # 记录处理了哪些文件 (作为 Excel 表头)
        self.coords_record = []  # 记录用到的坐标 (作为 Excel 前两列)

    def run(self):
        # 1. 检查输出目录
        try:
            log.info("Batch Crop Processing Started")  # 🟢 [新增]
            self.out_dir.mkdir(parents=True, exist_ok=True)
            # 创建一个专门放截图的子目录，保持整洁
            self.crop_save_dir = self.out_dir / "matrix_crops"
            self.crop_save_dir.mkdir(exist_ok=True)
        except Exception as e:
            log.error(f"Crop Output Dir Error: {e}")  # 🟢 [新增]
            self.log_signal.emit(f"❌ Output Error: {e}")
            self.finished_signal.emit()
            return

        # 2. 扫描文件
        files = []
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        if self.in_dir.exists():
            for f in self.in_dir.iterdir():
                if f.is_file() and f.suffix.lower() in valid_ext:
                    if self.filter_str and self.filter_str.lower() not in f.name.lower():
                        continue
                    files.append(f)

        # 按文件名排序，保证 Excel 列顺序一致
        files.sort(key=lambda x: x.name)

        total = len(files)
        if total == 0:
            self.log_signal.emit("⚠️ No matching images found.")
            self.finished_signal.emit()
            return

        self.log_signal.emit(f"🚀 Found {total} files. Processing...")

        # 3. 准备坐标列表 (为了统一逻辑，Single模式也视为只有1个坐标的列表)
        if self.config['mode'] == 'coords':
            self.coords_record = self.config['coord_list']
            # 解包参数
            radius, out_size = self.config['snap_params']
            resize_target = (out_size, out_size)
            resize_enabled = True  # Coords 模式强制 Resize
        else:
            # Single 模式：只有一个坐标 (Center X, Center Y)
            # 注意：config['rect'] 已经是 TopLeft 了，我们需要反算回 Center 存入 Excel，或者直接存 TL
            # 这里为了 Excel 好看，我们重新算一下 Center
            rx, ry, rw, rh = self.config['rect']
            cx = rx + rw // 2
            cy = ry + rh // 2
            self.coords_record = [(cx, cy)]

            radius = None  # Single模式不用radius
            resize_enabled = self.config.get('resize_enabled', False)
            resize_target = self.config.get('resize_target', (60, 60))

        # 4. 循环处理每一张大图
        for file_idx, f in enumerate(files):
            if not self.is_running: break

            try:
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is None: continue

                self.processed_files.append(f.name)  # 记录文件名用于表头

                # 遍历所有坐标点
                for coord_idx, (cx, cy) in enumerate(self.coords_record):

                    # --- 计算裁剪区域 ---
                    if self.config['mode'] == 'coords':
                        # Coords 模式：基于 Radius
                        half = radius
                        x_tl = cx - half
                        y_tl = cy - half
                        w_box = h_box = half * 2
                    else:
                        # Single 模式：基于 Rect
                        x_tl, y_tl, w_box, h_box = self.config['rect']

                    # 安全裁剪
                    fx, fy, fw, fh = get_safe_roi(img.shape, x_tl, y_tl, w_box, h_box)
                    if fw <= 0 or fh <= 0: continue

                    crop = img[fy:fy + fh, fx:fx + fw]

                    if crop.size > 0:
                        # Resize 处理
                        if resize_enabled:
                            crop = cv2.resize(crop, resize_target, interpolation=cv2.INTER_NEAREST)

                        # 保存文件
                        # 命名格式: fileIdx_coordIdx.png (简单短小，避免路径过长)
                        save_name = f"F{file_idx}_C{coord_idx}.png"
                        full_path = self.crop_save_dir / save_name
                        cv2.imwrite(str(full_path), crop)

                        # 🟢 [关键] 记录路径到矩阵字典
                        self.matrix_data[(coord_idx, file_idx)] = str(full_path)

                if file_idx % 5 == 0:
                    self.log_signal.emit(f"✅ Processed: {f.name}")
                self.progress_signal.emit(file_idx + 1, total)

            except Exception as e:
                log.error(f"crop Error {f.name}: {e}")
                self.log_signal.emit(f"❌ Error {f.name}: {e}")

        # 5. 🟢 生成 Excel 矩阵
        if self.is_running and self.matrix_data:
            self.log_signal.emit("📊 Generating Matrix Excel...")
            self.generate_matrix_excel(resize_target[1])  # 传入高度用于设置行高

        self.finished_signal.emit()

    def generate_matrix_excel(self, img_height):
        excel_path = self.out_dir / "Comparison_Matrix.xlsx"
        try:
            wb = xlsxwriter.Workbook(str(excel_path))
            ws = wb.add_worksheet("Matrix")

            # 样式
            fmt_header = wb.add_format(
                {'bold': True, 'bg_color': '#333', 'font_color': 'white', 'border': 1, 'align': 'center',
                 'valign': 'vcenter'})
            fmt_coord = wb.add_format(
                {'bold': True, 'bg_color': '#eee', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
            fmt_norm = wb.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})

            # --- 1. 写表头 (Row 0) ---
            # Col 0, 1: 坐标
            ws.write(0, 0, "Center X", fmt_header)
            ws.write(0, 1, "Center Y", fmt_header)

            # Col 2...N: 图片文件名
            for col_idx, fname in enumerate(self.processed_files):
                ws.write(0, col_idx + 2, fname, fmt_header)
                # 设置列宽，稍微比图片宽一点 (假设图片宽=img_height，这里粗略估算)
                ws.set_column(col_idx + 2, col_idx + 2, img_height / 6)

                # --- 2. 写数据行 (Row 1...M) ---
            for row_idx, (cx, cy) in enumerate(self.coords_record):
                excel_row = row_idx + 1

                # 设置行高 (比图片略高)
                ws.set_row(excel_row, img_height + 5)

                # 写坐标
                ws.write(excel_row, 0, cx, fmt_coord)
                ws.write(excel_row, 1, cy, fmt_coord)

                # 写图片
                for col_idx in range(len(self.processed_files)):
                    # 检查是否有图
                    key = (row_idx, col_idx)
                    if key in self.matrix_data:
                        img_path = self.matrix_data[key]
                        # 插入图片
                        # x_offset, y_offset 让图片居中一点
                        ws.insert_image(excel_row, col_idx + 2, img_path,
                                        {'x_offset': 5, 'y_offset': 2, 'object_position': 1})
                    else:
                        ws.write(excel_row, col_idx + 2, "N/A", fmt_norm)

            wb.close()
            self.log_signal.emit(f"🏆 Excel Saved: {excel_path}")

        except Exception as e:
            log.error(f"Crop Excel Error: {e}")  # 🟢
            self.log_signal.emit(f"⚠️ Excel Error: {e}")

    def stop(self):
        self.is_running = False
    pass
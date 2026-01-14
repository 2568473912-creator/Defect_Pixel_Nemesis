import os
import csv
import cv2
import xlsxwriter
import gc
import traceback
from pathlib import Path
from PyQt6.QtCore import QThread, pyqtSignal
# 🟢 [修改 1] 导入 ThreadPoolExecutor (多线程) 代替 ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.algorithm import CoreAlgorithm
from core.tasks import process_single_image_task
from utils.helpers import get_safe_roi, FileHandler
from utils.logger import log


# ==========================================
# 🧵 2.1 单图分析线程
# ==========================================
class SingleWorker(QThread):
    result_signal = pyqtSignal(object, object, object, object)
    error_occurred = pyqtSignal(str)

    def __init__(self, path, params):
        super().__init__()
        self.path = path
        self.params = params

    def run(self):
        try:
            log.info(f"Worker started for: {self.path}")

            width = self.params.get('w', 0)
            height = self.params.get('h', 0)
            channels = self.params.get('ch', 1)
            bit_depth = 8

            img_raw = FileHandler.load_image_file(self.path, width, height, channels, bit_depth)

            if img_raw is None:
                raise ValueError("Failed to load image. Check file format or path.")

            # 1. 执行核心检测算法
            log.debug("Running CoreAlgorithm...")
            vis_raw, data = CoreAlgorithm.run_dispatch(img_raw, self.params)

            # 2. 生成通道网格图
            vis_grid = None

            # 3. 发送结果
            log.info(f"Analysis complete. Defects found: {len(data)}")
            self.result_signal.emit(vis_raw, vis_grid, data, img_raw)

        except Exception as e:
            err_msg = traceback.format_exc()
            log.error(f"SingleWorker Crashed:\n{err_msg}")
            self.error_occurred.emit(f"Analysis Error:\n{str(e)}")


# ==========================================
# 🧵 2.2 批量处理线程 (多线程版 - 修复SharedMemory错误)
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
            log.info("Batch Processing Started (Threading Mode)")  # 🟢 [日志更新]
            self.out_dir.mkdir(exist_ok=True, parents=True)
        except Exception as e:
            log.error(f"Output Dir Error: {e}")
            self.log_signal.emit(f"❌ Output Dir Error: {e}")
            self.finished_signal.emit()
            return

        files = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.raw', '.bin'}
        if self.in_dir.exists():
            for f in self.in_dir.iterdir():
                if f.is_file() and f.suffix.lower() in valid_extensions:
                    if self.filter_str and self.filter_str.lower() not in f.name.lower(): continue
                    files.append(f)

        total_files = len(files)
        if total_files == 0:
            self.log_signal.emit("❌ No matching images found.")
            self.finished_signal.emit()
            return

        self.log_signal.emit(f"🚀 Found {total_files} files. Starting ThreadPool...")

        summary_data = []
        all_cluster_details = []

        # 🟢 [修改 2] 线程数设置
        # 对于 IO 密集型或释放 GIL 的任务，线程数可以设大一点 (例如 CPU核心数 * 2)
        max_workers = min(32, (os.cpu_count() or 4) * 2)

        try:
            # 🟢 [修改 3] 使用 ThreadPoolExecutor
            # 这不需要跨进程复制内存，彻底消除 SharedMemory read failed 错误
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_image_task,
                        f, self.out_dir, self.params, self.specs, self.snap_params, self.export_details
                    ): f for f in files
                }

                completed_count = 0
                for future in as_completed(futures):
                    if not self.is_running:
                        # 线程池无法强制杀死运行中的线程，只能不再调度新任务
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    try:
                        res = future.result()
                        if res['status'] == 'error':
                            self.log_signal.emit(f"❌ {res['msg']}")
                            log.warning(f"Task failed: {res['msg']}")
                        else:
                            summary_data.append(res['summary_row'])
                            if 'cluster_details' in res:
                                all_cluster_details.extend(res['cluster_details'])
                            self._save_single_csv(res['file_stem'], res['data'])
                            log_icon = "🟢" if res['result_str'] == "PASS" else "🔴"
                            self.log_signal.emit(f"{log_icon} {res['filename']} -> {res['result_str']}")

                    except Exception as e:
                        log.error(f"Future Result Error: {e}", exc_info=True)
                        self.log_signal.emit(f"⚠️ Process Error: {e}")

                    completed_count += 1
                    self.progress_signal.emit(completed_count, total_files)

            if self.is_running:
                self._save_summary_excel(summary_data, all_cluster_details)

        except Exception as e:
            log.critical(f"BatchWorker Critical Error: {e}", exc_info=True)
            self.log_signal.emit(f"🔥 Critical Error: {e}")

        self.finished_signal.emit()

    def _save_single_csv(self, file_stem, data):
        csv_path = self.out_dir / f"{file_stem}_detail_report.csv"
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as cf:
                writer = csv.writer(cf)
                writer.writerow(['CH', 'Cluster ID', 'Type', 'Polarity', 'X', 'Y', 'Val', 'Size', 'CropPath'])
                for d in data:
                    # 兼容对象属性和字典访问
                    try:
                        # 如果 d 是 DefectPoint 对象
                        ch = getattr(d, 'ch', d['ch'])
                        # 优先取 cluster_id, 如果没有则取 0
                        cid = getattr(d, 'cluster_id', d.get('cluster_id', 0))
                        ftype = getattr(d, 'final_type', d.get('final_type'))

                        pol_raw = getattr(d, 'polarity', d.get('polarity'))
                        pol = "White" if pol_raw == 'Bright' else "Black"

                        gx = getattr(d, 'gx', d['gx'])
                        gy = getattr(d, 'gy', d['gy'])
                        val = getattr(d, 'val', d['val'])
                        size = getattr(d, 'size', d.get('size', 1))
                        crop = getattr(d, 'CropPath', d.get('CropPath', ''))
                    except Exception:
                        # 兜底
                        ch, cid, ftype, pol, gx, gy, val, size, crop = 0, 0, "ERR", "ERR", 0, 0, 0, 0, ""

                    writer.writerow([ch, cid, ftype, pol, gx, gy, val, size, crop])
        except Exception as e:
            log.error(f"CSV Error: {e}")

    def _save_summary_excel(self, summary_data, all_cluster_details):
        if not summary_data: return
        try:
            excel_path = str(self.out_dir / "Batch_Report.xlsx")
            workbook = xlsxwriter.Workbook(excel_path)

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
                headers2 = ["Filename", "Cluster ID", "CH", "Type", "Polarity", "X", "Y", "Val", "Size", "Snapshot"]
                ws2.write_row(0, 0, headers2, header_fmt)
                MAX_ROWS = 1048500
                all_cluster_details.sort(key=lambda x: x['Filename'])

                for r, d in enumerate(all_cluster_details, start=1):
                    if r > MAX_ROWS:
                        ws2.write(r, 0, "⚠️ TRUNCATED", norm_fmt)
                        break
                    ws2.set_row(r, 65)
                    ws2.write(r, 0, d["Filename"], norm_fmt)
                    # 兼容对象或字典
                    cid = d.get("Cluster ID", d.get("cluster_id", 0))
                    ws2.write(r, 1, cid, norm_fmt)
                    ws2.write(r, 2, d["CH"], norm_fmt)
                    ws2.write(r, 3, d["Type"], norm_fmt)
                    ws2.write(r, 4, d["Polarity"], norm_fmt)
                    ws2.write(r, 5, d["X"], norm_fmt)
                    ws2.write(r, 6, d["Y"], norm_fmt)
                    ws2.write(r, 7, d["Val"], norm_fmt)
                    ws2.write(r, 8, d["Size"], norm_fmt)

                    crop_path = d.get("CropPath", "")
                    if crop_path and os.path.exists(crop_path):
                        ws2.insert_image(r, 9, crop_path, {'x_offset': 5, 'y_offset': 2})

                ws2.set_column(0, 0, 25)
                ws2.set_column(9, 9, 12)

            workbook.close()
            self.log_signal.emit(f"✅ Excel Saved: {excel_path}")

        except Exception as e:
            log.error(f"Excel Error: {e}")
            self.log_signal.emit(f"⚠️ Excel Error: {e}")

    def stop(self):
        self.is_running = False


# BatchCropWorker 可以保持不变，或者也建议同样的逻辑替换为 ThreadPoolExecutor
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
        self.matrix_data = {}
        self.processed_files = []
        self.coords_record = []

    def run(self):
        # 简单起见，Crop 逻辑暂时保持单线程或之前的逻辑，
        # 如果 Crop 也想并行，可以使用 ThreadPoolExecutor 改造
        # 这里为了不改动太多，先保持原样，因为 Crop 一般不会报 SharedMemory 错误
        # ... (您原来的 BatchCropWorker 代码) ...
        # 如果您原先的代码是单线程循环，那没问题。
        # 如果是 ProcessPoolExecutor，也请改为 ThreadPoolExecutor

        # 以下是之前提供的 BatchCropWorker 逻辑 (单线程循环版)，直接使用即可
        try:
            log.info("Batch Crop Processing Started")
            self.out_dir.mkdir(parents=True, exist_ok=True)
            self.crop_save_dir = self.out_dir / "matrix_crops"
            self.crop_save_dir.mkdir(exist_ok=True)
        except Exception as e:
            log.error(f"Crop Output Dir Error: {e}")
            self.log_signal.emit(f"❌ Output Error: {e}")
            self.finished_signal.emit()
            return

        files = []
        valid_ext = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        if self.in_dir.exists():
            for f in self.in_dir.iterdir():
                if f.is_file() and f.suffix.lower() in valid_ext:
                    if self.filter_str and self.filter_str.lower() not in f.name.lower():
                        continue
                    files.append(f)

        files.sort(key=lambda x: x.name)
        total = len(files)
        if total == 0:
            self.log_signal.emit("⚠️ No matching images found.")
            self.finished_signal.emit()
            return

        self.log_signal.emit(f"🚀 Found {total} files. Processing...")

        if self.config['mode'] == 'coords':
            self.coords_record = self.config['coord_list']
            radius, out_size = self.config['snap_params']
            resize_target = (out_size, out_size)
            resize_enabled = True
        else:
            rx, ry, rw, rh = self.config['rect']
            cx = rx + rw // 2
            cy = ry + rh // 2
            self.coords_record = [(cx, cy)]
            radius = None
            resize_enabled = self.config.get('resize_enabled', False)
            resize_target = self.config.get('resize_target', (60, 60))

        for file_idx, f in enumerate(files):
            if not self.is_running: break
            try:
                img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
                if img is None: continue
                self.processed_files.append(f.name)

                for coord_idx, (cx, cy) in enumerate(self.coords_record):
                    if self.config['mode'] == 'coords':
                        half = radius
                        x_tl, y_tl = cx - half, cy - half
                        w_box = h_box = half * 2
                    else:
                        x_tl, y_tl, w_box, h_box = self.config['rect']

                    fx, fy, fw, fh = get_safe_roi(img.shape, x_tl, y_tl, w_box, h_box)
                    if fw <= 0 or fh <= 0: continue
                    crop = img[fy:fy + fh, fx:fx + fw]

                    if crop.size > 0:
                        if resize_enabled:
                            crop = cv2.resize(crop, resize_target, interpolation=cv2.INTER_NEAREST)

                        save_name = f"F{file_idx}_C{coord_idx}.png"
                        full_path = self.crop_save_dir / save_name
                        cv2.imwrite(str(full_path), crop)
                        self.matrix_data[(coord_idx, file_idx)] = str(full_path)

                if file_idx % 5 == 0:
                    self.log_signal.emit(f"✅ Processed: {f.name}")
                self.progress_signal.emit(file_idx + 1, total)

            except Exception as e:
                log.error(f"crop Error {f.name}: {e}")
                self.log_signal.emit(f"❌ Error {f.name}: {e}")

        if self.is_running and self.matrix_data:
            self.log_signal.emit("📊 Generating Matrix Excel...")
            # 传入高度用于设置行高 (如果没有 resize_target, 给个默认值)
            row_h = resize_target[1] if 'resize_target' in locals() else 60
            self.generate_matrix_excel(row_h)

        self.finished_signal.emit()

    def generate_matrix_excel(self, img_height):
        # ... (保持原样) ...
        # 复用您上传的 generate_matrix_excel 逻辑
        excel_path = self.out_dir / "Comparison_Matrix.xlsx"
        try:
            wb = xlsxwriter.Workbook(str(excel_path))
            ws = wb.add_worksheet("Matrix")
            fmt_header = wb.add_format(
                {'bold': True, 'bg_color': '#333', 'font_color': 'white', 'border': 1, 'align': 'center',
                 'valign': 'vcenter'})
            fmt_coord = wb.add_format(
                {'bold': True, 'bg_color': '#eee', 'border': 1, 'align': 'center', 'valign': 'vcenter'})
            fmt_norm = wb.add_format({'border': 1, 'align': 'center', 'valign': 'vcenter'})

            ws.write(0, 0, "Center X", fmt_header)
            ws.write(0, 1, "Center Y", fmt_header)

            for col_idx, fname in enumerate(self.processed_files):
                ws.write(0, col_idx + 2, fname, fmt_header)
                ws.set_column(col_idx + 2, col_idx + 2, img_height / 6)

            for row_idx, (cx, cy) in enumerate(self.coords_record):
                excel_row = row_idx + 1
                ws.set_row(excel_row, img_height + 5)
                ws.write(excel_row, 0, cx, fmt_coord)
                ws.write(excel_row, 1, cy, fmt_coord)

                for col_idx in range(len(self.processed_files)):
                    key = (row_idx, col_idx)
                    if key in self.matrix_data:
                        img_path = self.matrix_data[key]
                        ws.insert_image(excel_row, col_idx + 2, img_path,
                                        {'x_offset': 5, 'y_offset': 2, 'object_position': 1})
                    else:
                        ws.write(excel_row, col_idx + 2, "N/A", fmt_norm)
            wb.close()
            self.log_signal.emit(f"🏆 Excel Saved: {excel_path}")
        except Exception as e:
            log.error(f"Crop Excel Error: {e}")
            self.log_signal.emit(f"⚠️ Excel Error: {e}")

    def stop(self):
        self.is_running = False
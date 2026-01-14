import sys
import os
import csv
import cv2
import xlsxwriter
from pathlib import Path
import numpy as np  # 🟢 [新增] 必须导入 numpy
from utils.logger import log  # 🟢 导入日志
from PyQt6.QtWidgets import QDialog  # 部分 helper 可能用到


# 1. 定义 BASE_DIR 和 get_base_path
def get_base_path():
    if hasattr(sys, '_MEIPASS'):
        return sys._MEIPASS
    return os.path.abspath(".")


BASE_DIR = get_base_path()


# 2. 放入 get_safe_roi 函数
def get_safe_roi(image_shape, x, y, w, h):
    """
    智能计算安全的截图区域，防止越界崩溃
    返回: (final_x, final_y, final_w, final_h)
    """
    img_h, img_w = image_shape[:2]

    # 1. 修正起始点 (不能小于0)
    x = max(0, int(x))
    y = max(0, int(y))

    # 2. 修正宽高 (如果 起始点+宽 > 图片总宽，则缩小宽)
    final_w = min(int(w), img_w - x)
    final_h = min(int(h), img_h - y)

    # 3. 防止宽高变为负数或0
    final_w = max(1, final_w)
    final_h = max(1, final_h)

    return x, y, final_w, final_h


# 🟢 [新增] FileHandler 类 (负责安全的图片读取)
class FileHandler:
    @staticmethod
    def load_image_file(file_path, width, height, channels, bit_depth):
        """
        统一读取图像文件 (支持 Raw/Bin/Bmp/Png/Jpg) 并包含异常处理
        """
        try:
            log.info(f"Loading image: {file_path} | Params: W={width}, H={height}, C={channels}, Bit={bit_depth}")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            ext = os.path.splitext(file_path)[-1].lower()

            # --- RAW / BIN ---
            if ext in ['.raw', '.bin']:
                file_size = os.path.getsize(file_path)
                expected_pixels = width * height * channels

                # 简单估算理论大小 (仅用于日志警告)
                bytes_per_pixel = 2 if bit_depth > 8 else 1
                expected_size = expected_pixels * bytes_per_pixel

                if file_size != expected_size:
                    log.warning(f"File size mismatch! Real: {file_size}, Expected (approx): {expected_size}")

                dtype = np.uint16 if bit_depth > 8 else np.uint8

                # 读取数据
                img = np.fromfile(file_path, dtype=dtype)

                # 校验数据完整性
                if img.size != expected_pixels:
                    error_msg = f"Pixel count mismatch. Read {img.size}, expected {width}x{height}x{channels}={expected_pixels}"
                    log.error(error_msg)
                    raise ValueError(error_msg)

                img = img.reshape((height, width, channels))

                if channels == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                return img

            # --- BMP / PNG / JPG ---
            elif ext in ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                # 支持中文路径读取
                img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
                if img is None:
                    raise IOError(f"OpenCV failed to decode image: {file_path}")

                # 如果是灰度图但需要当做多通道处理 (或者反之)，这里视情况转换
                # 这里保持原样返回，由 Worker 或 Algorithm 进一步处理
                return img

            else:
                raise TypeError(f"Unsupported file extension: {ext}")

        except Exception as e:
            log.error(f"Failed to load image: {file_path}\nError: {str(e)}", exc_info=True)
            return None

# 3. 放入 ExportHandler 类
class ExportHandler:
    @staticmethod
    def save_report(data, original_img, filename_stem, output_dir, stats=None, specs=None, save_details=True,
                    snap_params=(5, 60)):
        """
        生成 Excel 和 CSV 报告
        包含：Cluster ID 逻辑，单 Cluster 单截图逻辑
        """
        # 解包参数
        snap_radius, snap_size = snap_params

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        crop_dir = out_path / "crops"
        if save_details:
            crop_dir.mkdir(exist_ok=True)

        # 1. 导出 CSV (包含 Cluster ID)
        csv_path = out_path / f"{filename_stem}_detail.csv"
        try:
            with open(csv_path, 'w', newline='', encoding='utf-8-sig') as cf:
                writer = csv.writer(cf)
                # 写入表头
                writer.writerow(['CH', 'Cluster ID', 'Type', 'Polarity', 'X', 'Y', 'Val', 'Size'])
                for d in data:
                    writer.writerow([
                        d['ch'],
                        d.get('cluster_id', 0),  # 写入 ID
                        d['final_type'],
                        d['polarity'],
                        d['gx'],
                        d['gy'],
                        d['val'],
                        d.get('size', 1)
                    ])
        except Exception as e:
            log.error(f"CSV Export Error: {e}")  # 🟢 [修改] print -> log.error

        # 2. 生成 Excel
        # 🟢 [定义 excel_path]
        excel_path = out_path / f"{filename_stem}_Report.xlsx"

        # 准备数据列表
        excel_details = []
        # 🟢 [定义 h, w]
        h, w = original_img.shape[:2]
        # 🟢 [定义 saved_count]
        saved_count = 0

        # 🟢 [定义 seen_ids 用于去重截图]
        seen_ids = set()

        # --- 数据准备循环 ---
        for d in data:
            crop_path_str = ""
            cid = d.get('cluster_id', 0)

            # 截图逻辑：Cluster 类型且 ID 未处理过，或者 ID=0 (Single)
            # 如果你只想截 Cluster 的图，可以加 "Cluster" in d['final_type'] 判断
            if save_details and ("Cluster" in d['final_type']):
                # 核心逻辑：ID > 0 且没见过，或者是 ID=0 的异常 Cluster
                if cid == 0 or (cid > 0 and cid not in seen_ids):
                    gx, gy = d['gx'], d['gy']

                    # 计算截图范围
                    half = snap_radius
                    y_s, y_e = max(0, int(gy - half)), min(h, int(gy + half))
                    x_s, x_e = max(0, int(gx - half)), min(w, int(gx + half))

                    src_crop = original_img[y_s:y_e, x_s:x_e]

                    if src_crop.size > 0:
                        vis_crop = cv2.resize(src_crop, (snap_size, snap_size), interpolation=cv2.INTER_NEAREST)
                        # 文件名带上 CID
                        crop_name = f"crop_{filename_stem}_CID{cid}_{saved_count}.png"
                        full_crop_path = crop_dir / crop_name
                        cv2.imwrite(str(full_crop_path), vis_crop)

                        crop_path_str = str(full_crop_path)
                        saved_count += 1

                        # 标记该 ID 已截图
                        if cid > 0:
                            seen_ids.add(cid)

            row_data = d.copy()
            row_data['CropPath'] = crop_path_str
            row_data['Filename'] = filename_stem
            row_data['Cluster ID'] = cid  # 确保 ID 被记录
            excel_details.append(row_data)

        # --- 写入 Excel ---
        try:
            wb = xlsxwriter.Workbook(str(excel_path))

            # 定义样式
            fmt_header = wb.add_format(
                {'bold': True, 'bg_color': '#333', 'font_color': 'white', 'border': 1, 'align': 'center',
                 'valign': 'vcenter'})
            fmt_norm = wb.add_format({'align': 'center', 'border': 1, 'valign': 'vcenter'})
            fmt_pass = wb.add_format(
                {'bg_color': '#C6EFCE', 'font_color': '#006100', 'align': 'center', 'border': 1, 'valign': 'vcenter'})
            fmt_fail = wb.add_format(
                {'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'align': 'center', 'border': 1, 'valign': 'vcenter'})

            # Sheet 1: Summary
            ws1 = wb.add_worksheet("Summary")
            headers_sum = ["Filename", "Result", "Total Pixels", "Total Clusters", "White Pixels", "Black Pixels",
                           "White Clusters", "Black Clusters"]
            ws1.write_row(0, 0, headers_sum, fmt_header)
            ws1.set_column(0, 0, 25)

            if stats and specs:
                max_pts, max_cls = specs
                total_cls = stats['white_cls'] + stats['black_cls']
                is_fail = (stats['total_pts'] > max_pts) or (total_cls > max_cls)
                res_str = "FAIL" if is_fail else "PASS"
                res_fmt = fmt_fail if is_fail else fmt_pass

                ws1.write(1, 0, filename_stem, fmt_norm)
                ws1.write(1, 1, res_str, res_fmt)
                ws1.write(1, 2, stats['total_pts'], fmt_norm)
                ws1.write(1, 3, total_cls, fmt_norm)
                ws1.write(1, 4, stats['white_pts'], fmt_norm)
                ws1.write(1, 5, stats['black_pts'], fmt_norm)
                ws1.write(1, 6, stats['white_cls'], fmt_norm)
                ws1.write(1, 7, stats['black_cls'], fmt_norm)
            else:
                ws1.write(1, 0, filename_stem, fmt_norm)
                ws1.write(1, 1, "N/A", fmt_norm)
                ws1.write(1, 2, len(data), fmt_norm)

            # Sheet 2: Details (带 Cluster ID)
            ws2 = wb.add_worksheet("Defect_Details")
            headers_det = ["Filename", "Cluster ID", "CH", "Type", "Polarity", "X", "Y", "Val", "Size", "Snapshot"]
            ws2.write_row(0, 0, headers_det, fmt_header)

            ws2.set_column(0, 0, 20)
            ws2.set_column(9, 9, 12)

            # 3. 循环写入数据
            # Excel 最大行数限制 (保留几行给表头和底部)
            MAX_ROWS = 1048500

            for r, item in enumerate(excel_details, start=1):
                ws2.set_row(r, 65)
                # 🟢 [新增] 超限检查
                if r > MAX_ROWS:
                    ws2.write(r, 0, "⚠️ DATA TRUNCATED: EXCEL ROW LIMIT REACHED", fmt_norm)
                    print(
                        f"⚠️ Warning: Too many defects ({len(excel_details)}). Excel output truncated at row {MAX_ROWS}.")
                    break
                # 读取 CID，兼容大小写键名
                cid = item.get('Cluster ID', item.get('cluster_id', 0))

                ws2.write(r, 0, item['Filename'], fmt_norm)
                ws2.write(r, 1, cid, fmt_norm)  # 写入 Cluster ID
                ws2.write(r, 2, item['ch'], fmt_norm)
                ws2.write(r, 3, item['final_type'], fmt_norm)
                ws2.write(r, 4, item['polarity'], fmt_norm)
                ws2.write(r, 5, item['gx'], fmt_norm)
                ws2.write(r, 6, item['gy'], fmt_norm)
                ws2.write(r, 7, item['val'], fmt_norm)
                ws2.write(r, 8, item.get('size', 1), fmt_norm)

                if item['CropPath'] and os.path.exists(item['CropPath']):
                    ws2.insert_image(r, 9, item['CropPath'], {'x_offset': 5, 'y_offset': 2})
                    # 🟢 [新增] 只有在行数合法时才插入图片，否则 xlsxwriter 会报 Warning
                    if r <= 1048576:
                        ws2.insert_image(r, 9, item['CropPath'], {'x_offset': 5, 'y_offset': 2})
            wb.close()
            # 🟢 [使用 excel_path]
            return str(excel_path)

        except Exception as e:
            log.error(f"Excel Export Error: {e}", exc_info=True)  # 🟢 [修改] print -> log.error
            return None
    pass
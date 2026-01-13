import sys
import os
import csv
import cv2
import xlsxwriter
from pathlib import Path
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
    # 逻辑: 想要截取的右边界是 x+w，但不能超过 img_w
    final_w = min(int(w), img_w - x)
    final_h = min(int(h), img_h - y)

    # 3. 防止宽高变为负数或0 (比如 x 已经在图片外了)
    final_w = max(1, final_w)
    final_h = max(1, final_h)

    return x, y, final_w, final_h
    pass
# 3. 放入 ExportHandler 类
class ExportHandler:
    @staticmethod
    def save_report(data, original_img, filename_stem, output_dir, stats=None, specs=None, save_details=True, snap_params=(5, 60)):
        """
        stats: 字典, 包含 {'total_pts', 'white_cls', ...} (由 CoreAlgorithm.get_stats 生成)
        specs: 元组, (max_pts, max_cls)
        """
        # 解包参数
        snap_radius, snap_size = snap_params

        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        crop_dir = out_path / "crops"
        if save_details:
            crop_dir.mkdir(exist_ok=True)

        # 1. 导出 CSV (保持不变，含 Size)
        csv_path = out_path / f"{filename_stem}_detail.csv"
        with open(csv_path, 'w', newline='', encoding='utf-8-sig') as cf:
            writer = csv.writer(cf)
            writer.writerow(['CH', 'Type', 'Polarity', 'X', 'Y', 'Val', 'Size'])
            for d in data:
                writer.writerow([d['ch'], d['final_type'], d['polarity'], d['gx'], d['gy'], d['val'], d.get('size', 1)])

        # 2. 生成 Excel
        excel_path = out_path / f"{filename_stem}_Report.xlsx"

        # 准备截图列表
        excel_details = []
        h, w = original_img.shape[:2]
        saved_count = 0

        for d in data:
            crop_path_str = ""
            if save_details and ("Cluster" in d['final_type']):
                gx, gy = d['gx'], d['gy']
                # 🟢 [修改] 使用传入的参数
                half = snap_radius

                y_s, y_e = max(0, int(gy - half)), min(h, int(gy + half))
                x_s, x_e = max(0, int(gx - half)), min(w, int(gx + half))
                src_crop = original_img[y_s:y_e, x_s:x_e]

                if src_crop.size > 0:
                    # 🟢 [修改] 使用传入的 resize 尺寸
                    vis_crop = cv2.resize(src_crop, (snap_size, snap_size), interpolation=cv2.INTER_NEAREST)
                    crop_name = f"crop_{filename_stem}_{d['final_type']}_{saved_count}.png"
                    full_crop_path = crop_dir / crop_name
                    cv2.imwrite(str(full_crop_path), vis_crop)
                    crop_path_str = str(full_crop_path)
                    saved_count += 1

            row_data = d.copy()
            row_data['CropPath'] = crop_path_str
            row_data['Filename'] = filename_stem
            excel_details.append(row_data)

        try:
            wb = xlsxwriter.Workbook(str(excel_path))

            # --- 定义样式 (和 Batch 保持一致) ---
            fmt_header = wb.add_format(
                {'bold': True, 'bg_color': '#333', 'font_color': 'white', 'border': 1, 'align': 'center',
                 'valign': 'vcenter'})
            fmt_norm = wb.add_format({'align': 'center', 'border': 1, 'valign': 'vcenter'})
            fmt_pass = wb.add_format(
                {'bg_color': '#C6EFCE', 'font_color': '#006100', 'align': 'center', 'border': 1, 'valign': 'vcenter'})
            fmt_fail = wb.add_format(
                {'bg_color': '#FFC7CE', 'font_color': '#9C0006', 'align': 'center', 'border': 1, 'valign': 'vcenter'})

            # ==========================================================
            # 🟢 Sheet 1: Summary (完全复刻批量输出格式)
            # ==========================================================
            ws1 = wb.add_worksheet("Summary")

            # 表头
            headers_sum = ["Filename", "Result", "Total Pixels", "Total Clusters", "White Pixels", "Black Pixels",
                           "White Clusters", "Black Clusters"]
            ws1.write_row(0, 0, headers_sum, fmt_header)
            ws1.set_column(0, 0, 25)  # 文件名列宽

            # 计算 Result
            if stats and specs:
                max_pts, max_cls = specs
                total_cls = stats['white_cls'] + stats['black_cls']
                is_fail = (stats['total_pts'] > max_pts) or (total_cls > max_cls)
                res_str = "FAIL" if is_fail else "PASS"
                res_fmt = fmt_fail if is_fail else fmt_pass

                # 写入数据行
                ws1.write(1, 0, filename_stem, fmt_norm)
                ws1.write(1, 1, res_str, res_fmt)
                ws1.write(1, 2, stats['total_pts'], fmt_norm)
                ws1.write(1, 3, total_cls, fmt_norm)
                ws1.write(1, 4, stats['white_pts'], fmt_norm)
                ws1.write(1, 5, stats['black_pts'], fmt_norm)
                ws1.write(1, 6, stats['white_cls'], fmt_norm)
                ws1.write(1, 7, stats['black_cls'], fmt_norm)
            else:
                # 备用：如果没传 stats，只写简单数据
                ws1.write(1, 0, filename_stem, fmt_norm)
                ws1.write(1, 1, "N/A", fmt_norm)
                ws1.write(1, 2, len(data), fmt_norm)

            # ==========================================================
            # Sheet 2: Details (保持之前的逻辑，带 Size 和 Snapshot)
            # ==========================================================
            ws2 = wb.add_worksheet("Defect_Details")
            headers_det = ["Filename", "CH", "Type", "Polarity", "X", "Y", "Val", "Size", "Snapshot"]
            ws2.write_row(0, 0, headers_det, fmt_header)

            ws2.set_column(0, 0, 20)
            ws2.set_column(8, 8, 12)

            for r, item in enumerate(excel_details, start=1):
                ws2.set_row(r, 65)
                ws2.write(r, 0, item['Filename'], fmt_norm)
                ws2.write(r, 1, item['ch'], fmt_norm)
                ws2.write(r, 2, item['final_type'], fmt_norm)
                ws2.write(r, 3, item['polarity'], fmt_norm)
                ws2.write(r, 4, item['gx'], fmt_norm)
                ws2.write(r, 5, item['gy'], fmt_norm)
                ws2.write(r, 6, item['val'], fmt_norm)
                ws2.write(r, 7, item.get('size', 1), fmt_norm)  # Size

                if item['CropPath'] and os.path.exists(item['CropPath']):
                    ws2.insert_image(r, 8, item['CropPath'], {'x_offset': 5, 'y_offset': 2})

            wb.close()
            return str(excel_path)
        except Exception as e:
            print(f"Excel Export Error: {e}")
            return None
    pass
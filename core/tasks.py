import cv2
import numpy as np
from core.algorithm import CoreAlgorithm # 引用算法

# ==========================================
# 🚀 顶层函数：单张图片处理任务 (修复版)
# ==========================================
def process_single_image_task(f_path, out_dir, params, specs, snap_params, export_details):
    import cv2
    import numpy as np

    try:
        f_name = f_path.name
        img = cv2.imread(str(f_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return {'status': 'error', 'msg': f"Read Error: {f_name}", 'filename': f_name}

        # --- 1. 核心算法 ---
        vis, data = CoreAlgorithm.run_dispatch(img, params)

        # --- 2. 统计 ---
        h, w = img.shape[:2]
        g_dist = params.get('g_dist', 5)
        stats = CoreAlgorithm.get_stats(data, (h, w), g_dist)

        max_pts, max_cls = specs
        total_cluster_cnt = stats['white_cls'] + stats['black_cls']
        is_fail = (stats['total_pts'] > max_pts) or (total_cluster_cnt > max_cls)
        result_str = "FAIL" if is_fail else "PASS"

        # --- 3. 保存结果图 ---
        cv2.imwrite(str(out_dir / f"{f_path.stem}_result.png"), vis)

        # --- 4. 截图逻辑 & Excel数据准备 ---
        saved_crops_for_excel = []
        seen_cluster_ids = set()  # <--- [新增] 记录已截图的 ID

        if export_details:
            crop_dir = out_dir / "crops"
            crop_dir.mkdir(exist_ok=True, parents=True)

            snap_radius, snap_size = snap_params
            saved_count = 0

            for d in data:
                dtype = d.get('final_type', 'Single')
                cid = d.get('cluster_id', 0)  # 获取 ID
                full_crop_path_str = ""  # 默认为空

                # 截图逻辑
                if "Cluster" in dtype:
                    # 🟢 [核心修改] 只有当 Cluster ID 未出现过时，才截图
                    # 如果 cid == 0 (异常情况)，则保持原样截图
                    if cid == 0 or (cid > 0 and cid not in seen_cluster_ids):
                        gx, gy = d['gx'], d['gy']
                        half = snap_radius
                        y_s, y_e = max(0, int(gy - half)), min(h, int(gy + half))
                        x_s, x_e = max(0, int(gx - half)), min(w, int(gx + half))
                        src_crop = img[y_s:y_e, x_s:x_e]

                        if src_crop.size > 0:
                            vis_crop = cv2.resize(src_crop, (snap_size, snap_size), interpolation=cv2.INTER_NEAREST)
                            # 文件名带上 ID
                            crop_filename = f"crop_{f_path.stem}_CID{cid}_{saved_count}.png"
                            full_crop_path = crop_dir / crop_filename
                            cv2.imwrite(str(full_crop_path), vis_crop)

                            full_crop_path_str = str(full_crop_path)
                            saved_count += 1

                            # 标记该 ID 已处理
                            if cid > 0: seen_cluster_ids.add(cid)

                # 更新数据用于 CSV/Excel
                d['CropPath'] = full_crop_path_str  # 没截图的就是空字符串
                d['Size'] = d.get('size', 1)
                d['ClusterID'] = cid  # <--- [新增] 将 ID 存入数据

                excel_item = {
                    "Filename": f_name,
                    "Cluster ID": cid,  # <--- [新增] Excel 列
                    "CH": d['ch'],
                    "Type": dtype,
                    "Polarity": "White" if d.get('polarity') == 'Bright' else "Black",
                    "X": d['gx'],
                    "Y": d['gy'],
                    "Val": d['val'],
                    "Size": d.get('size', 1),
                    "CropPath": full_crop_path_str
                }
                saved_crops_for_excel.append(excel_item)
        # --- 5. 返回结果 ---
        return {
            'status': 'success',
            'filename': f_name,
            'file_stem': f_path.stem,  # 🟢 [修复 Process Error 'file_stem'] 必须包含此键
            'result_str': result_str,
            'stats': stats,
            'data': data,  # 原始数据 -> 给 CSV 用
            'summary_row': {  # 汇总数据 -> 给 Excel Sheet1 用
                "Filename": f_name,
                "Result": result_str,
                "Total_Points": stats['total_pts'],
                "White_Points": stats['white_pts'], "Black_Points": stats['black_pts'],
                "Total_Clusters": total_cluster_cnt,
                "White_Clusters": stats['white_cls'], "Black_Clusters": stats['black_cls']
            },
            'cluster_details': saved_crops_for_excel  # 详情数据 -> 给 Excel Sheet2 用
        }

    except Exception as e:
        # 错误时也要返回 filename，方便日志定位
        return {'status': 'error', 'msg': f"Error {f_path.name}: {str(e)}", 'filename': f_path.name}
    pass
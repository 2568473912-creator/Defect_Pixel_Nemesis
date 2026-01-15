import cv2
import numpy as np
from core.algorithm import CoreAlgorithm

# 🟢 常量定义
TYPE_SINGLE = 0
TYPE_CH_CLUSTER = 1
TYPE_SP_CLUSTER = 2
POLARITY_BRIGHT = 0
POLARITY_DARK = 1


def process_single_image_task(f_path, out_dir, params, specs, snap_params, export_details):
    try:
        f_name = f_path.name
        img = cv2.imread(str(f_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return {'status': 'error', 'msg': f"Read Error: {f_name}", 'filename': f_name}

        # --- 1. 核心算法 ---
        # data 可能是 NumPy 结构化数组，也可能是 DefectPoint 对象列表
        vis, data = CoreAlgorithm.run_dispatch(img, params)

        # --- 2. 统计 ---
        h, w = img.shape[:2]
        g_dist = params.get('g_dist', 5)
        # 兼容性处理：如果 data 是 NumPy 数组，get_stats 应该能处理；如果是对象列表也能处理
        stats = CoreAlgorithm.get_stats(data, (h, w), g_dist)

        max_pts, max_cls = specs
        total_cluster_cnt = stats.get('white_cls', 0) + stats.get('black_cls', 0)
        is_fail = (stats.get('total_pts', 0) > max_pts) or (total_cluster_cnt > max_cls)
        result_str = "FAIL" if is_fail else "PASS"

        # --- 3. 保存结果图 ---
        # 如果 out_dir 不存在则创建
        if not out_dir.exists():
            out_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(out_dir / f"{f_path.stem}_result.png"), vis)

        # --- 4. 截图逻辑 & Excel数据准备 ---
        saved_crops_for_excel = []
        seen_cluster_ids = set()

        if export_details and len(data) > 0:
            crop_dir = out_dir / "crops"

            snap_radius, snap_size = snap_params
            saved_count = 0

            # 遍历数据
            for d in data:
                # 🟢 [兼容性读取] 支持 对象属性访问 和 字典/NumPy访问
                # 定义一个内部 helper 来安全获取属性
                def get_val(item, key, default=None):
                    # 1. 尝试字典/NumPy索引访问
                    try:
                        return item[key]
                    except (TypeError, IndexError, ValueError, KeyError):
                        pass
                    # 2. 尝试属性访问
                    if hasattr(item, key):
                        return getattr(item, key)
                    # 3. 尝试 .get() 方法
                    if hasattr(item, 'get'):
                        return item.get(key, default)
                    return default

                # 读取基础字段 (全部转为 Python 原生类型，防止序列化报错)
                ftype_val = get_val(d, 'final_type', 'Single')
                # 兼容 V5 数组 (int) 和 V4 对象 (str)
                if isinstance(ftype_val, int):
                    if ftype_val == TYPE_CH_CLUSTER:
                        ftype_str = "Channel_Cluster"
                    elif ftype_val == TYPE_SP_CLUSTER:
                        ftype_str = "Spatial_Cluster"
                    else:
                        ftype_str = "Single"
                    ftype_int = ftype_val
                else:
                    ftype_str = str(ftype_val)
                    ftype_int = -1  # 未知或对象模式

                cid = int(get_val(d, 'cluster_id', 0))
                gx = int(get_val(d, 'gx', 0))
                gy = int(get_val(d, 'gy', 0))
                val = int(get_val(d, 'val', 0))
                ch = int(get_val(d, 'ch', 0))

                pol_val = get_val(d, 'polarity', 0)
                if isinstance(pol_val, int):
                    pol_str = "Black" if pol_val == POLARITY_DARK else "White"
                else:
                    pol_str = "White" if str(pol_val) == 'Bright' else "Black"

                size = int(get_val(d, 'size', 1))

                full_crop_path_str = ""

                # 截图逻辑 (仅针对 Cluster, 或者你想截所有的也可以改条件)
                is_cluster = ("Cluster" in ftype_str)

                if is_cluster:
                    if not crop_dir.exists():
                        crop_dir.mkdir(exist_ok=True, parents=True)

                    # ID=0 也可以截图，或者 ID>0 且未见过的 Cluster 截图
                    if cid == 0 or (cid > 0 and cid not in seen_cluster_ids):
                        half = snap_radius
                        y_s, y_e = max(0, int(gy - half)), min(h, int(gy + half))
                        x_s, x_e = max(0, int(gx - half)), min(w, int(gx + half))
                        src_crop = img[y_s:y_e, x_s:x_e]

                        if src_crop.size > 0:
                            vis_crop = cv2.resize(src_crop, (snap_size, snap_size), interpolation=cv2.INTER_NEAREST)
                            crop_filename = f"crop_{f_path.stem}_CID{cid}_{saved_count}.png"
                            full_crop_path = crop_dir / crop_filename
                            cv2.imwrite(str(full_crop_path), vis_crop)

                            full_crop_path_str = str(full_crop_path)
                            saved_count += 1
                            if cid > 0: seen_cluster_ids.add(cid)

                # 🟢 [修复关键]
                # 尝试将 CropPath 写回 d (供 workers.py 里的 CSV 导出使用)
                # 如果 d 是对象且有 __slots__ 限制，且没定义 CropPath，这里会报错
                # 所以我们用 try-except 包裹，或者仅当它是 dict/numpy 时写入
                try:
                    # 如果 d 支持 item assignment
                    d['CropPath'] = full_crop_path_str
                except:
                    try:
                        # 如果 d 是对象，尝试 setattr
                        setattr(d, 'CropPath', full_crop_path_str)
                    except:
                        pass  # 无法写入也无所谓，Excel 数据在下面生成

                # 🟢 [修复关键]
                # 不要执行 d['Size'] = ... 或 d['ClusterID'] = ...
                # 直接构建要返回的字典
                excel_item = {
                    "Filename": f_name,
                    "Cluster ID": cid,  # 使用变量
                    "CH": ch,
                    "Type": ftype_str,
                    "Polarity": pol_str,
                    "X": gx,
                    "Y": gy,
                    "Val": val,
                    "Size": size,  # 使用变量
                    "CropPath": full_crop_path_str
                }
                saved_crops_for_excel.append(excel_item)

        # --- 5. 返回结果 ---
        return {
            'status': 'success',
            'filename': f_name,
            'file_stem': f_path.stem,
            'result_str': result_str,
            'stats': stats,
            'data': data,  # 原始数据 -> 给 CSV 用
            'summary_row': {
                "Filename": f_name,
                "Result": result_str,
                "Total_Points": stats.get('total_pts', 0),
                "White_Points": stats.get('white_pts', 0), "Black_Points": stats.get('black_pts', 0),
                "Total_Clusters": total_cluster_cnt,
                "White_Clusters": stats.get('white_cls', 0), "Black_Clusters": stats.get('black_cls', 0)
            },
            'cluster_details': saved_crops_for_excel  # 列表字典 -> 给 Excel 用
        }

    except Exception as e:
        import traceback
        return {'status': 'error', 'msg': f"Error {f_path.name}: {str(e)}\n{traceback.format_exc()}",
                'filename': f_path.name}
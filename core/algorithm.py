import cv2
import numpy as np
from numba import jit

# 1. 放入 _numba_extract_points 函数 (注意装饰器 @jit)
@jit(nopython=True)
def _numba_extract_points(ys, xs, ch_neighbor_counts, sub_img, bg, img_gray_val, step, offset_x, offset_y, mode_int,
                          is_16bit):
    """
    Numba 加速的坏点提取逻辑
    mode_int: 0=Dark, 1=Bright
    """
    n = len(ys)

    # 预分配结果数组
    res_gx = np.empty(n, dtype=np.int32)
    res_gy = np.empty(n, dtype=np.int32)
    res_val = np.empty(n, dtype=np.int32)
    res_is_cluster = np.empty(n, dtype=np.int8)  # 1=Cluster, 0=Single
    res_is_dark = np.empty(n, dtype=np.int8)  # 1=Dark, 0=Bright

    for i in range(n):
        py = ys[i]
        px = xs[i]

        # 1. 获取邻域计数 (假设 ch_neighbor_counts 是 float32)
        # 归一化：卷积结果是 255 的倍数
        val_count = ch_neighbor_counts[py, px]
        count = int(val_count / 255.0 + 0.5)

        if count > 1:
            res_is_cluster[i] = 1
        else:
            res_is_cluster[i] = 0

        # 2. 坐标换算
        gx = px * step + offset_x
        gy = py * step + offset_y
        res_gx[i] = gx
        res_gy[i] = gy

        # 3. 极性判断
        # 默认 Bright (White) -> is_dark = 0
        is_dark = 0
        if mode_int == 1:  # Bright mode
            if sub_img[py, px] < bg[py, px]:
                is_dark = 1
        res_is_dark[i] = is_dark

        # 4. 获取像素值
        raw_val = img_gray_val[gy, gx]
        if is_16bit:
            res_val[i] = int(raw_val / 256)
        else:
            res_val[i] = int(raw_val)

    return res_gx, res_gy, res_val, res_is_cluster, res_is_dark

    pass

# 2. 放入 CoreAlgorithm 类
class CoreAlgorithm:
    @staticmethod
    def robust_masked_blur(image, filter_size, bg_ceiling=None):
        if bg_ceiling is not None:
            mask = (image < bg_ceiling).astype(np.float32)
            kernel = np.ones((filter_size, filter_size), dtype=np.float32)
            # cv2.filter2D 已经非常快了，保持原样
            sum_valid = cv2.filter2D(image * mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
            count_valid = cv2.filter2D(mask, -1, kernel, borderType=cv2.BORDER_REFLECT)
            count_valid[count_valid == 0] = 1.0
            return sum_valid / count_valid
        else:
            return cv2.blur(image.astype(np.float32), (filter_size, filter_size))

    @staticmethod
    def process_dark_field(img_raw, channels, filter_size, threshold_8bit, ch_cluster_dist, global_cluster_dist):
        return CoreAlgorithm._internal_process(img_raw, channels, filter_size,
                                               mode='Dark', param1=threshold_8bit,
                                               ch_dist=ch_cluster_dist, g_dist=global_cluster_dist)

    @staticmethod
    def process_bright_field(img_raw, channels, filter_size, threshold_pct, ch_cluster_dist, global_cluster_dist):
        return CoreAlgorithm._internal_process(img_raw, channels, filter_size,
                                               mode='Bright', param1=threshold_pct,
                                               ch_dist=ch_cluster_dist, g_dist=global_cluster_dist)

    @staticmethod
    def _internal_process(img_raw, channels, filter_size, mode, param1, ch_dist, g_dist):
        if img_raw is None: return None, []

        is_16bit = (img_raw.dtype == np.uint16)
        if len(img_raw.shape) == 3:
            img_gray_val = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        else:
            img_gray_val = img_raw

        step = int(np.sqrt(channels))
        detected_points = []

        # 显示用底图
        vis_display = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if len(vis_display.shape) == 2: vis_display = cv2.cvtColor(vis_display, cv2.COLOR_GRAY2RGB)

        h_img, w_img = img_raw.shape[:2]

        # 预备 Numba 需要的参数
        mode_int = 0 if mode == 'Dark' else 1
        is_16bit_bool = bool(is_16bit)

        for offset_y in range(step):
            for offset_x in range(step):
                ch_idx = (offset_y * step + offset_x) + 1
                sub_img = img_raw[offset_y::step, offset_x::step].astype(np.float32)

                # 1. 计算背景
                if mode == 'Dark':
                    thresh_val = param1 * 256 if is_16bit else param1
                    bg_ceiling = thresh_val * 2
                    bg = CoreAlgorithm.robust_masked_blur(sub_img, filter_size, bg_ceiling)
                else:
                    bg = CoreAlgorithm.robust_masked_blur(sub_img, filter_size, None)

                # 2. 生成 Mask
                if mode == 'Dark':
                    diff = sub_img - bg
                    _, mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
                else:
                    pct = param1 / 100.0
                    mask_high = (sub_img > bg * (1 + pct)).astype(np.uint8) * 255
                    mask_low = (sub_img < bg * (1 - pct)).astype(np.uint8) * 255
                    mask = cv2.bitwise_or(mask_high, mask_low)

                if cv2.countNonZero(mask) == 0: continue

                # 3. 提取点信息 (优化版)
                # mask = mask.astype(np.uint8)
                kernel_c = np.ones((ch_dist, ch_dist), dtype=np.float32)
                ch_neighbor_counts = cv2.filter2D(mask, -1, kernel_c, borderType=cv2.BORDER_CONSTANT)

                # 仅在需要提取坐标时转为 uint8 (节省这次转换开销)
                mask_u8 = mask.astype(np.uint8)
                ys, xs = np.where(mask_u8 > 0)

                # 🚀 替换原有的 for zip(ys, xs) 循环，改用 Numba 加速
                if len(ys) > 0:
                    r_gx, r_gy, r_val, r_cluster, r_dark = _numba_extract_points(
                        ys, xs, ch_neighbor_counts, sub_img, bg, img_gray_val,
                        step, offset_x, offset_y, mode_int, is_16bit_bool
                    )

                    # 快速构建结果列表 (Python 构建 dict 还是比较快，只要避免了内部的计算)
                    for i in range(len(ys)):
                        raw_type = "Channel_Cluster" if r_cluster[i] else "Single"
                        polarity = "Dark" if r_dark[i] else "Bright"
                        detected_points.append({
                            'gx': int(r_gx[i]),
                            'gy': int(r_gy[i]),
                            'ch': ch_idx,
                            'raw_type': raw_type,
                            'final_type': raw_type,
                            'val': int(r_val[i]),
                            'polarity': polarity
                        })

        # ==========================================================
        # 4. 全局聚合 (优化版：向量化赋值)
        # ==========================================================
        if detected_points:
            # 🚀 向量化：一次性生成全局掩膜，不再使用 for 循环
            g_mask = np.zeros((h_img, w_img), dtype=np.uint8)

            # 提取所有坐标
            all_gys = [pt['gy'] for pt in detected_points]
            all_gxs = [pt['gx'] for pt in detected_points]

            # Numpy 高级索引赋值 (极快)
            g_mask[all_gys, all_gxs] = 1

            # 卷积计算邻域
            g_kernel = np.ones((g_dist, g_dist), dtype=np.float32)
            neighbor_counts = cv2.filter2D(g_mask.astype(np.float32), -1, g_kernel, borderType=cv2.BORDER_CONSTANT)

            # 🚀 向量化读取计数 (可选优化，这里用列表推导式也很快)
            # 为了保持 detected_points 字典结构的完整性，这里用 lookup
            # 这里的 lookup 因为已经是指针/整数访问，速度足够快
            for pt in detected_points:
                count = neighbor_counts[pt['gy'], pt['gx']]
                if count > 1 and pt['raw_type'] == "Single":
                    pt['final_type'] = "Spatial_Cluster"

        # ==========================================================
        # 4.5: 计算 Cluster Size (保持逻辑，微调性能)
        # ==========================================================
        for pt in detected_points: pt['size'] = 1

        cluster_pts = [p for p in detected_points if "Cluster" in p['final_type']]
        if cluster_pts:
            mask_cls = np.zeros((h_img, w_img), dtype=np.uint8)
            # 🚀 向量化赋值
            c_gys = [p['gy'] for p in cluster_pts]
            c_gxs = [p['gx'] for p in cluster_pts]
            mask_cls[c_gys, c_gxs] = 255

            step = int(np.sqrt(channels))
            visual_kernel_size = max(g_dist, ch_dist * step)
            if visual_kernel_size % 2 == 0: visual_kernel_size += 1

            kernel = np.ones((visual_kernel_size, visual_kernel_size), np.uint8)
            dilated_mask = cv2.dilate(mask_cls, kernel, iterations=1)
            num_labels, labels = cv2.connectedComponents(dilated_mask, connectivity=8)

            label_counts = {}
            # 这里必须遍历点来统计，因为我们要知道"哪些坏点"属于哪个 Label
            # 这一步量级通常较小，Python 循环可以接受
            for pt in cluster_pts:
                lbl = labels[pt['gy'], pt['gx']]
                if lbl > 0:
                    label_counts[lbl] = label_counts.get(lbl, 0) + 1

            for pt in cluster_pts:
                lbl = labels[pt['gy'], pt['gx']]
                if lbl in label_counts:
                    pt['size'] = label_counts[lbl]

        # ==========================================================
        # 5. 绘图逻辑 (优化版：向量化绘图)
        # ==========================================================
        PADDING = 8
        LINE_THICKNESS = 1

        # 5.1 绘制点 (使用 Numpy 索引批量赋值颜色)
        # 将点分类
        normal_pts_indices = []
        normal_colors = []
        cluster_pts_for_bbox = []

        for pt in detected_points:
            if "Cluster" not in pt['final_type']:
                is_dark = pt['polarity'] == "Dark"
                color = [0, 255, 0] if not is_dark else [255, 0, 0]
                # vis_display[pt['gy'], pt['gx']] = color # 单点赋值较慢
                normal_pts_indices.append((pt['gy'], pt['gx']))
                normal_colors.append(color)
            else:
                cluster_pts_for_bbox.append(pt)

        # 🚀 批量绘制普通点
        if normal_pts_indices:
            # 拆分坐标和颜色
            ind = np.array(normal_pts_indices)
            cols = np.array(normal_colors, dtype=np.uint8)
            # 批量赋值
            vis_display[ind[:, 0], ind[:, 1]] = cols

        # 5.2 绘制 Cluster 包围框 (保持逻辑，使用向量化 Mask)
        if cluster_pts_for_bbox:
            cluster_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            # 🚀 向量化赋值
            c_gys = [p['gy'] for p in cluster_pts_for_bbox]
            c_gxs = [p['gx'] for p in cluster_pts_for_bbox]
            cluster_mask[c_gys, c_gxs] = 255

            kernel = np.ones((g_dist, g_dist), np.uint8)
            dilated_cluster_mask = cv2.dilate(cluster_mask, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated_cluster_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # 统计极性逻辑保持不变
                dark_count = 0
                total_count_in_box = 0
                for pt in cluster_pts_for_bbox:
                    if x <= pt['gx'] < x + w and y <= pt['gy'] < y + h:
                        total_count_in_box += 1
                        if pt['polarity'] == "Dark": dark_count += 1

                is_mostly_dark = (total_count_in_box > 0) and (dark_count / total_count_in_box > 0.5)
                c_color = (0, 0, 255) if not is_mostly_dark else (255, 0, 255)

                tl_x = max(0, x - PADDING)
                tl_y = max(0, y - PADDING)
                br_x = min(w_img, x + w + PADDING)
                br_y = min(h_img, y + h + PADDING)
                cv2.rectangle(vis_display, (tl_x, tl_y), (br_x, br_y), c_color, LINE_THICKNESS)

        return vis_display, detected_points

    @staticmethod
    def generate_channel_grid(img_raw, channels):
        # 保持原样，这部分主要是切片操作，已经是 Numpy 原生速度
        if img_raw is None or channels <= 1: return img_raw
        if len(img_raw.shape) == 3:
            img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_raw
        if img_gray.dtype == np.uint16:
            # 右移 8 位 等同于 除以 256 (仅对整数有效)
            img_vis_base = (img_gray >> 8).astype(np.uint8)
        else:
            img_vis_base = img_gray.astype(np.uint8)
        step = int(np.sqrt(channels))
        h_raw, w_raw = img_gray.shape[:2]
        target_sub_h = h_raw // step
        target_sub_w = w_raw // step
        if target_sub_h == 0 or target_sub_w == 0: return img_raw
        grid_h = target_sub_h * step
        grid_w = target_sub_w * step
        canvas = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)
        neon_colors = [(0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255), (255, 255, 0)]
        for offset_y in range(step):
            for offset_x in range(step):
                ch_idx = (offset_y * step + offset_x) + 1
                extracted = img_vis_base[offset_y::step, offset_x::step]
                sub_img = extracted[:target_sub_h, :target_sub_w]
                sub_img_bgr = cv2.cvtColor(sub_img, cv2.COLOR_GRAY2BGR)
                start_y = offset_y * target_sub_h
                start_x = offset_x * target_sub_w
                end_y = start_y + target_sub_h
                end_x = start_x + target_sub_w
                canvas[start_y:end_y, start_x:end_x] = sub_img_bgr
                border_color = neon_colors[ch_idx % len(neon_colors)]
                cv2.rectangle(canvas, (start_x, start_y), (end_x - 1, end_y - 1), border_color, 2)
        return canvas

    @staticmethod
    def get_stats(data, img_shape, global_dist):
        # 1. 基础计数
        white_points = [d for d in data if d.get('polarity', 'Bright') == 'Bright']
        black_points = [d for d in data if d.get('polarity', 'Bright') == 'Dark']
        cnt_pts_white = len(white_points)
        cnt_pts_black = len(black_points)

        # 2. 计算 Cluster 团数量 (优化版)
        def count_clusters(pt_list):
            c_pts = [d for d in pt_list if d['final_type'] != 'Single']
            if not c_pts: return 0

            h, w = img_shape[:2]
            temp_mask = np.zeros((h, w), dtype=np.uint8)

            # 🚀 向量化赋值：替代原来的 for 循环
            if c_pts:
                gys = [cp['gy'] for cp in c_pts]
                gxs = [cp['gx'] for cp in c_pts]
                temp_mask[gys, gxs] = 255

            kernel = np.ones((global_dist, global_dist), np.uint8)
            dilated = cv2.dilate(temp_mask, kernel, iterations=1)
            num_labels, _, _, _ = cv2.connectedComponentsWithStats(dilated, connectivity=8)
            return num_labels - 1

        cnt_cls_white = count_clusters(white_points)
        cnt_cls_black = count_clusters(black_points)

        return {
            "total_pts": len(data),
            "white_pts": cnt_pts_white,
            "black_pts": cnt_pts_black,
            "white_cls": cnt_cls_white,
            "black_cls": cnt_cls_black
        }
    @staticmethod
    def run_dispatch(img, params):
        """
        根据参数模式 (Dark/Bright) 调度不同的处理函数
        """
        if params['mode'] == 'Dark':
            return CoreAlgorithm.process_dark_field(
                img, params['ch'], params['fs'],
                params['thresh'], params['ch_dist'], params['g_dist']
            )
        else:
            return CoreAlgorithm.process_bright_field(
                img, params['ch'], params['fs'],
                params['thresh'], params['ch_dist'], params['g_dist']
            )
    pass
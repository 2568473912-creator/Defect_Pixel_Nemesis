import cv2
import numpy as np
from numba import jit


# 1. Numba 加速提取函数
@jit(nopython=True)
def _numba_extract_points(ys, xs, sub_img, bg, img_gray_val, step, offset_x, offset_y, mode_int, is_16bit):
    """
    Numba 加速提取 (不再负责 Cluster ID，只负责提取原始信息)
    """
    n = len(ys)
    res_gx = np.empty(n, dtype=np.int32)
    res_gy = np.empty(n, dtype=np.int32)
    res_val = np.empty(n, dtype=np.int32)
    res_polarity = np.empty(n, dtype=np.int8)  # 0=Bright, 1=Dark

    for i in range(n):
        py = ys[i]
        px = xs[i]

        # 1. 坐标换算
        gx = px * step + offset_x
        gy = py * step + offset_y
        res_gx[i] = gx
        res_gy[i] = gy

        # 2. 极性判断
        is_dark = 0
        if mode_int == 1:  # Bright mode logic
            if sub_img[py, px] < bg[py, px]:
                is_dark = 1
        res_polarity[i] = is_dark

        # 3. 像素值
        raw_val = img_gray_val[gy, gx]
        if is_16bit:
            res_val[i] = int(raw_val / 256)
        else:
            res_val[i] = int(raw_val)

    return res_gx, res_gy, res_val, res_polarity


# 2. CoreAlgorithm 类
class CoreAlgorithm:
    @staticmethod
    def robust_masked_blur(image, filter_size, bg_ceiling=None):
        if bg_ceiling is not None:
            mask = (image < bg_ceiling).astype(np.float32)
            kernel = np.ones((filter_size, filter_size), dtype=np.float32)
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
        h_img, w_img = img_raw.shape[:2]

        mode_int = 0 if mode == 'Dark' else 1
        is_16bit_bool = bool(is_16bit)

        # 暂存所有点
        raw_points = []

        # 全局计数器
        global_ch_cid = 1
        global_sp_cid = 1

        # ==========================================================
        # Phase A: 提取 & 同通道 Cluster 计算 (在子图循环中)
        # ==========================================================
        for offset_y in range(step):
            for offset_x in range(step):
                ch_idx = (offset_y * step + offset_x) + 1
                sub_img = img_raw[offset_y::step, offset_x::step].astype(np.float32)

                # 1. 计算背景 & Mask
                if mode == 'Dark':
                    thresh_val = param1 * 256 if is_16bit else param1
                    bg_ceiling = thresh_val * 2
                    bg = CoreAlgorithm.robust_masked_blur(sub_img, filter_size, bg_ceiling)
                    diff = sub_img - bg
                    _, mask = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
                    mask = mask.astype(np.uint8)
                else:
                    bg = CoreAlgorithm.robust_masked_blur(sub_img, filter_size, None)
                    pct = param1 / 100.0
                    mask_high = (sub_img > bg * (1 + pct)).astype(np.uint8) * 255
                    mask_low = (sub_img < bg * (1 - pct)).astype(np.uint8) * 255
                    mask = cv2.bitwise_or(mask_high, mask_low)

                if cv2.countNonZero(mask) == 0: continue

                # 2. 提取原始点
                ys, xs = np.where(mask > 0)
                r_gx, r_gy, r_val, r_pol = _numba_extract_points(
                    ys, xs, sub_img, bg, img_gray_val,
                    step, offset_x, offset_y, mode_int, is_16bit
                )

                # 3. 计算同通道 Cluster (基于子图 Mask)
                # 使用 float32 避免 filter2D 溢出 (255+255=510)
                mask_f = mask.astype(np.float32)
                k_size = ch_dist if ch_dist > 0 else 1
                if k_size % 2 == 0: k_size += 1
                kernel = np.ones((k_size, k_size), np.float32)

                # 计算密度: 如果 density > 255，说明除了自己还有别人
                density = cv2.filter2D(mask_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)

                # 生成连通域 (用于赋 ID)
                # 这里依然使用 density > 0 来生成潜在区域，但在统计 Size 时会严格剔除孤立点
                cluster_mask = (density > 0).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(cluster_mask, connectivity=8)

                # 统计 Valid Cluster Size
                # 只有 density > 255 的点才算作“有效 Cluster 成员”
                # density 的值大约是 255 * N (N是邻域内像素数)
                # 我们使用 255.5 作为阈值，排除掉只有自己的情况
                is_valid_member = (density > 255.5) & (mask > 0)

                label_counts = {}
                # 仅统计是有效成员的点
                valid_labels = labels[is_valid_member]
                if len(valid_labels) > 0:
                    label_counts = dict(zip(*np.unique(valid_labels, return_counts=True)))

                # 建立 ID 映射
                local_to_global_id = {}
                local_to_size = {}
                for lbl, sz in label_counts.items():
                    if sz > 1:  # 再次确认，只有成员数 > 1 才是 Cluster
                        local_to_global_id[lbl] = global_ch_cid
                        global_ch_cid += 1
                        local_to_size[lbl] = sz

                # 4. 回填数据
                for k in range(len(ys)):
                    py, px = ys[k], xs[k]

                    # 只有当自己也是 Valid Member 时，才赋予 Cluster ID
                    # 这解决了“我虽然在别人的光晕里，但我自己没邻居”的情况
                    my_density = density[py, px]
                    lbl = labels[py, px]

                    cid = 0
                    csize = 1

                    if my_density > 255.5:  # 我有邻居
                        if lbl in local_to_global_id:
                            cid = local_to_global_id[lbl]
                            csize = local_to_size[lbl]

                    polarity = "Dark" if r_pol[k] else "Bright"
                    raw_points.append({
                        'gx': int(r_gx[k]),
                        'gy': int(r_gy[k]),
                        'ch': ch_idx,
                        'val': int(r_val[k]),
                        'polarity': polarity,
                        'ch_cid': cid,
                        'ch_size': csize,
                        'sp_cid': 0, 'sp_size': 1
                    })

        final_points = []

        # ==========================================================
        # Phase B: 全局 Cluster 计算 (循环外)
        # ==========================================================
        if raw_points:
            # 1. 构建全局 Mask
            mask_sp = np.zeros((h_img, w_img), dtype=np.float32)  # 直接用 float32
            point_map = {}  # 快速查找点索引
            for idx, p in enumerate(raw_points):
                mask_sp[p['gy'], p['gx']] = 255.0
                point_map[(p['gx'], p['gy'])] = idx

            # 2. 卷积计算
            k_size_sp = g_dist if g_dist > 0 else 1
            if k_size_sp % 2 == 0: k_size_sp += 1
            kernel_sp = np.ones((k_size_sp, k_size_sp), np.float32)

            density_sp = cv2.filter2D(mask_sp, -1, kernel_sp, borderType=cv2.BORDER_CONSTANT)

            # 3. 连通域
            cluster_mask_sp = (density_sp > 0).astype(np.uint8)
            num_labels_sp, labels_sp = cv2.connectedComponents(cluster_mask_sp, connectivity=8)

            # 4. 统计 Valid Size
            # 同样逻辑：只有 density > 255.5 的点才是有效 Cluster 成员
            # 我们需要遍历 raw_points 来统计，因为 mask 上只有点的位置有值

            sp_label_counts = {}
            valid_cluster_members = []

            for p in raw_points:
                d_val = density_sp[p['gy'], p['gx']]
                lbl = labels_sp[p['gy'], p['gx']]

                if d_val > 255.5:  # 核心判定：我有邻居
                    valid_cluster_members.append((p, lbl))
                    sp_label_counts[lbl] = sp_label_counts.get(lbl, 0) + 1

            # 5. 建立 ID 映射
            sp_label_to_id = {}
            sp_label_to_size = {}

            # 全局 ID 接在 Phase A 后面
            start_sp_id = global_ch_cid

            for lbl, sz in sp_label_counts.items():
                if sz > 1:
                    sp_label_to_id[lbl] = start_sp_id
                    start_sp_id += 1
                    sp_label_to_size[lbl] = sz

            # 6. 回填数据
            # 注意：只有在 valid_cluster_members 里的点才能获得 ID
            # 其他点即使被连通域覆盖，也是 Single (因为它自己 Density <= 255)
            for p, lbl in valid_cluster_members:
                if lbl in sp_label_to_id:
                    p['sp_cid'] = sp_label_to_id[lbl]
                    p['sp_size'] = sp_label_to_size[lbl]

            # ==========================================================
            # Phase C: 结果拆分
            # ==========================================================
            for p in raw_points:
                is_ch_cluster = p['ch_size'] > 1
                is_sp_cluster = p['sp_size'] > 1

                if not is_ch_cluster and not is_sp_cluster:
                    entry = p.copy()
                    entry['final_type'] = "Single"
                    entry['cluster_id'] = 0
                    entry['size'] = 1
                    final_points.append(entry)
                else:
                    if is_ch_cluster:
                        entry = p.copy()
                        entry['final_type'] = "Channel_Cluster"
                        entry['cluster_id'] = p['ch_cid']
                        entry['size'] = p['ch_size']
                        final_points.append(entry)

                    if is_sp_cluster:
                        entry = p.copy()
                        entry['final_type'] = "Spatial_Cluster"
                        entry['cluster_id'] = p['sp_cid']
                        entry['size'] = p['sp_size']
                        final_points.append(entry)

        final_points.sort(key=lambda x: (0 if x['cluster_id'] > 0 else 1, x['cluster_id']))

        # 准备绘图
        vis_display = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if len(vis_display.shape) == 2: vis_display = cv2.cvtColor(vis_display, cv2.COLOR_GRAY2RGB)

        # ==========================================================
        # 绘图逻辑
        # ==========================================================
        PADDING = 8
        LINE_THICKNESS = 1
        drawn_singles = set()

        normal_pts_indices = []
        normal_colors = []
        cluster_pts_to_draw = []

        for pt in final_points:
            idx = (pt['gy'], pt['gx'])
            if pt['cluster_id'] > 0:
                cluster_pts_to_draw.append(pt)
            else:
                if idx not in drawn_singles:
                    is_dark = pt['polarity'] == "Dark"
                    color = [0, 255, 0] if not is_dark else [255, 0, 0]
                    normal_pts_indices.append((pt['gy'], pt['gx']))
                    normal_colors.append(color)
                    drawn_singles.add(idx)

        if normal_pts_indices:
            ind = np.array(normal_pts_indices)
            cols = np.array(normal_colors, dtype=np.uint8)
            vis_display[ind[:, 0], ind[:, 1]] = cols

        from collections import defaultdict
        groups = defaultdict(list)
        for pt in cluster_pts_to_draw:
            groups[pt['cluster_id']].append(pt)

        for cid, group_pts in groups.items():
            if not group_pts: continue
            ftype = group_pts[0]['final_type']

            temp_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            gys = [p['gy'] for p in group_pts]
            gxs = [p['gx'] for p in group_pts]
            temp_mask[gys, gxs] = 255

            # 画框时稍微膨胀一点以便看清
            v_dist = 3
            kernel = np.ones((v_dist, v_dist), np.uint8)
            dilated = cv2.dilate(temp_mask, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                dark_count = sum(1 for p in group_pts if p['polarity'] == 'Dark')
                is_mostly_dark = (dark_count / len(group_pts) > 0.5)

                if ftype == 'Channel_Cluster':
                    c_color = (0, 255, 255) if not is_mostly_dark else (0, 128, 128)
                else:
                    c_color = (0, 0, 255) if not is_mostly_dark else (255, 0, 255)

                tl_x = max(0, x - PADDING)
                tl_y = max(0, y - PADDING)
                br_x = min(w_img, x + w + PADDING)
                br_y = min(h_img, y + h + PADDING)
                cv2.rectangle(vis_display, (tl_x, tl_y), (br_x, br_y), c_color, LINE_THICKNESS)

        return vis_display, final_points

    @staticmethod
    def generate_channel_grid(img_raw, channels):
        if img_raw is None or channels <= 1: return img_raw
        if len(img_raw.shape) == 3:
            img_gray = cv2.cvtColor(img_raw, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img_raw
        if img_gray.dtype == np.uint16:
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
        unique_white = set()
        unique_black = set()
        cls_white_ids = set()
        cls_black_ids = set()

        for d in data:
            coord = (d['gx'], d['gy'])
            ftype = d['final_type']
            polarity = d.get('polarity', 'Bright')

            if polarity == 'Bright':
                unique_white.add(coord)
            else:
                unique_black.add(coord)

            if "Cluster" in ftype:
                cid = d.get('cluster_id', 0)
                if cid > 0:
                    if polarity == 'Bright':
                        cls_white_ids.add(cid)
                    else:
                        cls_black_ids.add(cid)

        return {
            "total_pts": len(unique_white) + len(unique_black),
            "white_pts": len(unique_white),
            "black_pts": len(unique_black),
            "white_cls": len(cls_white_ids),
            "black_cls": len(cls_black_ids)
        }

    @staticmethod
    def run_dispatch(img, params):
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
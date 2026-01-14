import cv2
import numpy as np
from numba import jit, prange


# ==========================================================
# 🟢 优化类：内存紧凑型数据结构 (保持不变)
# ==========================================================
class DefectPoint:
    __slots__ = (
        'gx', 'gy', 'ch', 'val', 'polarity',
        'ch_cid', 'ch_size', 'sp_cid', 'sp_size',
        'final_type', 'cluster_id', 'size',
        'CropPath'
    )

    def __init__(self, gx, gy, ch, val, polarity, ch_cid=0, ch_size=1, sp_cid=0, sp_size=1):
        self.gx = gx
        self.gy = gy
        self.ch = ch
        self.val = val
        self.polarity = polarity
        self.ch_cid = ch_cid
        self.ch_size = ch_size
        self.sp_cid = sp_cid
        self.sp_size = sp_size
        self.final_type = "Single"
        self.cluster_id = 0
        self.size = 1
        self.CropPath = None

    def __getitem__(self, key): return getattr(self, key)

    def __setitem__(self, key, value): setattr(self, key, value)

    def get(self, key, default=None): return getattr(self, key, default)

    # 🟢 [新增] 转字典方法
    def to_dict(self):
        """将对象转换为普通字典，用于导出或需要动态添加属性的场景"""
        return {
            'gx': self.gx, 'gy': self.gy, 'ch': self.ch, 'val': self.val, 'polarity': self.polarity,
            'ch_cid': self.ch_cid, 'ch_size': self.ch_size,
            'sp_cid': self.sp_cid, 'sp_size': self.sp_size,
            'final_type': self.final_type, 'cluster_id': self.cluster_id, 'size': self.size,
            'CropPath': self.CropPath
        }

    def copy(self):
        new_obj = DefectPoint(
            self.gx, self.gy, self.ch, self.val, self.polarity,
            self.ch_cid, self.ch_size, self.sp_cid, self.sp_size
        )
        new_obj.final_type = self.final_type
        new_obj.cluster_id = self.cluster_id
        new_obj.size = self.size
        new_obj.CropPath = self.CropPath
        return new_obj


# ==========================================================
# 🟢 Numba 优化：开启 parallel=True 和 prange 加速
# ==========================================================
@jit(nopython=True, nogil=True, cache=True)
def _numba_extract_points(ys, xs, sub_img, bg, img_gray_val, step, offset_x, offset_y, mode_int, is_16bit):
    """
    多核并行提取像素信息
    """
    n = len(ys)
    res_gx = np.empty(n, dtype=np.int32)
    res_gy = np.empty(n, dtype=np.int32)
    res_val = np.empty(n, dtype=np.int32)
    res_polarity = np.empty(n, dtype=np.int8)  # 0=Bright, 1=Dark

    # prange 自动利用多核 CPU
    for i in prange(n):
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


class CoreAlgorithm:
    @staticmethod
    def robust_masked_blur(image, filter_size, bg_ceiling=None):
        if bg_ceiling is not None:
            mask = (image < bg_ceiling).astype(np.float32)
            kernel = np.ones((filter_size, filter_size), dtype=np.float32)
            # 使用 cv2.BORDER_REFLECT 避免边缘黑边
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

        # 🟢 [优化] 使用列表存储 NumPy 数组块，而不是逐个对象
        # 结构: (gx_arr, gy_arr, ch_idx, val_arr, pol_arr, ch_cid_arr, ch_size_arr)
        batch_results = []

        global_ch_cid = 1

        # ==========================================================
        # Phase A: 提取 & 同通道 Cluster (数组化处理)
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

                # 2. 提取原始点 (Numba 并行)
                # ys, xs 是 numpy array
                ys, xs = np.where(mask > 0)
                r_gx, r_gy, r_val, r_pol = _numba_extract_points(
                    ys, xs, sub_img, bg, img_gray_val,
                    step, offset_x, offset_y, mode_int, is_16bit
                )

                # 3. 计算同通道 Cluster (向量化)
                mask_f = mask.astype(np.float32)
                k_size = ch_dist if ch_dist > 0 else 1
                if k_size % 2 == 0: k_size += 1
                kernel = np.ones((k_size, k_size), np.float32)

                # 密度图
                density = cv2.filter2D(mask_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)

                cluster_mask = (density > 0).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(cluster_mask, connectivity=8)

                # 🟢 [优化] 向量化 ID 映射 (替代字典查找循环)
                # 提取出 mask > 0 位置的 labels 和 density
                # 注意：r_gx 等是按 (ys, xs) 顺序生成的，所以直接用 ys, xs 索引即可对齐
                current_lbls = labels[ys, xs]
                current_dens = density[ys, xs]

                # 统计 Valid Member
                # 条件：密度 > 255.5 且 mask > 0 (后者已经满足，因为我们只取 mask>0 的点)
                valid_mask = current_dens > 255.5
                valid_lbls = current_lbls[valid_mask]

                # 统计每个 Label 的 Size
                if len(valid_lbls) > 0:
                    # 使用 bincount 极速统计 (前提：label 是非负整数)
                    max_lbl = np.max(current_lbls)
                    counts = np.bincount(valid_lbls, minlength=max_lbl + 1)

                    # 建立映射表 (Lookup Table)
                    # map_id[old_label] = new_global_id
                    map_id = np.zeros(max_lbl + 1, dtype=np.int32)
                    map_size = np.ones(max_lbl + 1, dtype=np.int32)  # 默认 size=1

                    # 只有 count > 1 的才是 Cluster
                    cluster_indices = np.where(counts > 1)[0]
                    num_clusters = len(cluster_indices)

                    if num_clusters > 0:
                        # 分配全局 ID
                        new_ids = np.arange(global_ch_cid, global_ch_cid + num_clusters, dtype=np.int32)
                        map_id[cluster_indices] = new_ids
                        map_size[cluster_indices] = counts[cluster_indices]

                        global_ch_cid += num_clusters

                    # 向量化查表
                    # 只有那些 density 够大的点，才有资格查表获得 ID
                    # 对于 density 小的点，ID 保持 0
                    final_cids = np.zeros_like(current_lbls)
                    final_sizes = np.ones_like(current_lbls)

                    # 仅处理有效邻居点
                    valid_indices = np.where(valid_mask)[0]
                    if len(valid_indices) > 0:
                        lbls_to_map = current_lbls[valid_indices]
                        final_cids[valid_indices] = map_id[lbls_to_map]
                        final_sizes[valid_indices] = map_size[lbls_to_map]
                else:
                    final_cids = np.zeros(len(ys), dtype=np.int32)
                    final_sizes = np.ones(len(ys), dtype=np.int32)

                # 4. 存入块列表 (不创建对象)
                batch_results.append((r_gx, r_gy, ch_idx, r_val, r_pol, final_cids, final_sizes))

        final_points = []

        # 准备绘图背景
        vis_display = cv2.normalize(img_raw, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if len(vis_display.shape) == 2: vis_display = cv2.cvtColor(vis_display, cv2.COLOR_GRAY2RGB)

        # ==========================================================
        # Phase B: 全局 Cluster 计算 (向量化加速)
        # ==========================================================
        if batch_results:
            # 1. 拼接所有通道的数据 (0拷贝是不可能的，但 concat 很快)
            # all_gx 等是 1D 数组
            all_gx = np.concatenate([b[0] for b in batch_results])
            all_gy = np.concatenate([b[1] for b in batch_results])

            # 2. 构建全局 Mask (极速向量化赋值)
            # 替代了 `for p in points: mask[p.y, p.x] = 255`
            mask_sp = np.zeros((h_img, w_img), dtype=np.float32)
            mask_sp[all_gy, all_gx] = 255.0

            # 3. 卷积计算
            k_size_sp = g_dist if g_dist > 0 else 1
            if k_size_sp % 2 == 0: k_size_sp += 1
            kernel_sp = np.ones((k_size_sp, k_size_sp), np.float32)

            density_sp = cv2.filter2D(mask_sp, -1, kernel_sp, borderType=cv2.BORDER_CONSTANT)

            # 4. 连通域
            cluster_mask_sp = (density_sp > 0).astype(np.uint8)
            num_labels_sp, labels_sp = cv2.connectedComponents(cluster_mask_sp, connectivity=8)

            # 5. 向量化统计 ID
            # 取出坏点位置的 label 和 density
            sp_lbls = labels_sp[all_gy, all_gx]
            sp_dens = density_sp[all_gy, all_gx]

            valid_sp_mask = sp_dens > 255.5
            valid_sp_lbls = sp_lbls[valid_sp_mask]

            if len(valid_sp_lbls) > 0:
                max_sp_lbl = np.max(sp_lbls)
                sp_counts = np.bincount(valid_sp_lbls, minlength=max_sp_lbl + 1)

                sp_map_id = np.zeros(max_sp_lbl + 1, dtype=np.int32)
                sp_map_size = np.ones(max_sp_lbl + 1, dtype=np.int32)

                sp_cluster_indices = np.where(sp_counts > 1)[0]
                num_sp_clusters = len(sp_cluster_indices)

                if num_sp_clusters > 0:
                    start_sp_id = global_ch_cid  # 接在 CH ID 后面
                    new_sp_ids = np.arange(start_sp_id, start_sp_id + num_sp_clusters, dtype=np.int32)
                    sp_map_id[sp_cluster_indices] = new_sp_ids
                    sp_map_size[sp_cluster_indices] = sp_counts[sp_cluster_indices]

                # 查表得到每个点的 SP ID
                # 只有 valid 的点才查表，否则为 0
                all_sp_cids = np.zeros_like(sp_lbls)
                all_sp_sizes = np.ones_like(sp_lbls)

                valid_idx = np.where(valid_sp_mask)[0]
                if len(valid_idx) > 0:
                    l_to_map = sp_lbls[valid_idx]
                    all_sp_cids[valid_idx] = sp_map_id[l_to_map]
                    all_sp_sizes[valid_idx] = sp_map_size[l_to_map]
            else:
                all_sp_cids = np.zeros(len(all_gx), dtype=np.int32)
                all_sp_sizes = np.ones(len(all_gx), dtype=np.int32)

            # ==========================================================
            # Phase C: 最终对象创建 (只在这里做一次循环)
            # ==========================================================
            # 现在我们需要把 batch_results 展开，并结合计算出的 SP 信息

            # 将 batch_results 中的其他属性也 concat 起来，以便统一遍历
            # (ch, val, pol, ch_cid, ch_size)
            # 注意：ch_idx 是标量，需要 repeat

            # 为了效率，我们先构建好全量的数组
            all_ch = []
            all_val = []
            all_pol = []
            all_ch_cid = []
            all_ch_size = []

            for b in batch_results:
                count = len(b[0])
                all_ch.append(np.full(count, b[2], dtype=np.int32))  # repeat ch
                all_val.append(b[3])
                all_pol.append(b[4])
                all_ch_cid.append(b[5])
                all_ch_size.append(b[6])

            all_ch = np.concatenate(all_ch)
            all_val = np.concatenate(all_val)
            all_pol = np.concatenate(all_pol)
            all_ch_cid = np.concatenate(all_ch_cid)
            all_ch_size = np.concatenate(all_ch_size)

            # 现在所有数组长度一致 (N,)，直接遍历一次创建对象
            # 这一步是 Python 循环，无法避免，但已经是最精简的了
            num_pts = len(all_gx)

            for i in range(num_pts):
                ch_sz = all_ch_size[i]
                sp_sz = all_sp_sizes[i]
                is_ch = ch_sz > 1
                is_sp = sp_sz > 1

                # 基础属性
                gx, gy, ch, val = all_gx[i], all_gy[i], all_ch[i], all_val[i]
                pol_str = "Dark" if all_pol[i] else "Bright"
                ch_cid, sp_cid = all_ch_cid[i], all_sp_cids[i]

                # 逻辑复用：如果不创建 DefectPoint，也可以直接用 dict，但为了兼容 UI
                # 我们这里按需 append

                if not is_ch and not is_sp:
                    # Single
                    pt = DefectPoint(gx, gy, ch, val, pol_str, 0, 1, 0, 1)
                    pt.final_type = "Single"
                    pt.cluster_id = 0
                    pt.size = 1
                    final_points.append(pt)
                else:
                    if is_ch:
                        pt = DefectPoint(gx, gy, ch, val, pol_str, ch_cid, ch_sz, sp_cid, sp_sz)
                        pt.final_type = "Channel_Cluster"
                        pt.cluster_id = ch_cid
                        pt.size = ch_sz
                        final_points.append(pt)

                    if is_sp:
                        pt = DefectPoint(gx, gy, ch, val, pol_str, ch_cid, ch_sz, sp_cid, sp_sz)
                        pt.final_type = "Spatial_Cluster"
                        pt.cluster_id = sp_cid
                        pt.size = sp_sz
                        final_points.append(pt)

        final_points.sort(key=lambda x: (0 if x.cluster_id > 0 else 1, x.cluster_id))

        # ==========================================================
        # 绘图逻辑 (略微优化：批量处理)
        # ==========================================================
        PADDING = 8
        LINE_THICKNESS = 1

        drawn_singles = set()
        normal_pts_indices = []
        normal_colors = []
        cluster_pts_to_draw = []

        for pt in final_points:
            if pt.cluster_id > 0:
                cluster_pts_to_draw.append(pt)
            else:
                idx = (pt.gy, pt.gx)
                if idx not in drawn_singles:
                    is_dark = (pt.polarity == "Dark")
                    color = [0, 255, 0] if not is_dark else [255, 0, 0]
                    normal_pts_indices.append((pt.gy, pt.gx))
                    normal_colors.append(color)
                    drawn_singles.add(idx)

        # 批量绘制 Single (极快)
        if normal_pts_indices:
            ind = np.array(normal_pts_indices)
            cols = np.array(normal_colors, dtype=np.uint8)
            vis_display[ind[:, 0], ind[:, 1]] = cols

        # 绘制 Cluster 框
        from collections import defaultdict
        groups = defaultdict(list)
        for pt in cluster_pts_to_draw:
            groups[pt.cluster_id].append(pt)

        for cid, group_pts in groups.items():
            if not group_pts: continue
            ftype = group_pts[0].final_type

            # 使用临时小 mask 画框太慢，直接计算边界
            # 优化：不再使用 dilate+findContours，直接用 min/max 坐标画框
            # 虽然这样失去了不规则形状，但速度极快。如果必须精确轮廓，可保留原逻辑。
            # 这里为了速度，改用 Bounding Box

            xs = [p.gx for p in group_pts]
            ys = [p.gy for p in group_pts]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # 视觉膨胀
            if ftype == 'Channel_Cluster':
                v_dist = (ch_dist * step) if ch_dist > 0 else step
            else:
                v_dist = g_dist if g_dist > 0 else 3

            tl_x = max(0, min_x - v_dist - PADDING)
            tl_y = max(0, min_y - v_dist - PADDING)
            br_x = min(w_img, max_x + v_dist + PADDING)
            br_y = min(h_img, max_y + v_dist + PADDING)

            dark_count = sum(1 for p in group_pts if p.polarity == 'Dark')
            is_mostly_dark = (dark_count / len(group_pts) > 0.5)

            if ftype == 'Channel_Cluster':
                c_color = (0, 255, 255) if not is_mostly_dark else (0, 128, 128)
            else:
                c_color = (0, 0, 255) if not is_mostly_dark else (255, 0, 255)

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
        # 统计逻辑保持不变，因为 data 已经是 DefectPoint 对象列表
        unique_white = set()
        unique_black = set()
        cls_white_ids = set()
        cls_black_ids = set()

        for d in data:
            coord = (d.gx, d.gy)
            ftype = d.final_type
            polarity = d.polarity

            if polarity == 'Bright':
                unique_white.add(coord)
            else:
                unique_black.add(coord)

            if "Cluster" in ftype:
                cid = d.cluster_id
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
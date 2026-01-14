import cv2
import numpy as np
from numba import jit


# ==========================================================
# 🟢 新增：内存优化类 (替代字典)
# ==========================================================
class DefectPoint:
    # 使用 __slots__ 限制属性，避免创建 __dict__，显著节省内存
    __slots__ = (
        'gx', 'gy', 'ch', 'val', 'polarity',
        'ch_cid', 'ch_size', 'sp_cid', 'sp_size',
        'final_type', 'cluster_id', 'size',
        'CropPath'  # 预留字段，兼容 Excel 导出时的截图路径
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

        # 默认值
        self.final_type = "Single"
        self.cluster_id = 0
        self.size = 1
        self.CropPath = None

        # --- 字典兼容层 (让 UI 和 Exporter 无需修改代码) ---

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def get(self, key, default=None):
        return getattr(self, key, default)

    def copy(self):
        # 快速浅拷贝
        new_obj = DefectPoint(
            self.gx, self.gy, self.ch, self.val, self.polarity,
            self.ch_cid, self.ch_size, self.sp_cid, self.sp_size
        )
        new_obj.final_type = self.final_type
        new_obj.cluster_id = self.cluster_id
        new_obj.size = self.size
        new_obj.CropPath = self.CropPath
        return new_obj


# 1. Numba 加速提取函数
@jit(nopython=True)
def _numba_extract_points(ys, xs, sub_img, bg, img_gray_val, step, offset_x, offset_y, mode_int, is_16bit):
    """
    Numba 加速提取
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

        # 🟢 暂存所有点 (现在存储 DefectPoint 对象，而非字典)
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

                # 3. 计算同通道 Cluster (基于子图 Mask + filter2D)
                mask_f = mask.astype(np.float32)
                k_size = ch_dist if ch_dist > 0 else 1
                if k_size % 2 == 0: k_size += 1
                kernel = np.ones((k_size, k_size), np.float32)

                # 密度判定 > 255.5
                density = cv2.filter2D(mask_f, -1, kernel, borderType=cv2.BORDER_CONSTANT)

                cluster_mask = (density > 0).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(cluster_mask, connectivity=8)

                is_valid_member = (density > 255.5) & (mask > 0)

                label_counts = {}
                valid_labels = labels[is_valid_member]
                if len(valid_labels) > 0:
                    label_counts = dict(zip(*np.unique(valid_labels, return_counts=True)))

                # 建立映射
                local_to_global_id = {}
                local_to_size = {}
                for lbl, sz in label_counts.items():
                    if sz > 1:
                        local_to_global_id[lbl] = global_ch_cid
                        global_ch_cid += 1
                        local_to_size[lbl] = sz

                # 4. 回填数据 (🟢 使用 DefectPoint 对象)
                for k in range(len(ys)):
                    py, px = ys[k], xs[k]

                    my_density = density[py, px]
                    lbl = labels[py, px]

                    cid = 0
                    csize = 1

                    if my_density > 255.5:
                        if lbl in local_to_global_id:
                            cid = local_to_global_id[lbl]
                            csize = local_to_size[lbl]

                    polarity = "Dark" if r_pol[k] else "Bright"

                    # 🟢 创建对象
                    pt = DefectPoint(
                        gx=int(r_gx[k]),
                        gy=int(r_gy[k]),
                        ch=ch_idx,
                        val=int(r_val[k]),
                        polarity=polarity,
                        ch_cid=cid,
                        ch_size=csize
                    )
                    raw_points.append(pt)

        final_points = []

        # ==========================================================
        # Phase B: 全局 Cluster 计算
        # ==========================================================
        if raw_points:
            # 1. 构建全局 Mask (使用 float32)
            mask_sp = np.zeros((h_img, w_img), dtype=np.float32)

            # 使用对象属性访问，比 dict 快
            for p in raw_points:
                mask_sp[p.gy, p.gx] = 255.0

            # 2. 卷积计算
            k_size_sp = g_dist if g_dist > 0 else 1
            if k_size_sp % 2 == 0: k_size_sp += 1
            kernel_sp = np.ones((k_size_sp, k_size_sp), np.float32)

            density_sp = cv2.filter2D(mask_sp, -1, kernel_sp, borderType=cv2.BORDER_CONSTANT)

            # 3. 连通域
            cluster_mask_sp = (density_sp > 0).astype(np.uint8)
            num_labels_sp, labels_sp = cv2.connectedComponents(cluster_mask_sp, connectivity=8)

            # 4. 统计 Valid Size
            sp_label_counts = {}
            valid_cluster_members = []

            for p in raw_points:
                d_val = density_sp[p.gy, p.gx]
                lbl = labels_sp[p.gy, p.gx]

                if d_val > 255.5:  # 判定密度
                    valid_cluster_members.append((p, lbl))
                    sp_label_counts[lbl] = sp_label_counts.get(lbl, 0) + 1

            # 5. 建立 ID 映射
            sp_label_to_id = {}
            sp_label_to_size = {}
            start_sp_id = global_ch_cid

            for lbl, sz in sp_label_counts.items():
                if sz > 1:
                    sp_label_to_id[lbl] = start_sp_id
                    start_sp_id += 1
                    sp_label_to_size[lbl] = sz

            # 6. 回填 Phase B 结果
            for p, lbl in valid_cluster_members:
                if lbl in sp_label_to_id:
                    p.sp_cid = sp_label_to_id[lbl]
                    p.sp_size = sp_label_to_size[lbl]

            # ==========================================================
            # Phase C: 结果拆分 (复制对象)
            # ==========================================================
            for p in raw_points:
                is_ch_cluster = p.ch_size > 1
                is_sp_cluster = p.sp_size > 1

                if not is_ch_cluster and not is_sp_cluster:
                    # Single
                    entry = p.copy()
                    entry.final_type = "Single"
                    entry.cluster_id = 0
                    entry.size = 1
                    final_points.append(entry)
                else:
                    if is_ch_cluster:
                        # Channel Cluster
                        entry = p.copy()
                        entry.final_type = "Channel_Cluster"
                        entry.cluster_id = p.ch_cid
                        entry.size = p.ch_size
                        final_points.append(entry)

                    if is_sp_cluster:
                        # Spatial Cluster
                        entry = p.copy()
                        entry.final_type = "Spatial_Cluster"
                        entry.cluster_id = p.sp_cid
                        entry.size = p.sp_size
                        final_points.append(entry)

        # 排序 (注意 key 的兼容性，DefectPoint 支持 __getitem__ 所以 x['cluster_id'] 和 x.cluster_id 都能用)
        # 这里为了效率改为属性访问
        final_points.sort(key=lambda x: (0 if x.cluster_id > 0 else 1, x.cluster_id))

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
            idx = (pt.gy, pt.gx)
            if pt.cluster_id > 0:
                cluster_pts_to_draw.append(pt)
            else:
                if idx not in drawn_singles:
                    is_dark = pt.polarity == "Dark"
                    color = [0, 255, 0] if not is_dark else [255, 0, 0]
                    normal_pts_indices.append((pt.gy, pt.gx))
                    normal_colors.append(color)
                    drawn_singles.add(idx)

        if normal_pts_indices:
            ind = np.array(normal_pts_indices)
            cols = np.array(normal_colors, dtype=np.uint8)
            vis_display[ind[:, 0], ind[:, 1]] = cols

        from collections import defaultdict
        groups = defaultdict(list)
        for pt in cluster_pts_to_draw:
            groups[pt.cluster_id].append(pt)

        for cid, group_pts in groups.items():
            if not group_pts: continue
            ftype = group_pts[0].final_type

            temp_mask = np.zeros((h_img, w_img), dtype=np.uint8)
            gys = [p.gy for p in group_pts]
            gxs = [p.gx for p in group_pts]
            temp_mask[gys, gxs] = 255

            # 画框视觉膨胀
            if ftype == 'Channel_Cluster':
                v_dist = (ch_dist * step) if ch_dist > 0 else step
            else:
                v_dist = g_dist if g_dist > 0 else 3

            if v_dist % 2 == 0: v_dist += 1

            kernel = np.ones((v_dist, v_dist), np.uint8)
            dilated = cv2.dilate(temp_mask, kernel, iterations=1)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                dark_count = sum(1 for p in group_pts if p.polarity == 'Dark')
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

        # 🟢 data 是 DefectPoint 对象列表
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
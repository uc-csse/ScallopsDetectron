import numpy as np
import math
from utils import geo_utils
import geopandas as gpd
from shapely.geometry import *
import matplotlib.pyplot as plt

TRANSECT_GTEXTP_RANGE = 12  # 12  # Distance over which propagated GT looses all it's confidence
SCALE_CORRECTION_DEFAULT = 1.025

METRES_PER_FOOT = 0.3048

PRINT_WARNINGS = False

# Things that will break this code
#  -> Lines without GT points on them


def lineseg_distance(seg, pnt):
    seg_vec = seg[1] - seg[0]
    seg_len = np.linalg.norm(seg_vec)
    pnt_vec = seg[0] - pnt
    prp_d = np.cross(seg_vec, pnt_vec) / seg_len
    pll_d1 = np.sqrt(np.linalg.norm(seg[0] - pnt) ** 2 - prp_d ** 2)
    pll_d2 = np.sqrt(np.linalg.norm(seg[1] - pnt) ** 2 - prp_d ** 2)
    # Need slightly looser threshold to account for wonky transect line leaving gaps so no true perpendicular
    # Good for up to 5 degrees of line deflection
    seg_len_larger = seg_len + 0.1
    on_seg = pll_d1 < seg_len_larger and pll_d2 < seg_len_larger
    if pll_d2 > seg_len:
        pll_d1 *= -1
    return pll_d1 / seg_len if on_seg else -1, prp_d


class TransectMapper:
    def __init__(self, gps_scale_factor=SCALE_CORRECTION_DEFAULT):
        self.scale_factor = gps_scale_factor
        self.line_segs_gps = None
        self.transect_segs = None
        self.transect_dist_limits = [1e3, 0]    # min, max
        self.transect_segs_wm = None    # zero start with seg lengths defined by world length

    def create_map_from_gdf(self, transect_df, plot=False):
        line_segs = []
        gt_pts = []
        gt_vals = []
        for i, row in transect_df.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString):
                line_segs.append(np.array(geom.coords, dtype=np.float64)[:, :2])
            elif isinstance(geom, Point):
                label = row.NAME
                valid_gtl = True
                if 'Tape' in label:
                    continue
                if 'm' in label:
                    label_l = label.split('m')
                    gt_dist_m = float(label_l[0])
                elif 'f' in label:
                    label_l = label.split('f')
                    gt_dist_m = METRES_PER_FOOT * float(label_l[0])
                else:
                    print(f"Invalid GT Point label: {label}")
                    continue
                assert len(label_l) == 2
                gt_pts.append(np.array(geom.coords, dtype=np.float64)[0, :2])
                gt_vals.append(gt_dist_m)
            else:
                raise "Geom type not supported!"
        self.create_map(line_segs, gt_pts, gt_vals)
        if plot:
            self.plot_transect_map()
            self.plot_gps_gt()
            plt.show()

    def create_map_from_gpkg(self, gpkg_path, plot=False):
        transect_df = gpd.read_file(gpkg_path)
        self.create_map_from_gdf(transect_df, plot=plot)

    def create_map(self, line_segs_gps, gt_points, gt_td):
        # Create transect segments object
        self.line_segs_gps = [list(seg) for seg in line_segs_gps]
        transect_gts_vals = [[None]*len(seg) for seg in line_segs_gps]
        self.gps_gt = []
        self.transect_gt = []
        # Find closest line seg and set transect distance, create new vertex if not near existing, otherwise replace
        for gt_idx, (gt_gps_pnt, gt_td_par) in enumerate(zip(gt_points, gt_td)):
            idx, sub_idx, (f_par, d_per) = self.closest_lineseg_gps(gt_gps_pnt)
            par_offseg_err = (f_par > 1.0) * (f_par - 1.0) + (f_par < 0.0) * f_par
            if abs(par_offseg_err) > 0 or abs(d_per) > 0.05 or idx == -1:
                print(f"Bad GT point IDX {gt_idx} - too far from line")
                continue
            gps_seg_len = self.gps_lineseg_len(idx, sub_idx, sub_idx+1)
            d_par = f_par * gps_seg_len
            insert_new_pnt = 0.01 < d_par < (gps_seg_len - 0.01)
            sub_idx = sub_idx + int(d_par >= 0.01)
            # print(insert_new_pnt, sub_idx, gt_td_par, f_par, d_par)
            if insert_new_pnt:
                self.line_segs_gps[idx].insert(sub_idx, gt_gps_pnt)
                transect_gts_vals[idx].insert(sub_idx, gt_td_par)
            else:
                d_err = geo_utils.measure_chordlen(self.line_segs_gps[idx][sub_idx], gt_gps_pnt)
                assert d_err < 0.02
                # print("Dist from existing pnt to GT point =", d_err)
                self.line_segs_gps[idx][sub_idx] = gt_gps_pnt
                transect_gts_vals[idx][sub_idx] = gt_td_par
            # For debug:
            self.gps_gt.append(gt_gps_pnt)
            self.transect_gt.append(gt_td_par)

        # Calculate vertex transect distances
        self.transect_segs = []
        for seg_idx, seg in enumerate(self.line_segs_gps):
            # propagate transect distance to neighbouring verts for each GT with diminishing confidence, then combine
            # set valid flag at end of all subs
            if len([sp for sp in transect_gts_vals[seg_idx] if sp is not None]) == 0:
                print(f"No GT points found on segment IDX {seg_idx}!!")
                continue
            gt_seg_array_list = []
            for gt_idx, gt_val in enumerate(transect_gts_vals[seg_idx]):
                if gt_val is None:
                    continue
                gt_seg_array = np.zeros((len(seg), 2), dtype=np.float32)
                gt_seg_array[gt_idx] = [gt_val, 1.0]
                for walk_idx in range(len(seg)):
                    for w_dir in [-1, 1]:
                        new_idx = gt_idx + w_dir * (walk_idx + 1)
                        if (w_dir == -1 and new_idx < 0) or (w_dir == 1 and new_idx >= len(seg)):
                            continue
                        gps_dist = self.gps_lineseg_len(seg_idx, new_idx-w_dir, new_idx)
                        base_dist, base_conf = gt_seg_array[new_idx-w_dir]
                        # TODO: Tune confidence
                        # confidence = 1 / (walk_idx + 2) ** 2
                        new_conf = base_conf * (1.0 - min(0.99, gps_dist / TRANSECT_GTEXTP_RANGE))
                        gt_seg_array[new_idx] = [base_dist + w_dir * gps_dist, new_conf]
                gt_seg_array_list.append(gt_seg_array)
            gt_seg_arrays = np.array(gt_seg_array_list, dtype=np.float64)
            gt_seg_arrays[:, :, 0] *= gt_seg_arrays[:, :, 1]
            gt_seg_combined = np.sum(gt_seg_arrays, axis=0)
            estimated_transect_dists = gt_seg_combined[:, 0] / gt_seg_combined[:, 1]
            self.transect_segs.append(estimated_transect_dists)

        self.transect_segs_wm = [[0.0] * len(seg) for seg in self.line_segs_gps]
        self.transect_segs_wm[0][0] = self.transect_segs[0][0]
        for seg_idx, seg in enumerate(self.transect_segs_wm):
            for sub_idx in range(len(seg) - 1):
                gps_dist = self.gps_lineseg_len(seg_idx, sub_idx, sub_idx + 1)
                # print(self.line_segs_gps[seg_idx][sub_idx], self.line_segs_gps[seg_idx][sub_idx + 1], dist)
                self.transect_segs_wm[seg_idx][sub_idx + 1] = self.transect_segs_wm[seg_idx][sub_idx] + gps_dist
            if seg_idx < len(self.transect_segs_wm) - 1:
                self.transect_segs_wm[seg_idx + 1][0] = self.transect_segs_wm[seg_idx][-1]

        # Get transect parallel distance limits
        # for seg in self.transect_segs:
        #     for t_vert in seg:
        #         self.transect_dist_limits[0] = min(self.transect_dist_limits[0], t_vert)
        #         self.transect_dist_limits[1] = max(self.transect_dist_limits[0], t_vert)
        # print(self.transect_dist_limits)

    def gps_lineseg_len(self, seg_idx, sub_idx_0, sub_idx_1):
        assert abs(sub_idx_1 - sub_idx_0) == 1
        seg_len_m = geo_utils.measure_chordlen(self.line_segs_gps[seg_idx][sub_idx_0],
                                               self.line_segs_gps[seg_idx][sub_idx_1])
        return self.scale_factor * seg_len_m

    def closest_transect_vert(self, pnt):
        d_tran, d_perp = pnt
        closest_idx = -1
        closest_sub_idx = 0
        closest_distance = 1000
        for tseg_idx, tseg in enumerate(self.transect_segs):
            for sub_idx in range(len(tseg)):
                dist = d_tran - tseg[sub_idx]
                if dist < 0:
                    break
                if dist < closest_distance:
                    closest_distance = dist
                    closest_idx = tseg_idx
                    closest_sub_idx = sub_idx
        # Check for off end of transect
        if closest_idx != -1 and closest_sub_idx == len(self.transect_segs[closest_idx])-1:
            closest_idx = -1
        return closest_idx, closest_sub_idx, closest_distance

    def closest_lineseg_gps(self, pnt_gps):
        closest_tloc = (0, 1000)
        closest_idx = -1
        closest_sub_idx = 0
        tmp_closest_sub_idx = 0
        num_segs = len(self.line_segs_gps)
        for seg_idx, seg_gps in enumerate(self.line_segs_gps):
            datum = np.array(seg_gps[0], dtype=np.float64)
            local_pnt = self.scale_factor * geo_utils.convert_gps2local(datum, [pnt_gps])[0]
            local_segs = self.scale_factor * geo_utils.convert_gps2local(datum, np.array(seg_gps, dtype=np.float64))
            if len(local_segs) < 2:
                raise Exception("Line with less than 2 points!")
            closest_sub_tloc = (0.0, 1000.0, 1000.0)
            for sub_idx in range(len(local_segs) - 1):
                f_par, d_per = lineseg_distance(local_segs[sub_idx:sub_idx + 2], local_pnt)
                par_offseg_err = (f_par > 1.0) * (f_par - 1.0) + (f_par < 0.0) * f_par
                off_transect_ends = ((seg_idx == 0 and sub_idx == 0 and par_offseg_err < 0) or
                                     (seg_idx == num_segs and sub_idx == len(local_segs)-2 and par_offseg_err > 0))
                par_offseg_closer = abs(par_offseg_err) <= closest_sub_tloc[2]
                if f_par != -1 and par_offseg_closer and not off_transect_ends:
                    # if par_offseg_err > 0.0:
                    #     print("Off seg slightly", seg_idx, sub_idx, f_par, d_per, par_offseg_err)
                    # else:
                    #     print("on seg", seg_idx, sub_idx, f_par, d_per, par_offseg_err)
                    tmp_closest_sub_idx = sub_idx
                    closest_sub_tloc = (f_par, d_per, abs(par_offseg_err))
            f_par, d_per, _ = closest_sub_tloc
            if (-0.1 < f_par < 1.1) and abs(d_per) < abs(closest_tloc[1]) and abs(d_per) <= 1.0:
                closest_idx = seg_idx
                closest_sub_idx = tmp_closest_sub_idx
                closest_tloc = (f_par, d_per)
        return closest_idx, closest_sub_idx, closest_tloc

    def gps2transect(self, gps_pnt):
        # lon, lat
        idx, sub_idx, (f_par, d_per_m) = self.closest_lineseg_gps(gps_pnt)
        if idx == -1:
            if PRINT_WARNINGS:
                print(f"GPS point not in transect! {gps_pnt}")
            return None
        # print("gps2transect:", idx, sub_idx, f_par, d_per_m)
        td1 = self.transect_segs[idx][sub_idx]
        td2 = self.transect_segs[idx][sub_idx + 1]
        d_par_m = td1 + (td2 - td1) * f_par
        return d_par_m, d_per_m

    def transect2gps(self, transect_pnt):
        idx, sub_idx, closest_par_dist = self.closest_transect_vert(transect_pnt)
        if idx == -1:
            if PRINT_WARNINGS:
                print(f"Transect point not in map! {transect_pnt}")
            return None
        # print("transect2gps:", idx, sub_idx, closest_par_dist)
        gps1 = self.line_segs_gps[idx][sub_idx]
        gps2 = self.line_segs_gps[idx][sub_idx + 1]
        seg_vec_local = geo_utils.convert_gps2local(gps1, [gps2])[0]
        # TODO: scale conversions between transect and GPS frames
        seg_vec_local /= np.linalg.norm(seg_vec_local) + 1e-32
        seg_vec_local_per = seg_vec_local[::-1] * np.array([1, -1], dtype=np.float64)
        d_para_scaled = closest_par_dist / self.scale_factor
        d_perp_scaled = transect_pnt[1] / self.scale_factor
        pnt_local_scaled = d_para_scaled * seg_vec_local + d_perp_scaled * seg_vec_local_per
        return geo_utils.convert_local2gps(gps1, np.array([pnt_local_scaled], dtype=np.float64))[0]

    def get_search_polygon_gps(self, left_dist, right_dist):
        NUM_POINTS = 1000
        left_pnts = np.stack(
            [np.linspace(0, left_dist, NUM_POINTS, dtype=np.float64), -np.ones((NUM_POINTS,), dtype=np.float64)]).T
        right_pnts = np.stack(
            [np.linspace(right_dist, 0, NUM_POINTS, dtype=np.float64), np.ones((NUM_POINTS,), dtype=np.float64)]).T
        mid_points = np.array([[left_dist, 0], [right_dist, 0]], dtype=np.float64)
        transect_polygon = np.concatenate([left_pnts, mid_points, right_pnts], axis=0)
        search_poly_gps = [self.transect2gps(t_pnt) for t_pnt in transect_polygon]
        search_poly_gps = Polygon([res for res in search_poly_gps if res is not None])
        simplified_search_poly_gps = search_poly_gps.simplify(0.05 / 111e3, preserve_topology=True)
        return simplified_search_poly_gps

    def plot_transect_map(self):
        plt.figure()
        for seg_idx in range(len(self.transect_segs_wm)):
            plt.plot(self.transect_segs_wm[seg_idx], self.transect_segs[seg_idx], 'b-o')
        plt.plot((0, 50), (0, 50), 'k--')
        for gt_transect_d in self.transect_gt:
            plt.plot([0, 50], [gt_transect_d] * 2, 'r--')
        plt.title("Transect Map")
        plt.ylabel("Distance along transect [m]")
        plt.xlabel("Corrected world distance along transect [m]")
        plt.grid(True)

    def plot_gps_gt(self):
        plt.figure()
        # line_segs_local = geo_utils.convert_gps2local(transect_map.line_segs_gps[0][0], [0])
        for line_seg_gps in self.line_segs_gps:
            plt.plot(np.array(line_seg_gps)[:, 0], np.array(line_seg_gps)[:, 1])
        gps_gt_arr = np.array(self.gps_gt)
        plt.scatter(gps_gt_arr[:, 0], gps_gt_arr[:, 1], marker='x', c='r')
        plt.title("Transect Line GPS")
        plt.ylabel("lat")
        plt.xlabel("lon")


if __name__ == '__main__':

    transect_map = TransectMapper()
    transect_map.create_map_from_gpkg('test_transect_1.gpkg')

    transect_map.plot_transect_map()

    transect_map.plot_gps_gt()

    plt.figure()
    print("Testing transect point round trip conversion error...")
    N_TEST_PTS = 100
    test_trans_pts = np.array([50.0, 2.0]) * (np.array([0.0, -0.5]) + np.random.rand(N_TEST_PTS, 2))
    round_trip_errors = np.zeros_like(test_trans_pts)
    for i, transect_pnt in enumerate(test_trans_pts):
        gps_pnt = transect_map.transect2gps(transect_pnt)
        if gps_pnt is not None:
            trans_pnt2 = transect_map.gps2transect(gps_pnt)
            if trans_pnt2 is not None:
                round_trip_errors[i] = trans_pnt2 - transect_pnt
    plt.figure()
    plt.scatter(round_trip_errors[:, 0], round_trip_errors[:, 1])
    plt.title("Test round trip errors")
    plt.xlabel("Parallel Error [m]")
    plt.ylabel("Perpendicular Error [m]")

    print("Testing shapefile example performance...")
    print("[174.5306693  -35.84659462] and [174.53065988 -35.84661336] should produce not in transect error")
    test_pts_gdf = gpd.read_file('test_tpnts_1.gpkg')
    num_test_pts = len(test_pts_gdf)
    test_errors_2trans = np.zeros((num_test_pts, 2), dtype=np.float32)
    test_error_2gps = np.zeros((num_test_pts,), dtype=np.float32)
    t_pnts = []
    for i, row in test_pts_gdf.iterrows():
        geom = row.geometry
        label_l = row.NAME.split(',')
        assert isinstance(geom, Point) and len(label_l) == 2
        gps_pnt_gt = np.array(geom.coords, dtype=np.float64)[0, :2]
        trans_pnt_gt = np.array([float(v) for v in label_l], np.float64)
        t_pnts.append(trans_pnt_gt)
        gps_pnt_est = transect_map.transect2gps(trans_pnt_gt)
        print("GT GPS:  ", gps_pnt_gt)
        print("EST GPS: ", gps_pnt_est)
        trans_pnt_est = transect_map.gps2transect(gps_pnt_gt)
        print("GT transect pnt:  ", trans_pnt_gt)
        print("EST transect pnt: ", trans_pnt_est)
        print()
        if trans_pnt_est is not None:
            test_errors_2trans[i] = trans_pnt_gt - trans_pnt_est
        if gps_pnt_est is not None:
            test_error_2gps[i] = np.linalg.norm(geo_utils.convert_gps2local(gps_pnt_gt, [gps_pnt_est])[0])

    plt.figure()
    plt.scatter(test_errors_2trans[:, 0], test_errors_2trans[:, 1], c='r')
    for t_pnt, err in zip(t_pnts, list(test_errors_2trans)):
        plt.text(err[0], err[1], str(t_pnt))
    plt.title("Test shapefile error gps->transect")
    plt.xlabel("Parallel Error [m]")
    plt.ylabel("Perpendicular Error [m]")
    plt.figure()
    plt.scatter(np.arange(0, len(test_error_2gps)), test_error_2gps, c='b')
    for idx, (t_pnt, err) in enumerate(zip(t_pnts, list(test_error_2gps))):
        plt.text(idx, err, str(t_pnt))
    plt.title("Test shapefile error transect->gps distance")
    plt.ylabel("Distance Error [m]")
    plt.xlabel("idx")

    plt.show()

import numpy as np
import math
from utils import geo_utils
import geopandas as gpd
from shapely.geometry import *


TRANSECT_GTEXTP_RANGE = 12  # Distance over which propagated GT looses all it's confidence

METRES_PER_FOOT = 0.3048

# Things that will break this code
#  -> Lines without GT points on them

def lineseg_distance(seg, pnt):
    seg_vec = seg[1] - seg[0]
    seg_len = np.linalg.norm(seg_vec)
    pnt_vec = seg[0] - pnt
    prp_d = np.cross(seg_vec, pnt_vec) / seg_len
    pll_d1 = np.sqrt(np.linalg.norm(seg[0] - pnt) ** 2 - prp_d ** 2)
    pll_d2 = np.sqrt(np.linalg.norm(seg[1] - pnt) ** 2 - prp_d ** 2)
    on_seg = pll_d1 < seg_len and pll_d2 < seg_len
    return pll_d1 / seg_len if on_seg else -1, prp_d

def closest_lineseg_gps(linesegs_gps, pnt_gps):
    closest_tloc = (0, 1000)
    closest_idx = -1
    closest_sub_idx = 0
    tmp_closest_sub_idx = 0
    for seg_idx, seg_gps in enumerate(linesegs_gps):
        datum = np.array(seg_gps[0])
        local_pnt = geo_utils.convert_gps2local(datum, [pnt_gps])[0]
        local_segs = geo_utils.convert_gps2local(datum, np.array(seg_gps))
        if len(local_segs) < 2:
            raise "Line with less than 2 points!"
        elif len(local_segs) >= 2:
            closest_sub_tloc = (0.0, 1000.0)
            for sub_idx in range(len(local_segs) - 1):
                f_par, d_per = lineseg_distance(local_segs[sub_idx:sub_idx + 2], local_pnt)
                if f_par > 0 and d_per < closest_sub_tloc[1]:
                    tmp_closest_sub_idx = sub_idx
                    closest_sub_tloc = (f_par, d_per)
            f_par, d_per = closest_sub_tloc
        if 0.0 < f_par < 1.0 and d_per < closest_tloc[1]:
            closest_idx = seg_idx
            closest_sub_idx = tmp_closest_sub_idx
            closest_tloc = (f_par, d_per)
    return closest_idx, closest_sub_idx, closest_tloc


def closest_transect_vert(transect_segs, pnt):
    d_tran, d_perp = pnt
    closest_idx = -1
    closest_sub_idx = 0
    closest_distance = 1000
    for tseg_idx, tseg in enumerate(transect_segs):
        for sub_idx in range(len(tseg) - 1):
            dist = d_tran - tseg[sub_idx]
            if d_tran == 0.14:
                print(dist)
            if dist < 0:
                break
            if dist < closest_distance:
                closest_distance = dist
                closest_idx = tseg_idx
                closest_sub_idx = sub_idx
    return closest_idx, closest_sub_idx, closest_distance

class TransectMapper:
    def __init__(self):
        self.line_segs_gps = None
        self.gps_gt = []
        self.transect_gt = []
        self.transect_segs = None
        self.transect_segs_wm = None    # zero start with seg lengths defined by world length

    def create_map_from_gdf(self, transect_df):
        line_segs = []
        gt_pts = []
        gt_vals = []
        for i, row in transect_df.iterrows():
            geom = row.geometry
            if isinstance(geom, LineString):
                line_segs.append(np.array(geom.coords)[:, :2])
            elif isinstance(geom, Point):
                label = row.NAME
                valid_gtl = True
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
                gt_pts.append(np.array(geom.coords)[0, :2])
                gt_vals.append(gt_dist_m)
            else:
                raise "Geom type not supported!"
        self.create_map(line_segs, gt_pts, gt_vals)

    def create_map_from_gpkg(self, gpkg_path):
        transect_df = gpd.read_file(gpkg_path)
        self.create_map_from_gdf(transect_df)

    def create_map(self, line_segs_gps, gt_points, gt_td):
        # Create transect segments object
        self.line_segs_gps = [list(seg) for seg in line_segs_gps]
        transect_gts_idxval = [[] for seg in line_segs_gps]
        # Find closest line seg and set transect distance, create new vertex if not near existing, otherwise replace
        for gt_idx, (gt_gps_pnt, gt_td_par) in enumerate(zip(gt_points, gt_td)):
            idx, sub_idx, (f_par, d_per) = closest_lineseg_gps(self.line_segs_gps, gt_gps_pnt)
            if d_per > 0.05:
                print(f"Bad GT point IDX {gt_idx} - too far from line")
                continue
            tmp_sub_idx = sub_idx + int(f_par >= 0.01)
            if 0.01 < f_par < 0.99:
                self.line_segs_gps[idx].insert(tmp_sub_idx, gt_gps_pnt)
                # shift existing higher indexes along by one when inserting:
                new_transect_gts_idxval = []
                for sub_transect_gts_idxval in transect_gts_idxval:
                    new_transect_gts_idxval.append([])
                    for i, v in sub_transect_gts_idxval:
                        new_transect_gts_idxval[-1].append([i + (1 if i > sub_idx else 0), v])
                transect_gts_idxval = new_transect_gts_idxval
            transect_gts_idxval[idx].append([tmp_sub_idx, gt_td_par])
            self.gps_gt.append(gt_gps_pnt)
            self.transect_gt.append(gt_td_par)

        # print(transect_gts_idxval)

        self.transect_segs_wm = [[0.0]*len(seg) for seg in self.line_segs_gps]
        for seg_idx, seg in enumerate(self.transect_segs_wm):
            for sub_idx in range(len(seg)-1):
                dist = geo_utils.measure_chordlen(self.line_segs_gps[seg_idx][sub_idx],
                                                  self.line_segs_gps[seg_idx][sub_idx + 1])
                # print(self.line_segs_gps[seg_idx][sub_idx], self.line_segs_gps[seg_idx][sub_idx + 1], dist)
                self.transect_segs_wm[seg_idx][sub_idx + 1] = self.transect_segs_wm[seg_idx][sub_idx] + dist
            if seg_idx < len(self.transect_segs_wm) - 1:
                self.transect_segs_wm[seg_idx + 1][0] = self.transect_segs_wm[seg_idx][-1]

        # Calculate vertex transect distances
        self.transect_segs = []
        for seg_idx, seg in enumerate(self.line_segs_gps):
            seg_gt_pnts = transect_gts_idxval[seg_idx]
            # propagate transect distance to neighbouring verts for each GT with diminishing confidence, then combine
            # set valid flag at end of all subs
            if len(seg_gt_pnts) == 0:
                print(f"No GT points found on segment IDX {seg_idx}!!")
                continue
            gt_seg_arrays = np.zeros((len(seg_gt_pnts), len(seg), 2), dtype=np.float32)
            for i, (gt_idx, gt_val) in enumerate(seg_gt_pnts):
                gt_seg_arrays[i, gt_idx] = [gt_val, 1.0]
                for walk_idx in range(len(seg)):
                    for w_dir in [-1, 1]:
                        new_idx = gt_idx + w_dir * (walk_idx + 1)
                        if (w_dir == -1 and new_idx < 0) or (w_dir == 1 and new_idx >= len(seg)):
                            continue
                        dist = geo_utils.measure_chordlen(self.line_segs_gps[seg_idx][new_idx-w_dir],
                                                          self.line_segs_gps[seg_idx][new_idx])
                        base_dist, base_conf = gt_seg_arrays[i, new_idx-w_dir]
                        # TODO: Tune confidence
                        # confidence = 1 / (walk_idx + 2) ** 2
                        new_conf = base_conf * (1.0 - min(0.99, dist / TRANSECT_GTEXTP_RANGE))
                        gt_seg_arrays[i, new_idx] = [base_dist + w_dir * dist, new_conf]
            gt_seg_arrays[:, :, 0] *= gt_seg_arrays[:, :, 1]
            gt_seg_combined = np.sum(gt_seg_arrays, axis=0)
            estimated_transect_dists = gt_seg_combined[:, 0] / gt_seg_combined[:, 1]
            self.transect_segs.append(estimated_transect_dists)

    def gps2transect(self, gps_pnt):
        # lon, lat
        idx, sub_idx, (f_par, d_per_m) = closest_lineseg_gps(self.line_segs_gps, gps_pnt)
        if idx == -1:
            print(f"GPS point not in transect! {gps_pnt}")
            return None
        # print("gps2transect:", idx, sub_idx, f_par, d_per_m)
        td1 = self.transect_segs[idx][sub_idx]
        td2 = self.transect_segs[idx][sub_idx + 1]
        d_par_m = td1 + (td2 - td1) * f_par
        return d_par_m, d_per_m

    def transect2gps(self, transect_pnt):
        idx, sub_idx, closest_par_dist = closest_transect_vert(self.transect_segs, transect_pnt)
        if idx == -1:
            print(f"Transect point not in map! {transect_pnt}")
            return None
        # print("transect2gps:", idx, sub_idx, closest_par_dist)
        gps1 = self.line_segs_gps[idx][sub_idx]
        gps2 = self.line_segs_gps[idx][sub_idx + 1]
        transect_seg_len = self.transect_segs[idx][sub_idx + 1] - self.transect_segs[idx][sub_idx]
        seg_vec_local_par = geo_utils.convert_gps2local(gps1, [gps2])[0]
        gps_seg_length_m = np.linalg.norm(seg_vec_local_par) + 1e-32
        # if 100 * (transect_seg_len - gps_seg_length_m) / gps_seg_length_m > 1.0:
        #     print("Seg length differs by more than 1% between gps and transect:")
        #     print("gps length m", gps_seg_length_m)
        #     print("transect length m", transect_seg_len)
        seg_vec_local_par /= transect_seg_len  # gps_seg_length_m
        seg_vec_local_per = seg_vec_local_par[::-1] * np.array([1, -1])
        pnt_local = closest_par_dist * seg_vec_local_par + transect_pnt[1] * seg_vec_local_per
        return geo_utils.convert_local2gps(gps1, np.array([pnt_local]))[0]


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    transect_map = TransectMapper()
    transect_map.create_map_from_gpkg('test_transect_1.gpkg')

    for seg_idx in range(len(transect_map.transect_segs_wm)):
        plt.plot(transect_map.transect_segs_wm[seg_idx], transect_map.transect_segs[seg_idx], 'b-o')
    plt.plot((0, 50), (0, 50), 'k--')
    for gt_transect_d in transect_map.transect_gt:
        plt.plot([0, 50], [gt_transect_d]*2, 'r--')
    plt.title("Transect Map")
    plt.ylabel("Distance along transect [m]")
    plt.xlabel("Ground / world distance along transect [m]")
    plt.grid(True)

    plt.figure()
    # line_segs_local = geo_utils.convert_gps2local(transect_map.line_segs_gps[0][0], [0])
    for line_seg_gps in transect_map.line_segs_gps:
        plt.plot(np.array(line_seg_gps)[:, 0], np.array(line_seg_gps)[:, 1])
    gps_gt_arr = np.array(transect_map.gps_gt)
    plt.scatter(gps_gt_arr[:, 0], gps_gt_arr[:, 1], marker='x', c='r')
    plt.title("Transect Line GPS")
    plt.ylabel("lat")
    plt.xlabel("lon")

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
    for i, row in test_pts_gdf.iterrows():
        geom = row.geometry
        label_l = row.NAME.split(',')
        assert isinstance(geom, Point) and len(label_l) == 2
        gps_pnt_gt = np.array(geom.coords)[0, :2]
        trans_pnt_gt = np.array([float(v) for v in label_l])
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
    plt.title("Test shapefile error gps->transect")
    plt.xlabel("Parallel Error [m]")
    plt.ylabel("Perpendicular Error [m]")
    plt.figure()
    plt.scatter(np.arange(0, len(test_error_2gps)), test_error_2gps, c='b')
    plt.title("Test shapefile error transect->gps distance")
    plt.ylabel("Distance Error [m]")
    plt.xlabel("idx")


    plt.show()

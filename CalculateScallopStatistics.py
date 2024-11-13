import glob
import geopandas as gp
import pandas as pd
import shapely
from shapely.geometry import *
import numpy as np
from utils import geo_utils, vpz_utils, file_utils, tiff_utils
from utils import polygon_functions as spf
from utils.transect_mapper import transect_mapper
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import re

# Will break code:
# Overlapping polygons in the same class of either include OR exclude
# Tape reference must be part of vpz file not imported shape

SHAPE_CRS = "EPSG:4326"
MAG_DECLINATION = 19.3

SCALE_FACTOR = 1.025
SAVE_STATS_PLOTS = False
SHOW_STATS_PLOTS = False
SHOW_SHAPE_PLOTS = False

PROCESSED_BASEDIR = "/csse/research/CVlab/processed_bluerov_data/"
DONE_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_done.txt'
# DIRS_LIST = ['240713-134608', '240620-135134', '240714-113449', '240615-144558', '240617-080551', '240617-132136',
#              '240714-084519', '240713-104835']
# DIRS_LIST = ['240714-084519']
DIRS_LIST = []

PAIRED_SITE_ID_STRS = ['EX', 'MC', 'OP', 'UQ']

PARA_DIST_THRESH = 0.15
PERP_DIST_THRESH = 0.05
GPS_DIST_THRESHOLD_M = 0.03

YELLOWC = '\033[93m'
REDC = '\033[91m'
ENDC = '\033[0m'
BLUEC = '\033[94m'
CYANC = '\033[96m'
GREENC = '\033[92m'


def bin_widths_1_150_mm(widths):
    counts, bins = np.histogram(widths, bins=np.arange(start=1, stop=152))
    hist_dict = {}
    for bin in bins[:-1]:
        hist_dict[str(bin)] = counts[bin - 1]
    return hist_dict, counts, bins[:-1]


def append_to_csv(filepath, df):
    csv_exists = os.path.isfile(filepath)
    with open(filepath, 'a' if csv_exists else 'w') as f:
        df.to_csv(f, header=not csv_exists, index=False)


def process_dir(dir_name):
    dir_full = PROCESSED_BASEDIR + dir_name + '/'
    print(f"\n------------ Processing {dir_name} ------------")

    print("Initialising DEM Reader")
    dem_obj = tiff_utils.DEM(dir_full + 'geo_tiffs/')

    file_utils.ensure_dir_exists(dir_full + 'shapes_ann')

    # TODO: multiple shape files?? - yes but need to be careful not to double count scallops
    # Load scallop polygons
    scallop_gpkg_paths = glob.glob(dir_full + 'shapes_pred/*detections_filtered_3d.gpkg')
    scallop_shapes = {'detected': []}
    for spoly_path in scallop_gpkg_paths:
        spoly_gdf = gp.read_file(spoly_path)
        scallop_shapes["detected"].extend(list(spoly_gdf.geometry))

    # get include / exclude regions from viewer file
    exclude_polys = []
    include_polys = []
    ann_layer_keys = []
    transect_map = None
    vpz_diver_measurements_gps = None
    shape_layers_gpd = vpz_utils.get_shape_layers_gpd(dir_full, dir_name + '.vpz')
    for label, shape_layer in shape_layers_gpd:
        if label in ['Exclude Areas', 'Include Areas']:
            dst_list = exclude_polys if label == 'Exclude Areas' else include_polys
            for i, row in shape_layer.iterrows():
                if isinstance(row.geometry, Polygon):
                    dst_list.append(row.geometry)
                if isinstance(row.geometry, MultiPolygon):
                    dst_list.extend(row.geometry.geoms)
        if label == "Tape Reference" and len(shape_layer) > 1:
            transect_map = transect_mapper.TransectMapper()
            transect_map.create_map_from_gdf(shape_layer, plot=False)
            print(f"{CYANC}Tape reference found{ENDC}")
        if "polygon" in label.lower() and not 'first' in label.lower():
            # Human annotation(s)?
            scallop_shapes['UC_annotated'] = []
            if not label in ann_layer_keys:
                ann_layer_keys.append(label)
            for i, row in shape_layer.iterrows():
                if isinstance(row.geometry, Polygon):
                    scallop_shapes["UC_annotated"].append(row.geometry)
        if "live" in label.lower():
            scallop_shapes['NIWA_annotated'] = []
            for i, row in shape_layer.iterrows():
                if isinstance(row.geometry, LineString):
                    scallop_shapes["NIWA_annotated"].append(row.geometry)
        if "diver_measurements_all" in label.lower():
            vpz_diver_measurements_gps = []
            for i, row in shape_layer.iterrows():
                assert isinstance(row.geometry, Point)
                vpz_diver_measurements_gps.append([np.array(row.geometry.coords.xy), row.NAME])
    # print(f"VPZ polygon annotation layers: {ann_layer_keys}")
    # assert len(ann_layer_keys) == 1

    # Get site metadata
    if os.path.isfile(dir_full + "scan_metadata.json"):
        with open(dir_full + "scan_metadata.json", 'r') as meta_doc:
            metadata = json.load(meta_doc)
    else:
        raise Exception(f"Site {dir_name} has no JSON metadata!")

    # Get site name and standardize
    site_name = metadata['NAME']
    print("Site name:", site_name)
    site_name_nums = re.findall(r'\d+', site_name)
    if len(site_name_nums) == 2 and site_name_nums[1] == '1':
        site_id_num = site_name_nums[0]
    else:
        site_id_num = '.'.join(site_name_nums)
    site_id_str = None
    for str_tmp in PAIRED_SITE_ID_STRS:
        if str_tmp.lower() in site_name.lower():
            site_id_str = str_tmp
            break
    if transect_map is not None and site_id_str is None:
        raise Exception(f"{REDC}Site {site_name} has transect map but doesnt match any known paired sites!{ENDC}")
    if site_id_str is None:
        site_id = site_name.split(' ')[-1]
    else:
        site_id = site_id_str + ' ' + site_id_num
    if site_id == 'EX 16':
        site_id = 'EX 13'
        site_name = site_id
    if dir_name == '240714-113449':
        site_id = 'UQ 18'
        site_name = site_id
    print("Site ID:", site_id)

    df_row_shared = {'site ID': [site_id],
                     'site name': [site_name],
                     'date-time site': [dir_name],
                     'date-time proc': [datetime.now().strftime("%y%m%d-%H%M%S")],
                     'longitude': [metadata['lonlat'][0]],
                     'latitude': [metadata['lonlat'][1]],}

    rov_mag_heading = metadata['M.Heading'] if 'M.Heading' in metadata else metadata['T.Heading'] - MAG_DECLINATION

    # TODO: check for overlap interclass and trim - cant handle intraclass overlap

    # Calculate total valid survey area
    total_inc_area = 0.0
    for inc_poly in include_polys:
        inc_area = spf.get_poly_area_m2(inc_poly)
        inc_area_3d = dem_obj.poly3d_from_dem(np.array(inc_poly.exterior.coords))
        inc_area_3d = inc_area_3d[np.abs(inc_area_3d[:, 2]) < 300]
        inc_area_3d_local = geo_utils.convert_gps2local(inc_area_3d[0], inc_area_3d)
        normal_vec = spf.get_avg_plane_normal(inc_area_3d_local)
        area_multiplier = 1 / np.dot(normal_vec, np.array([0, 0, 1], dtype=np.float64))
        norm_z_angle = np.degrees(np.arccos(1 / area_multiplier))
        # print(f"Inc area angle from z: {round(norm_z_angle, 2)}, area multiplier = {area_multiplier}")
        if area_multiplier > 1.05:
            # print(f"{REDC} !!!!!!!!!!!!!!!!!!!!!!! Some metashape bullshit occurring !!!!!!!!!!!!!!!!!!!!!!! {ENDC}")
            area_multiplier = min(1.2, area_multiplier)
            # ax = plt.axes(projection='3d')
            # ax.plot3D(inc_area_3d_local[:, 0], inc_area_3d_local[:, 1], inc_area_3d_local[:, 2])
            # vec_start = inc_area_3d_local.mean(axis=0)
            # vec_end = vec_start + 20 * normal_vec
            # arrow = np.stack([vec_start, vec_end])
            # ax.plot3D(arrow[:, 0], arrow[:, 1], arrow[:, 2], 'r')
            # BOX_MINMAX = [-50, 50]
            # ax.auto_scale_xyz(BOX_MINMAX, BOX_MINMAX, BOX_MINMAX)
            # plt.show()

        exc_area = 0.0
        for exc_poly in exclude_polys:
            if inc_poly.intersects(exc_poly):
                inters_poly = inc_poly.intersection(exc_poly)
                if isinstance(inters_poly, MultiPolygon):
                    for i_poly in inters_poly.geoms:
                        exc_area += spf.get_poly_area_m2(i_poly)
                else:
                    exc_area += spf.get_poly_area_m2(inters_poly)
        total_inc_area += area_multiplier * (inc_area - exc_area)
    site_area = round(total_inc_area * SCALE_FACTOR**2, 2)
    print(f"VALID ROV search area = {site_area}m2")
    if site_area < 10.0:
        raise Exception("Site has no valid search area!")

    def check_inbounds(pnt, inc_polys, exc_polys):
        valid = False
        for bound_polys, keep in [[inc_polys, True], [exc_polys, False]]:
            for b_poly in bound_polys:
                if b_poly.contains(pnt):
                    valid = keep
                    break
        return valid

    # Filter scallops in valid survey area(s)
    valid_scallop_shapes = {k: [] for k in scallop_shapes.keys()}
    for key, shapes in scallop_shapes.items():
        # print(f"Total number of ROV {key} scallops = {len(shapes)}")
        for s_shp in shapes:
            # Check if scallop polygon center is in ANY include area and out of ALL exclude areas
            if check_inbounds(s_shp.centroid, include_polys, exclude_polys):
                valid_scallop_shapes[key].append(s_shp)
        # print(f"Number of valid ROV {key} scallops = {len(valid_scallop_shapes[key])}")

    # Calculate valid scallop polygon widths (annotations and detections)
    scallop_stats = {k: {'lat': [], 'lon': [], 'width_mm': [], 'shape': []} for k in valid_scallop_shapes}
    for key, valid_shapes in valid_scallop_shapes.items():
        width_linestrings_gps = []
        for v_shp in valid_shapes:
            if isinstance(v_shp, LineString):
                assert key == 'NIWA_annotated'
                line = np.array(v_shp.coords, dtype=np.float64)[:, :2]
                max_width = np.linalg.norm(geo_utils.convert_gps2local(line[0], line[1][None]))
                lon, lat = np.mean(line, axis=0)
            else:
                poly_2d = np.array(v_shp.exterior.coords)[:, :2]
                poly_3d = np.array(v_shp.exterior.coords) if v_shp.has_z else dem_obj.poly3d_from_dem(poly_2d)
                local_poly_3d = spf.get_local_poly_arr_3D(poly_3d)

                # TODO: improve sizing (shape fitting?) - PCA??
                # naive_max_w:
                scallop_vert_mat = np.repeat(local_poly_3d[None], local_poly_3d.shape[0], axis=0)
                scallop_vert_dists = np.linalg.norm(scallop_vert_mat - scallop_vert_mat.transpose([1, 0, 2]), axis=2)
                max_dist_idx = np.argmax(scallop_vert_dists)
                vert_idxs = max_dist_idx // scallop_vert_dists.shape[0], max_dist_idx % scallop_vert_dists.shape[0]
                max_width = min(0.2, np.max(scallop_vert_dists))

                width_line_local_2d = local_poly_3d[vert_idxs, 0:2]
                width_line_gps = geo_utils.convert_local2gps(poly_2d[0], width_line_local_2d)
                width_linestrings_gps.append(LineString(width_line_gps))

                poly_arr = spf.get_poly_arr_2d(v_shp)
                lon, lat = np.mean(poly_arr, axis=0)

                if SHOW_SHAPE_PLOTS:
                    ax = plt.axes(projection='3d')
                    ax.plot3D(local_poly_3d[:, 0], local_poly_3d[:, 1], local_poly_3d[:, 2])
                    ax.plot3D(local_poly_3d[vert_idxs, 0], local_poly_3d[vert_idxs, 1], local_poly_3d[vert_idxs, 2])
                    BOX_MINMAX = [-0.06, 0.06]
                    ax.auto_scale_xyz(BOX_MINMAX, BOX_MINMAX, BOX_MINMAX)
                    plt.show()

            scallop_stats[key]['width_mm'].append(round(SCALE_FACTOR * max_width * 1000))
            scallop_stats[key]['lat'].append(lat)
            scallop_stats[key]['lon'].append(lon)
            scallop_stats[key]['shape'].append(v_shp)

        if len(width_linestrings_gps):
            labels = [str(w) + ' mm' for w in scallop_stats[key]['width_mm']]
            width_lines_gdf = gp.GeoDataFrame({'NAME': labels, 'geometry': width_linestrings_gps}, geometry='geometry')
            width_lines_gdf.set_crs(SHAPE_CRS, inplace=True)
            width_lines_gdf.to_file(dir_full + f"shapes_ann/ALL_width_lines_{key}.geojson", driver='GeoJSON')

        # CSV with every scallop detection
        # site_dataframe = pd.DataFrame(scallop_stats)
        # with open(dir_full + 'valid_scallop_sizes.csv', 'w') as f:
        #     site_dataframe.to_csv(f, header=True, index=False)

        # If not test site, add row in rov detection / annotation stats csv for site
        if transect_map is None:
            df_row_rov = {'area m2': [site_area],
                          'count': [len(scallop_stats[key]['width_mm'])],
                          'depth': [metadata['Depth']],
                          'altitude': [metadata['Altitude']],
                          'm.bearing': [rov_mag_heading]}
            rov_meas_bins_dict, counts, bins = bin_widths_1_150_mm(scallop_stats[key]['width_mm'])
            df_row_rov.update(rov_meas_bins_dict)
            df_row = dict(df_row_shared, **df_row_rov)
            append_to_csv(PROCESSED_BASEDIR + f"scallop_rov_{key}_stats.csv", pd.DataFrame(df_row))
            if SHOW_STATS_PLOTS or SAVE_STATS_PLOTS:
                plt.figure()
                plt.bar(bins, counts)
                plt.title(f"ROV {key} scallop size distribution")
                plt.xlabel("Width [mm]")
                plt.ylabel("Count")

    # If paired site, read from diver data and process
    if transect_map:
        # Get relevant diver data from provided xlsx IF diver anns layer isnt in VPZ file
        # TODO: read diver measurements from shape layer instead of xlsx, make sure metadata is in layer too
        diver_data_xls = pd.ExcelFile(PROCESSED_BASEDIR + 'ruk2401_dive_slate_data_entry Kura Reihana.xlsx')
        survey_meas_df = pd.read_excel(diver_data_xls, 'scallop_data')
        survey_metadata_df = pd.read_excel(diver_data_xls, 'metadata')
        site_meas_df = survey_meas_df.loc[survey_meas_df['site'] == site_id]
        site_metadata_df = survey_metadata_df.loc[survey_metadata_df['site'] == site_id]
        print("Site Habitat:", site_metadata_df['habitat'].values[0])

        # Add row in diver stats csv for paired site
        search_distance_left = site_metadata_df.loc[site_metadata_df['diver'].str.contains('Left')]['distance'].values[0]
        search_distance_right = site_metadata_df.loc[site_metadata_df['diver'].str.contains('Right')]['distance'].values[0]
        total_diver_area = 1.0 * (search_distance_left + search_distance_right)
        diver_depth = round(np.mean([np.mean(site_metadata_df[k]) for k in ['depth_s', 'depth_f']]), 2)
        diver_bearing = round(np.mean(site_metadata_df['bearing']))
        print(f"Diver TOTAL search area for {site_id} = {total_diver_area}m2")
        search_poly_gps = transect_map.get_search_polygon_gps(search_distance_left, search_distance_right)
        valid_search_polys = []
        for inc_poly in include_polys:
            if inc_poly.intersects(search_poly_gps):
                intersection_poly = inc_poly.intersection(search_poly_gps)
                if isinstance(intersection_poly, Polygon):
                    valid_search_polys.append(intersection_poly)
                else:
                    valid_int_polys = [geom for geom in intersection_poly.geoms if isinstance(geom, Polygon)]
                    valid_search_polys.extend(valid_int_polys)
        inbounds_diver_area = 0
        for search_poly in valid_search_polys:
            inbounds_diver_area += spf.get_poly_area_m2(search_poly)
            for exc_poly in exclude_polys:
                if exc_poly.intersects(search_poly):
                    inbounds_diver_area -= spf.get_poly_area_m2(exc_poly.intersection(search_poly))
        inbounds_diver_area = round(inbounds_diver_area * SCALE_FACTOR**2, 2)
        # print(f"Diver VALID search area for {site_id} = {inbounds_diver_area}m2")
        if len(valid_search_polys):
            search_gdf = gp.GeoDataFrame({'NAME': 'valid_search_area', 'geometry': valid_search_polys}, geometry='geometry')
            search_gdf.to_file(dir_full + 'shapes_ann/valid_search_area.geojson', driver='GeoJSON')

        diver_points_gps = []
        diver_entries_valid = []
        diver_entries = []
        diver_point_tags = []
        for i, csv_row in site_meas_df.iterrows():
            t_para, t_perp = csv_row['y_m'], csv_row["x_cm"]
            diver_initials = csv_row['diver'].split(' ')[0]
            left_side = 'Left' in str(csv_row['diver'])
            t_perp = (-1 if left_side else 1) * t_perp / 100
            meas_width_mm = csv_row["SCA_mm"]
            gps_coord = transect_map.transect2gps([t_para, t_perp])
            if gps_coord is None:
                # Cant find diver measurement in transect, place outside bounds for later
                gps_coord = metadata['lonlat'] + (0.5 - np.random.random((2,))) / 111e3
            diver_entries.append([t_para, t_perp, meas_width_mm])
            diver_entries_valid.append(check_inbounds(Point(gps_coord), valid_search_polys, exclude_polys))
            diver_points_gps.append(Point(gps_coord))
            diver_point_tags.append(diver_initials + ' ' + str(meas_width_mm) + ' mm')
        diver_entries_arr = np.array(diver_entries)
        total_diver_count = diver_entries_arr.shape[0]

        if len(diver_points_gps):
            diver_meas_gdf = gp.GeoDataFrame({'NAME': diver_point_tags, 'geometry': diver_points_gps,
                                              'Dist along T': diver_entries_arr[:, 0],
                                              'Dist to T': diver_entries_arr[:, 1]}, geometry='geometry')
            diver_meas_gdf.set_crs(SHAPE_CRS, inplace=True)
            diver_meas_gdf.to_file(dir_full + 'shapes_ann/Diver_measurements_ALL.geojson', driver='GeoJSON')


        # Take only inbound and on-transect diver measurements
        diver_entries_arr = diver_entries_arr[diver_entries_valid]
        diver_points_gps = [p for i, p in enumerate(diver_points_gps) if diver_entries_valid[i]]
        diver_point_tags = [t for i, t in enumerate(diver_point_tags) if diver_entries_valid[i]]
        inbound_diver_count = diver_entries_arr.shape[0]
        diver_widths_valid = diver_entries_arr[:, 2]

        # if len(diver_points_gps):
        #     diver_meas_gdf = gp.GeoDataFrame({'NAME': diver_point_tags, 'geometry': diver_points_gps,
        #                                       'Dist along T': diver_entries_arr[:, 0],
        #                                       'Dist to T': diver_entries_arr[:, 1]}, geometry='geometry')
        #     diver_meas_gdf.set_crs(SHAPE_CRS, inplace=True)
        #     diver_meas_gdf.to_file(dir_full + 'shapes_ann/Diver_measurements_in_search_area.geojson', driver='GeoJSON')

        if vpz_diver_measurements_gps is not None:
            total_diver_count = len(vpz_diver_measurements_gps)
            vpz_diver_measurements_gps = [[pnt, tag] for pnt, tag in vpz_diver_measurements_gps if check_inbounds(Point(pnt), include_polys, exclude_polys)]
            inbound_diver_count = len(vpz_diver_measurements_gps)
            diver_widths_valid = [int(tag.split(' ')[1]) for pnt, tag in vpz_diver_measurements_gps]
        # print(f"Diver TOTAL scallop count = {total_diver_count}")
        print(f"{GREENC}Diver VALID search area scallop count = {inbound_diver_count}{ENDC}")

        if len(diver_widths_valid):
            # TODO: needs to be in separate file for each site or not?
            df_row = {'site id': [site_id] * len(diver_widths_valid),
                      'match id': list(range(len(diver_widths_valid))),
                      'width mm': diver_widths_valid}
            append_to_csv(PROCESSED_BASEDIR + 'individual_diver_measurements.csv', pd.DataFrame(df_row))

        df_row_dive = {'area m2': [inbounds_diver_area],
                       'count': [inbound_diver_count],
                       'depth': [diver_depth],
                       'altitude': [0],
                       'm.bearing': [diver_bearing],}
        rov_meas_bins_dict, counts, bins = bin_widths_1_150_mm(diver_widths_valid)
        df_row_dive.update(rov_meas_bins_dict)
        df_row = dict(df_row_shared, **df_row_dive)
        append_to_csv(PROCESSED_BASEDIR + 'scallop_dive_stats.csv', pd.DataFrame(df_row))
        if SHOW_STATS_PLOTS or SAVE_STATS_PLOTS:
            plt.figure()
            plt.bar(bins, counts)
            plt.title(f"Diver VALID measured scallop size distribution")
            plt.xlabel("Width [mm]")
            plt.ylabel("Count")

        print("Matching with diver measurements...")
        matched_scallop_widths = {k: [] for k in scallop_stats.keys()}
        for key, stats in scallop_stats.items():
            in_search_area = []
            false_positive_num = 0
            # if key != 'detected':
            #     continue
            for lon, lat, width_rov in zip(stats['lon'], stats['lat'], stats['width_mm']):
                is_inbounds = check_inbounds(Point(lon, lat), valid_search_polys, exclude_polys)
                in_search_area.append(is_inbounds)
                if vpz_diver_measurements_gps is None:
                    res = transect_map.gps2transect((lon, lat))
                    if res is None:
                        continue
                    t_para, t_perp = res
                    near_para = np.abs(diver_entries_arr[:, 0] - t_para) < PARA_DIST_THRESH
                    near_perp = np.abs(diver_entries_arr[:, 1] - t_perp) < PERP_DIST_THRESH
                    scallop_near = near_para * near_perp
                else:
                    gps_dist = np.array([geo_utils.measure_chordlen([lon, lat], diver_pnt) for diver_pnt, tag in vpz_diver_measurements_gps]) * SCALE_FACTOR
                    scallop_near = gps_dist < GPS_DIST_THRESHOLD_M

                num_matches = np.sum(scallop_near)
                if num_matches == 0:
                    false_positive_num += int(is_inbounds)
                    continue
                if num_matches > 1:
                    false_positive_num += int(is_inbounds)
                    print("Multiple matches!")
                    # TODO: deal with multiple (detected / annotated) <-> diver matches
                    continue

                if vpz_diver_measurements_gps is None:
                    diver_width_mm = diver_widths_valid[scallop_near][0]
                else:
                    diver_width_mm = diver_widths_valid[np.where(scallop_near)[0][0]]
                diver_meas_idx = np.where(scallop_near)[0][0]
                matched_scallop_widths[key].append([width_rov, diver_width_mm, diver_meas_idx])

            rov_in_search_area_widths = [w for i, w in enumerate(stats['width_mm']) if in_search_area[i]]
            rov_in_search_tags = [f"{key}_{str(w)}mm" for w in rov_in_search_area_widths]
            rov_in_search_area_shapes = [s for i, s in enumerate(stats['shape']) if in_search_area[i]]
            if len(rov_in_search_area_shapes):
                shapes_gdf = gp.GeoDataFrame({'NAME': rov_in_search_tags, 'geometry': rov_in_search_area_shapes},
                                             geometry='geometry')
                shapes_gdf.to_file(dir_full + f"shapes_ann/{key}_shapes_in_search_area.geojson", driver='GeoJSON')
            num_scallops_in_search = len(rov_in_search_area_widths)
            print(f"ROV {key} in search area scallop count = {num_scallops_in_search}")
            if num_scallops_in_search:
                area_sizing_error = np.mean(rov_in_search_area_widths) - np.mean(diver_widths_valid)
                print(f"{REDC}ROV {key} in search area mean width error = {round(area_sizing_error, 2)}mm{ENDC}")
            # Add row in rov detection / annotation stats csv for site
            df_row_rov = {'area m2': [inbounds_diver_area],
                          'count': [len(rov_in_search_area_widths)],
                          'depth': [metadata['Depth']],
                          'altitude': [metadata['Altitude']],
                          'm.bearing': [rov_mag_heading]}
            rov_meas_bins_dict, counts, bins = bin_widths_1_150_mm(rov_in_search_area_widths)
            df_row_rov.update(rov_meas_bins_dict)
            df_row = dict(df_row_shared, **df_row_rov)
            append_to_csv(PROCESSED_BASEDIR + f"scallop_rov_{key}_stats.csv", pd.DataFrame(df_row))
            if SHOW_STATS_PLOTS or SAVE_STATS_PLOTS:
                plt.figure()
                plt.bar(bins, counts)
                plt.title(f"{key} within search scallop size distribution")
                plt.xlabel("Width [mm]")
                plt.ylabel("Count")

            matched_arr = np.array(matched_scallop_widths[key]).T
            if len(matched_arr):
                if key == "detected":
                    diver_match_idxs = matched_arr[2]
                    diver_widths_mm = matched_arr[1]
                    df_row = {'site id': [site_id] * len(diver_match_idxs),
                              'match id': list(diver_match_idxs),
                              'width mm': list(diver_widths_mm)}
                    append_to_csv(PROCESSED_BASEDIR + 'individual_cnn_measurements.csv', pd.DataFrame(df_row))

                    # TODO: append to CNN measurement csv

                matched_error = matched_arr[0] - matched_arr[1]
                rov_count_eff_matched = matched_arr.shape[1] / inbound_diver_count
                print(f"ROV {key} matched count efficacy = {round(rov_count_eff_matched * 100)}%")
                print(f"ROV {key} in search area false positive = {false_positive_num}")
                print(f"ROV {key} in search area false negative = {inbound_diver_count - matched_arr.shape[1]}")
                print(f"ROV {key} matched sizing error STD = {round(np.std(matched_error), 2)}mm")
                mean_error = np.mean(matched_error)
                print(f"ROV {key} matched sizing bias = {round(mean_error, 2)}mm")

                if SHOW_STATS_PLOTS or SAVE_STATS_PLOTS:
                    plt.figure()
                    plt.hist(matched_error, bins=50)
                    plt.title(f"{key} matched sizing error [mm]")
                    plt.xlabel("Error [mm]")
                    plt.ylabel("Count")

        # TODO: NIWA - Need err / bias for scallop detection efficiency and sizing, by detected size category
        # TODO: NIWA - Per (ROV detected) size class count multiplier
        # TODO: NIWA - Per (ROV detected) size class sizing bias

    if SAVE_STATS_PLOTS:
        file_utils.ensure_dir_exists(dir_full + "statistics_plots")
        for i in plt.get_fignums():
            fig = plt.figure(i)
            plt.savefig(dir_full + f"statistics_plots/{fig.axes[0].get_title()}.png")
        file_utils.SetFolderPermissions(dir_full + "statistics_plots/")
    if SHOW_STATS_PLOTS:
        plt.show()

    file_utils.SetFolderPermissions(dir_full + "shapes_ann/")


if __name__ == "__main__":
    if len(DIRS_LIST):
        dirs_list = DIRS_LIST
    else:
        with open(DONE_DIRS_FILE, 'r') as f:
            dirs_list = f.readlines()
    # dirs_list = dirs_list[132:]
    # dirs_list = ['240713-134608\n']
    for dir_entry in dirs_list:
        if len(dir_entry) == 1 or '#' in dir_entry:
            continue
        if 'STOP' in dir_entry:
            break
        dir_name = dir_entry[:13]
        # process_dir(dir_name)
        try:
            process_dir(dir_name)
        except Exception as e:
            print(e)

       #  break

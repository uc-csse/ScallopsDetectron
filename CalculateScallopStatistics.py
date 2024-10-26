import glob
import geopandas as gp
import pandas as pd
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

PROCESSED_BASEDIR = "/csse/research/CVlab/processed_bluerov_data/"
DONE_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_done.txt'
DIRS_LIST = ['240713-134608', '240620-135134', '240714-113449', '240615-144558', '240617-080551', '240617-132136',
             '240714-084519', '240713-104835']
# DIRS_LIST = ['240714-084519']

PAIRED_SITE_ID_STRS = ['EX', 'MC', 'OP', 'UQ']

PARA_DIST_THRESH = 0.2
PERP_DIST_THRESH = 0.1

OUTPUT_SHAPE_LAYERS = True
ONLY_IN_DIVER_SEARCH_AREA = True

def get_poly_arr_2d(poly):
    return np.array(poly.exterior.coords)[:, :2]

def get_local_poly_arr(poly_gps):
    poly_arr_2d = get_poly_arr_2d(poly_gps)
    poly_arr_m = geo_utils.convert_gps2local(poly_arr_2d[0], poly_arr_2d)
    return poly_arr_m

def get_local_poly_arr_3D(poly_gps):
    poly_arr_m = geo_utils.convert_gps2local(poly_gps[0], poly_gps)
    return poly_arr_m

def get_poly_area_m2(poly_gps):
    return Polygon(get_local_poly_arr(poly_gps)).area

def bin_widths_1_150_mm(widths):
    counts, bins = np.histogram(widths, bins=np.arange(start=1, stop=152))
    # plt.bar(bins[:-1], counts)
    # plt.show()
    hist_dict = {}
    for bin in bins[:-1]:
        hist_dict[str(bin)] = counts[bin - 1]
    return hist_dict

def append_to_csv(filepath, df):
    csv_exists = os.path.isfile(filepath)
    with open(filepath, 'a' if csv_exists else 'w') as f:
        df.to_csv(f, header=not csv_exists, index=False)


def process_dir(dir_name):
    dir_full = PROCESSED_BASEDIR + dir_name + '/'

    print("Initialising DEM Reader")
    dem_obj = tiff_utils.DEM(dir_full + 'geo_tiffs/')

    # TODO: multiple shape files - yes but need to be careful not to double count scallops
    # Load scallop polygons
    scallop_gpkg_paths = glob.glob(dir_full + '*detections_Filtered*.gpkg')
    scallop_polygons = {'detected': [], 'UC_annotated': [], 'NIWA_annotated': []}
    for spoly_path in scallop_gpkg_paths:
        spoly_gdf = gp.read_file(spoly_path)
        scallop_polygons["detected"].extend(list(spoly_gdf.geometry))

    # get include / exclude regions from viewer file
    exclude_polys = []
    include_polys = []
    ann_layer_keys = []
    transect_map = None
    shape_layers_gpd = vpz_utils.get_shape_layers_gpd(dir_full, dir_name + '.vpz')
    for label, shape_layer in shape_layers_gpd:
        if label in ['Exclude Areas', 'Include Areas']:
            dst_list = exclude_polys if label == 'Exclude Areas' else include_polys
            for i, row in shape_layer.iterrows():
                if isinstance(row.geometry, Polygon):
                    dst_list.append(row.geometry)
                if isinstance(row.geometry, MultiPolygon):
                    dst_list.extend(row.geometry.geoms)
        if label == "Tape Reference":
            transect_map = transect_mapper.TransectMapper()
            transect_map.create_map_from_gdf(shape_layer)
            print("Tape reference found")
        if "polygon" in label.lower() and not 'first' in label.lower():
            # Human annotation(s)?
            if not label in ann_layer_keys:
                ann_layer_keys.append(label)
            for i, row in shape_layer.iterrows():
                if isinstance(row.geometry, Polygon):
                    scallop_polygons["UC_annotated"].append(row.geometry)
        if "live" in label.lower():
            for i, row in shape_layer.iterrows():
                if isinstance(row.geometry, LineString):
                    scallop_polygons["NIWA_annotated"].append(row.geometry)
    print(f"VPZ polygon annotation layers: {ann_layer_keys}")
    assert len(ann_layer_keys) == 1

    if os.path.isfile(dir_full + "scan_metadata.json"):
        with open(dir_full + "scan_metadata.json", 'r') as meta_doc:
            metadata = json.load(meta_doc)
    else:
        raise Exception(f"Site {dir_name} has no JSON metadata!")

    # Get site name and standardize
    site_name = metadata['NAME']
    site_id_num = re.findall(r'\b\d+\b', site_name)[0]
    site_id_str = None
    for str_tmp in PAIRED_SITE_ID_STRS:
        if str_tmp.lower() in site_name.lower():
            site_id_str = str_tmp
            break
    if site_id_str is None:
        raise Exception(f"site {site_name} doesnt match any known paired sites")
    site_id = site_id_str + ' ' + site_id_num
    if site_id == 'EX 16':
        site_id = 'EX 13'
        site_name = site_id
    if dir_name == '240714-113449':
        site_id = 'UQ 18'
        site_name = site_id
    print("Site ID:", site_id)
    print("folder name:", dir_name)

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
        inc_area = get_poly_area_m2(inc_poly)
        exc_area = 0.0
        for exc_poly in exclude_polys:
            if inc_poly.intersects(exc_poly):
                inters_poly = inc_poly.intersection(exc_poly)
                exc_area += get_poly_area_m2(inters_poly)
        total_inc_area += inc_area - exc_area
    site_area = round(total_inc_area, 2)
    print(f"ROV search area = {site_area} m2")

    def check_inbounds(pnt):
        valid = False
        for bound_polys, keep in [[include_polys, True], [exclude_polys, False]]:
            for b_poly in bound_polys:
                if b_poly.contains(pnt):
                    valid = keep
                    break
        return valid

    # Filter scallops in valid survey area(s)
    valid_scallop_polygons = {k: [] for k in scallop_polygons.keys()}
    for key, polygons in scallop_polygons.items():
        print(f"Total number of {key} scallops = {len(polygons)}")
        for spoly in polygons:
            # Check if scallop polygon center is in ANY include area and out of ALL exclude areas
            if check_inbounds(spoly.centroid):
                valid_scallop_polygons[key].append(spoly)
        print(f"Number of valid {key} scallops = {len(valid_scallop_polygons[key])}")

    # test_write_gdf = gp.GeoDataFrame({'NAME': 'test', 'geometry': valid_scallop_polygons}, geometry='geometry')
    # test_write_gdf.to_file('/csse/users/tkr25/Desktop/valid_scallops.geojson', driver='GeoJSON')

    # Calculate valid scallop polygon widths (annotations and detections)
    scallop_stats = {k: {'lat': [], 'lon': [], 'width_mm': []} for k in valid_scallop_polygons}
    for key, valid_polygons in valid_scallop_polygons.items():
        width_linestrings_gps = []
        for vspoly in valid_polygons:
            if isinstance(vspoly, LineString):
                assert key == 'NIWA_annotated'
                line = np.array(vspoly.coords, dtype=np.float64)[:, :2]
                max_width = np.linalg.norm(geo_utils.convert_gps2local(line[0], line[1][None]))
                lon, lat = np.mean(line, axis=0)
            else:
                poly_2d = np.array(vspoly.exterior.coords)[:, :2]
                poly_3d = dem_obj.poly3d_from_dem(poly_2d)
                local_poly_3d = get_local_poly_arr_3D(poly_3d)

                mean_3d = local_poly_3d.mean(axis=0)
                z_dist = np.abs(local_poly_3d[:, 2] - mean_3d[2])
                local_poly_3d[:, 2][z_dist > 0.01] = mean_3d[2]

                # eig_vecs, eig_vals, center_pnt = spf.pca(local_poly_3d)
                # local_poly_3d = np.matmul(np.linalg.inv(eig_vecs), (local_poly_3d - center_pnt).T).T
                # local_poly_3d[:, 2] *= 0

                # TODO: improve sizing (shape fitting?) - PCA??
                # naive_max_w:
                scallop_vert_mat = np.repeat(local_poly_3d[None], local_poly_3d.shape[0], axis=0)
                scallop_vert_dists = np.linalg.norm(scallop_vert_mat - scallop_vert_mat.transpose([1, 0, 2]), axis=2)
                max_dist_idx = np.argmax(scallop_vert_dists)
                vert_idxs = max_dist_idx // scallop_vert_dists.shape[0], max_dist_idx % scallop_vert_dists.shape[0]
                max_width = np.max(scallop_vert_dists)

                width_line_local_2d = local_poly_3d[vert_idxs, 0:2]
                width_line_gps = geo_utils.convert_local2gps(poly_2d[0], width_line_local_2d)
                width_linestrings_gps.append(LineString(width_line_gps))

                poly_arr = get_poly_arr_2d(vspoly)
                lon, lat = np.mean(poly_arr, axis=0)

                # ax = plt.axes(projection='3d')
                # ax.plot3D(local_poly_3d[:, 0], local_poly_3d[:, 1], local_poly_3d[:, 2])
                # ax.plot3D(local_poly_3d[vert_idxs, 0], local_poly_3d[vert_idxs, 1], local_poly_3d[vert_idxs, 2])
                # BOX_MINMAX = [-0.06, 0.06]
                # ax.auto_scale_xyz(BOX_MINMAX, BOX_MINMAX, BOX_MINMAX)
                # plt.show()

            scallop_stats[key]['width_mm'].append(round(max_width * 1000))
            scallop_stats[key]['lat'].append(lat)
            scallop_stats[key]['lon'].append(lon)

        if OUTPUT_SHAPE_LAYERS and len(width_linestrings_gps):
            labels = [str(w) + ' mm' for w in scallop_stats[key]['width_mm']]
            width_lines_gdf = gp.GeoDataFrame({'NAME': labels, 'geometry': width_linestrings_gps}, geometry='geometry')
            width_lines_gdf.set_crs(SHAPE_CRS, inplace=True)
            file_utils.ensure_dir_exists(dir_full + 'shapes_ann')
            width_lines_gdf.to_file(dir_full + f"shapes_ann/{'width lines ' + key}.geojson", driver='GeoJSON')

        # CSV with every scallop detection
        # site_dataframe = pd.DataFrame(scallop_stats)
        # with open(dir_full + 'valid_scallop_sizes.csv', 'w') as f:
        #     site_dataframe.to_csv(f, header=True, index=False)

        # Add row in rov detection / annotation stats csv for site
        if transect_map is None or not ONLY_IN_DIVER_SEARCH_AREA:
            df_row_rov = {'area m2': [site_area],
                          'count': [len(scallop_stats[key]['width_mm'])],
                          'depth': [metadata['Depth']],
                          'altitude': [metadata['Altitude']],
                          'm.bearing': [rov_mag_heading]}
            rov_meas_bins_dict = bin_widths_1_150_mm(scallop_stats[key]['width_mm'])
            df_row_rov.update(rov_meas_bins_dict)
            df_row = dict(df_row_shared, **df_row_rov)
            append_to_csv(PROCESSED_BASEDIR + f"scallop_rov_{key}_stats.csv", pd.DataFrame(df_row))

    # If paired site, read from diver data and process
    if transect_map:
        # Get relevant diver data from provided xlsx IF diver anns layer isnt in VPZ file
        # TODO: read from shape layer instead of xlsx, make sure metadata is in layer too
        diver_data_xls = pd.ExcelFile(PROCESSED_BASEDIR + 'ruk2401_dive_slate_data_entry Kura Reihana.xlsx')
        survey_meas_df = pd.read_excel(diver_data_xls, 'scallop_data')
        survey_metadata_df = pd.read_excel(diver_data_xls, 'metadata')
        site_meas_df = survey_meas_df.loc[survey_meas_df['site'] == site_id]
        site_metadata_df = survey_metadata_df.loc[survey_metadata_df['site'] == site_id]

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
                gps_coord = metadata['lonlat'] + (70 + np.random.random((2,))) / 111e3
            diver_entries.append([t_para, t_perp, meas_width_mm])
            diver_entries_valid.append(check_inbounds(Point(gps_coord)))
            diver_points_gps.append(Point(gps_coord))
            diver_point_tags.append(diver_initials + ' ' + str(meas_width_mm) + ' mm')

        diver_entries_arr = np.array(diver_entries)

        if OUTPUT_SHAPE_LAYERS:
            diver_meas_gdf = gp.GeoDataFrame({'NAME': diver_point_tags, 'geometry': diver_points_gps,
                                              'Dist along T': diver_entries_arr[:, 0],
                                              'Dist to T': diver_entries_arr[:, 1]}, geometry='geometry')
            diver_meas_gdf.set_crs(SHAPE_CRS, inplace=True)
            file_utils.ensure_dir_exists(dir_full + 'shapes_ann')
            diver_meas_gdf.to_file(dir_full + 'shapes_ann/Diver Measurements.geojson', driver='GeoJSON')

        # Take only inbound and on-transect diver measurements
        diver_entries_arr = diver_entries_arr[diver_entries_valid]
        total_diver_count = diver_entries_arr.shape[0]
        # print(total_diver_count, len(diver_entries_inbounds_arr))

        # Add row in diver stats csv for paired site
        search_distance_left = site_metadata_df.loc[site_metadata_df['diver'].str.contains('Left')]['distance'].values[0]
        search_distance_right = site_metadata_df.loc[site_metadata_df['diver'].str.contains('Right')]['distance'].values[0]
        diver_search_area = 1.0 * (search_distance_left + search_distance_right)
        diver_depth = round(np.mean([np.mean(site_metadata_df[k]) for k in ['depth_s', 'depth_f']]), 2)
        diver_bearing = round(np.mean(site_metadata_df['bearing']))
        print(f"Diver TOTAL search area for {site_id} = {diver_search_area} m2")
        print(f"Diver TOTAL scallop count = {total_diver_count}")
        df_row_dive = {'area m2': [diver_search_area],
                       'count': [total_diver_count],
                       'depth': [diver_depth],
                       'altitude': [0],
                       'm.bearing': [diver_bearing],}
        rov_meas_bins_dict = bin_widths_1_150_mm(diver_entries_arr[:, 2])
        df_row_dive.update(rov_meas_bins_dict)
        df_row = dict(df_row_shared, **df_row_dive)
        append_to_csv(PROCESSED_BASEDIR + 'scallop_dive_stats.csv', pd.DataFrame(df_row))

        print("Converting ROV detections / annotations to transect frame and finding closest diver match")
        matched_scallop_widths = {k: [] for k in scallop_stats.keys()}
        for key, stats in scallop_stats.items():
            in_diver_search_widths = []
            for lon, lat, width_rov in zip(stats['lon'], stats['lat'], stats['width_mm']):
                res = transect_map.gps2transect((lon, lat))
                if res is None:
                    continue
                t_para, t_perp = res

                # If within diver search area, add to "Diver search overlap" list
                in_left_search_area = t_perp < 0 and t_para <= search_distance_left
                in_right_search_area = t_perp >= 0 and t_para <= search_distance_right
                if abs(t_perp) <= 1.0 and t_para >= 0.0 and (in_left_search_area or in_right_search_area):
                    in_diver_search_widths.append(width_rov)

                near_para = np.abs(diver_entries_arr[:, 0] - t_para) < PARA_DIST_THRESH
                near_perp = np.abs(diver_entries_arr[:, 1] - t_perp) < PERP_DIST_THRESH
                scallop_near = near_para * near_perp
                num_matches = np.sum(scallop_near)
                if num_matches != 1:
                    continue
                # assert num_matches == 1
                # TODO: deal with multiple matches
                matched_scallop_widths[key].append([width_rov, diver_entries_arr[scallop_near, 2][0]])

            if ONLY_IN_DIVER_SEARCH_AREA:
                print(f"Diver search area {key} scallop count = {len(in_diver_search_widths)}")
                # Add row in rov detection / annotation stats csv for site
                df_row_rov = {'area m2': [diver_search_area],
                              'count': [len(in_diver_search_widths)],
                              'depth': [metadata['Depth']],
                              'altitude': [metadata['Altitude']],
                              'm.bearing': [rov_mag_heading]}
                rov_meas_bins_dict = bin_widths_1_150_mm(in_diver_search_widths)
                df_row_rov.update(rov_meas_bins_dict)
                df_row = dict(df_row_shared, **df_row_rov)
                append_to_csv(PROCESSED_BASEDIR + f"scallop_rov_{key}_stats.csv", pd.DataFrame(df_row))

            matched_arr = np.array(matched_scallop_widths[key]).T
            if len(matched_arr):
                matched_error = matched_arr[0] - matched_arr[1]
                rov_count_eff_matched = matched_arr.shape[1] / total_diver_count
                print(f"ROV {key} matched count efficacy = {round(rov_count_eff_matched * 100)} %")
                # rov_count_eff_all = len(stats['width_mm']) / len(diver_measurements_gps)
                # print(f"ROV {key} ALL count efficacy = {round(rov_count_eff_all * 100)} %")
                print(f"ROV {key} matched sizing error STD = {round(np.std(matched_error), 2)} mm")
                mean_error = np.mean(matched_error)
                print(f"ROV {key} matched sizing error AVG = {round(np.mean(np.abs(matched_error - mean_error)), 2)} mm")
                print(f"ROV {key} matched sizing bias = {round(mean_error, 2)} mm")

                # plt.hist(matched_error, bins=50)
                # plt.show()

        # TODO: NIWA - Need err / bias for scallop detection efficiency and sizing, by detected size category
        # TODO: NIWA - Per (ROV detected) size class count multiplier
        # TODO: NIWA - Per (ROV detected) size class sizing bias


if __name__ == "__main__":
    if len(DIRS_LIST):
        dirs_list = DIRS_LIST
    else:
        with open(DONE_DIRS_FILE, 'r') as f:
            dirs_list = f.readlines()

    dirs_list = ['240714-113449\n']

    for dir_entry in dirs_list:
        if len(dir_entry) == 1 or '#' in dir_entry:
            continue
        dir_name = dir_entry[:13]

        process_dir(dir_name)

        # break

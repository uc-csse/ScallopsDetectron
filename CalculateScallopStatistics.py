import glob
import geopandas as gp
import pandas as pd
from shapely.geometry import *
import numpy as np
from utils import geo_utils, vpz_utils
from utils.transect_mapper import transect_mapper
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt

# Will break code:
# Overlapping polygons in the same class of either include OR exclude
# Tape reference must be part of vpz file not imported shape

SHAPE_CRS = "EPSG:4326"

PROCESSED_BASEDIR = "/csse/research/CVlab/processed_bluerov_data/"
DONE_DIRS_FILE = PROCESSED_BASEDIR + 'dirs_done.txt'

PARA_DIST_THRESH = 0.2
PERP_DIST_THRESH = 0.1

def get_poly_arr_2d(poly):
    return np.array(poly.exterior.coords)[:, :2]

def get_local_poly_arr(poly_gps):
    poly_arr_2d = get_poly_arr_2d(poly_gps)
    poly_arr_m = geo_utils.convert_gps2local(poly_arr_2d[0], poly_arr_2d)
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

    # TODO: multiple shape files - yes but need to be careful not to double count scallops
    # Load scallop polygons
    scallop_gpkg_paths = glob.glob(dir_full + '*detections_Filtered*.gpkg')
    scallop_polygons = {'detected': [], 'annotated': []}
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
        if "polygon" in label.lower():
            # Human annotation(s)?
            if not label in ann_layer_keys:
                ann_layer_keys.append(label)
            for i, row in shape_layer.iterrows():
                if isinstance(row.geometry, Polygon):
                    scallop_polygons["annotated"].append(row.geometry)
    print(ann_layer_keys)
    assert len(ann_layer_keys) == 1

    if os.path.isfile(dir_full + "scan_metadata.json"):
        with open(dir_full + "scan_metadata.json", 'r') as meta_doc:
            metadata = json.load(meta_doc)
    else:
        raise Exception(f"Site {dir_name} has no JSON metadata!")

    # TODO: Need err / bias for scallop annotation efficiency and sizing, by detected size category
    # TODO: csv row output for annotations

    # Get site name and standardize
    site_name = metadata['NAME']
    print(site_name)
    str_matches = [s for s in site_name.split(' ') if '.' in s]
    assert len(str_matches) == 1
    tmp_strs = str_matches[0].split('.')
    site_id = tmp_strs[0] + ' ' + '.'.join(tmp_strs[1:])
    print("Site ID:", site_id)
    # site_id = 'Mc 4'
    site_id = 'UQ 19'

    df_row_shared = {'site ID': [site_id],
                     'site name': [site_name],
                     'date-time site': [dir_name],
                     'date-time proc': [datetime.now().strftime("%y%m%d-%H%M%S")], }

    # TODO: check for overlap interclass and trim

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

    # Filter scallops in valid survey area(s)
    valid_scallop_polygons = {k: [] for k in scallop_polygons.keys()}
    for key, polygons in scallop_polygons.items():
        print(f"Total number of {key} scallops = {len(polygons)}")
        for spoly in polygons:
            # check if scallop is over 50% in include area and under 50% in any exclude areas
            valid = False
            scallop_area = get_poly_area_m2(spoly)
            for bound_polys, keep in [[include_polys, True], [exclude_polys, False]]:
                for b_poly in bound_polys:
                    if b_poly.intersects(spoly):
                        intersection_polygon = b_poly.intersection(spoly)
                        if intersection_polygon.is_empty:
                            continue
                        if isinstance(intersection_polygon, Polygon):
                            interestion_area = get_poly_area_m2(intersection_polygon)
                        elif isinstance(intersection_polygon, MultiPolygon) or isinstance(intersection_polygon,
                                                                                          GeometryCollection):
                            interestion_area = np.sum(
                                [get_poly_area_m2(p) for p in list(intersection_polygon.geoms) if isinstance(p, Polygon)])
                        else:
                            raise ValueError
                        if interestion_area > scallop_area / 2:
                            valid = keep
                            break
            if valid:
                valid_scallop_polygons[key].append(spoly)
        print(f"Number of valid {key} scallops = {len(valid_scallop_polygons[key])}")


    # test_write_gdf = gp.GeoDataFrame({'NAME': 'test', 'geometry': valid_scallop_polygons}, geometry='geometry')
    # test_write_gdf.to_file('/csse/users/tkr25/Desktop/valid_scallops.geojson', driver='GeoJSON')

    # TODO: Get 3D scallop polygon from DEM for annotation sizing...

    # Calculate valid scallop polygon widths (annotations and detections)
    scallop_stats = {k: {'lat': [], 'lon': [], 'width_mm': []} for k in valid_scallop_polygons}
    for key, valid_polygons in valid_scallop_polygons.items():
        for vspoly in valid_polygons:
            local_poly = get_local_poly_arr(vspoly)

            # TODO: improve sizing (shape fitting?)
            # naive_max_w:
            scallop_vert_mat = np.repeat(local_poly[None], local_poly.shape[0], axis=0)
            scallop_vert_dists = np.linalg.norm(scallop_vert_mat - scallop_vert_mat.transpose([1, 0, 2]), axis=2)
            max_width = np.max(scallop_vert_dists)

            poly_arr = get_poly_arr_2d(vspoly)
            scallop_stats[key]['width_mm'].append(round(max_width * 1000))
            lon, lat = np.mean(poly_arr, axis=0)
            scallop_stats[key]['lat'].append(lat)
            scallop_stats[key]['lon'].append(lon)

    # If paired site, read from diver data and process
    if transect_map:
        # Get relevant diver data from provided xlsx
        diver_data_xls = pd.ExcelFile(PROCESSED_BASEDIR + 'ruk2401_dive_slate_data_entry Kura Reihana.xlsx')
        survey_meas_df = pd.read_excel(diver_data_xls, 'scallop_data')
        site_meas_df = survey_meas_df.loc[survey_meas_df['site'] == site_id]
        survey_metadata_df = pd.read_excel(diver_data_xls, 'metadata')
        site_metadata_df = survey_metadata_df.loc[survey_metadata_df['site'] == site_id]

        diver_entries_transect = []
        diver_points_gps = []
        diver_measurements = []
        tags = []
        perp_ds = []
        para_ds = []
        for i, csv_row in site_meas_df.iterrows():
            t_para, t_perp = csv_row['y'], csv_row["x"]
            diver_initials = csv_row['diver'].split(' ')[0]
            left_side = 'Left' in str(csv_row['diver'])
            t_perp = (-1 if left_side else 1) * t_perp / 100
            meas_width_mm = csv_row["SCA_mm"]
            diver_entries_transect.append([t_para, t_perp, meas_width_mm])
            gps_coord = transect_map.transect2gps([t_para, t_perp])
            if gps_coord is not None:
                diver_points_gps.append(Point(gps_coord))
                diver_measurements.append(meas_width_mm)
                tags.append(diver_initials + ' ' + str(diver_measurements[-1]))
                perp_ds.append(t_perp)
                para_ds.append(t_para)
            else:
                print(f"Transect point {[t_para, t_perp]} is not in reconstructed transect!")
        diver_meas_gdf = gp.GeoDataFrame({'NAME': tags, 'geometry': diver_points_gps,
                                          'Dist along T': para_ds, 'Dist to T': perp_ds}, geometry='geometry')
        diver_meas_gdf.set_crs(SHAPE_CRS, inplace=True)
        diver_meas_gdf.to_file(dir_full + 'Diver Measurements.geojson', driver='GeoJSON')

        print("Converting ROV detections / annotations to transect frame and finding closest diver match")
        diver_meas_arr = np.array([para_ds, perp_ds, diver_measurements])
        matched_scallop_widths = {k: [] for k in scallop_stats.keys()}
        for key, stats in scallop_stats.items():
            for lon, lat, width_rov in zip(stats['lon'], stats['lat'], stats['width_mm']):
                res = transect_map.gps2transect((lon, lat))
                if res is None:
                    continue
                t_para, t_perp = res
                near_para = np.abs(diver_meas_arr[0] - t_para) < PARA_DIST_THRESH
                near_perp = np.abs(diver_meas_arr[1] - t_perp) < PERP_DIST_THRESH
                scallop_near = near_para * near_perp
                num_matches = np.sum(scallop_near)
                if num_matches == 0:
                    continue
                # assert np.sum(scallop_near) == 1
                # TODO: deal with multiple matches
                matched_scallop_widths[key].append([width_rov, diver_meas_arr[2][scallop_near][0]])

            matched_arr = np.array(matched_scallop_widths[key]).T
            matched_error = matched_arr[0] - matched_arr[1]
            rov_count_eff_matched = matched_arr.shape[1] / len(diver_measurements)
            rov_count_eff_all = len(stats['width_mm']) / len(diver_measurements)
            print(f"ROV {key} matched count efficacy = {round(rov_count_eff_matched * 100)} %")
            print(f"ROV {key} ALL count efficacy = {round(rov_count_eff_all * 100)} %")
            print(f"ROV {key} sizing error AVG = {round(np.mean(np.abs(matched_error)))} mm")
            print(f"ROV {key} sizing bias = {round(np.mean(matched_error))} mm")
            # plt.hist(matched_error, bins=20)
            # plt.show()

        # TODO: Need err / bias for scallop detection efficiency and sizing, by detected size category
        # TODO: Per (ROV detected) size class count multiplier
        # TODO: Per (ROV detected) size class sizing bias

        # Add row in diver stats csv for paired site
        diver_search_area = np.sum(site_metadata_df['distance'])
        diver_depth = round(np.mean([np.mean(site_metadata_df[k]) for k in ['depth_s', 'depth_f']]), 2)
        diver_bearing = round(np.mean(site_metadata_df['bearing']))
        print(f"Diver search area for {site_id} = {diver_search_area} m2")
        df_row_dive = {'depth': [diver_depth],
                       'bearing': [diver_bearing],
                       'area m2': [diver_search_area],
                       'count': [len(diver_measurements)]}
        rov_meas_bins_dict = bin_widths_1_150_mm(diver_measurements)
        df_row_dive.update(rov_meas_bins_dict)
        df_row = dict(df_row_shared, **df_row_dive)
        append_to_csv(PROCESSED_BASEDIR + 'scallop_dive_stats.csv', pd.DataFrame(df_row))

    # CSV with every scallop detection
    # site_dataframe = pd.DataFrame(scallop_stats)
    # with open(dir_full + 'valid_scallop_sizes.csv', 'w') as f:
    #     site_dataframe.to_csv(f, header=True, index=False)

    # Add row in rov detection / annotation stats csv for site
    for key, stats in scallop_stats.items():
        df_row_rov = {'longitude': [metadata['lonlat'][0]],
                      'latitude': [metadata['lonlat'][1]],
                      'area m2': [site_area],
                      'count': [len(stats['width_mm'])],
                      'depth': [metadata['Depth']],
                      'altitude': [metadata['Altitude']],
                      't.heading': [metadata['T.Heading']]}
        rov_meas_bins_dict = bin_widths_1_150_mm(stats['width_mm'])
        df_row_rov.update(rov_meas_bins_dict)
        df_row = dict(df_row_shared, **df_row_rov)
        append_to_csv(PROCESSED_BASEDIR + f"scallop_rov_{key}_stats.csv", pd.DataFrame(df_row))


if __name__ == "__main__":
    with open(DONE_DIRS_FILE, 'r') as f:
        dirs_list = f.readlines()

    dirs_list = ['240714-140552\n']

    for dir_entry in dirs_list:
        if len(dir_entry) == 1 or '#' in dir_entry:
            continue
        dir_name = dir_entry[:-1]

        process_dir(dir_name)
        break
from utils import vpz_utils, file_utils
import glob
import geopandas as gp


PROCESSED_BASE_DIR = '/csse/research/CVlab/processed_bluerov_data/'


if __name__ == '__main__':
    vpz_dir = PROCESSED_BASE_DIR + 'vpzs_annotated_niwa/paired/'
    vpz_paths = glob.glob(vpz_dir + '*.vpz')
    for vpz_path in vpz_paths:

        split_fp = vpz_path.split('/')
        dir_path = '/'.join(split_fp[:-1]) + '/'
        dir_name = vpz_path.split('/')[-1][:-4]
        print(f"Extracting shapes for {dir_name}")
        recon_dir = PROCESSED_BASE_DIR + dir_name + '/'
        shape_dir = recon_dir + 'shapes_ann/'
        file_utils.ensure_dir_exists(shape_dir)

        shape_layers_gpd = vpz_utils.get_shape_layers_gpd(dir_path, dir_name + '.vpz')
        for label, shape_layer in shape_layers_gpd:
            if any(s in label.lower() for s in ['poly', 'area', 'metadata', 'reference', '.gpkg']):
                continue
            geoms = []
            for i, row in shape_layer.iterrows():
                geoms.append(row.geometry)

            if len(geoms) > 0:
                gdf = gp.GeoDataFrame({'NAME': label, 'geometry': geoms}, geometry='geometry')
                gdf.to_file(shape_dir + label + '.gpkg')

        file_utils.SetFolderPermissions(shape_dir)

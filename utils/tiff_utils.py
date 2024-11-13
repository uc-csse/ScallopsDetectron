import tifffile
import glob
from utils import geo_utils
import numpy as np

TIFF_PAGE = 2

class DEM:
    def __init__(self, tiff_dir):
        dem_tiff_paths = glob.glob(tiff_dir + '*-dem-*.tif') + glob.glob(tiff_dir + '*-dem.tif')
        if len(dem_tiff_paths) == 0:
            raise Exception("No DEM tiffs found!")
        self.dem_imgs = []
        self.dem_res_gps = []
        ortho_lonlats = []
        for tif_pth in dem_tiff_paths:
            ortho_tiff_obj = tifffile.TiffFile(tif_pth)
            self.dem_imgs.append(tifffile.imread(tif_pth, key=TIFF_PAGE))
            ortho_page = ortho_tiff_obj.pages[TIFF_PAGE]  # Pages are resolution pyramid
            self.dem_res_gps.append(np.array(ortho_page.tags['ModelPixelScaleTag'].value)[:2])
            ortho_tiepoint = ortho_page.tags['ModelTiepointTag'].value
            ortho_lonlat = np.array(ortho_tiepoint)[3:5]
            ortho_lonlats.append(ortho_lonlat)  # Top left
            res_xy_m = geo_utils.convert_gps2local(ortho_lonlat, [ortho_lonlat + np.array(self.dem_res_gps[-1])])[0]
            # print(f"Tiff {tif_pth.split('/')[-1]} page {TIFF_PAGE}, res [m] = {np.round(res_xy_m, 4)}")
        self.ortho_lonlats = np.array(ortho_lonlats)

    def get_elevation_gps(self, gps_pnt):
        # lon lat
        vec_gps = gps_pnt - self.ortho_lonlats
        vec_gps[np.where(vec_gps[:, 0] < 0), :] = 100
        vec_gps[np.where(vec_gps[:, 1] > 0), :] = 100
        tiff_idx = np.argmin(np.abs(vec_gps).sum(axis=1))
        pix_idx = (np.array([1, -1]) * vec_gps[tiff_idx] / self.dem_res_gps[tiff_idx])[::-1].astype(int)
        dem_tile = self.dem_imgs[tiff_idx]
        if not (pix_idx[0] < dem_tile.shape[0] and pix_idx[1] < dem_tile.shape[1]):
            print(f"Pix idx: {pix_idx} not in DEM! Clipping...")
        pix_idx = np.clip(pix_idx, (0, 0), (dem_tile.shape[0]-1, dem_tile.shape[1]-1))
        return dem_tile[tuple(pix_idx)]

    def poly3d_from_dem(self, polygon_2d):
        polygon_3d = []
        for pnt in polygon_2d:
            z_val = self.get_elevation_gps(pnt)
            polygon_3d.append([pnt[0], pnt[1], z_val])
        return np.array(polygon_3d)

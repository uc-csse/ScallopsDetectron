from shapely import *
import zipfile
import tempfile
import geopandas as gp
import xml.etree.cElementTree as ET
import os
import json
from pathlib import Path


VIEWER_CRS_STRING = ("GEOGCS[\"WGS 84\","
                     "DATUM[\"World Geodetic System 1984 ensemble\","
                     "SPHEROID[\"WGS 84\",6378137,298.257223563,AUTHORITY[\"EPSG\",\"7030\"]],"
                     "TOWGS84[0,0,0,0,0,0,0],AUTHORITY[\"EPSG\",\"6326\"]],"
                     "PRIMEM[\"Greenwich\",0,AUTHORITY[\"EPSG\",\"8901\"]],"
                     "UNIT[\"degree\",0.01745329251994328,AUTHORITY[\"EPSG\",\"9102\"]],"
                     "AUTHORITY[\"EPSG\",\"4326\"]]")

SHAPE_CRS = "EPSG:4326"


def get_shape_layers_gpd(dir_path, vpz_fn):
    zf = zipfile.ZipFile(dir_path + vpz_fn)
    shape_layers = []
    with tempfile.TemporaryDirectory() as tempdir:
        zf.extractall(tempdir)
        vpz_root = ET.parse(tempdir + '/doc.xml').getroot()
        for child in vpz_root.iter('layer'):
            if child.attrib['type'] == 'shapes':
                elem_data = list(child.iter('data'))
                elem_src = list(child.iter('source'))
                if len(elem_data):
                    shape_fn = tempdir + '/' + elem_data[0].attrib['path']
                elif len(elem_src):
                    shape_fn = dir_path + elem_src[0].attrib['path']
                else:
                    continue
                try:
                    shape_layers.append([child.attrib['label'], gp.read_file(shape_fn)])
                except Exception as e:
                    print(e)
    return shape_layers


def write_vpz_file(vpz_filename, site_tags, tiff_paths):

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)
        doc = ET.Element("document", version="1.7.0")
        sites = ET.SubElement(doc, "sites")
        site = ET.SubElement(sites, "site", label="site0")
        layers = ET.SubElement(site, "layers")
        ET.SubElement(site, "reference").text = VIEWER_CRS_STRING

        for geo_tiff_path in tiff_paths:
            filename = geo_tiff_path.split('/')[-1]
            layer = ET.SubElement(layers, "layer", type='orthomosaic', label=filename[:-4], enabled='true')
            ET.SubElement(layer, "source", path=geo_tiff_path)

        filename = f'shape0.gpkg'
        metadata_gdf = gp.GeoDataFrame(site_tags, geometry='geometry')
        metadata_gdf.set_crs(SHAPE_CRS, inplace=True)
        metadata_gdf.to_file(temp_dir / filename, driver='GPKG')  # , driver='GPKG' , crs=SHAPE_CRS
        layer = ET.SubElement(layers, "layer", type='shapes', label='metadata', enabled='true')
        ET.SubElement(layer, "data", path=filename)

        ET.ElementTree(doc).write(temp_dir / "doc.xml", encoding='UTF-8', xml_declaration=True)

        if os.path.isfile(vpz_filename):
            print("Overwriting existing vpz file")
            os.remove(vpz_filename)
        with zipfile.ZipFile(vpz_filename, 'w') as vpz_file:
            for f in temp_dir.glob("*"):
                vpz_file.write(f, arcname=f.name)
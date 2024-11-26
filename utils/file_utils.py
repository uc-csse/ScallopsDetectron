import os, shutil
import zipfile
import glob
import xml.etree.ElementTree as ET


def del_if_exists(path):
    if os.path.exists(path):
        os.remove(path)

def SetFolderPermissions(folder, read_only=False):
    # TKRFILES=$(ls -l | grep tkr25 | grep 2304 | tr -s ' ' | cut -f9 -d ' ')
    # chmod -R g+rwX,o+rX $TKRFILES
    cmd = "chmod -R g+r-wX " if read_only else "chmod -R g+rwX"
    os.system(cmd + ' ' + folder)



def clear_dir(dir_path):
    files = os.listdir(dir_path)
    for file in files:
        file_path = os.path.join(dir_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
def ensure_dir_exists(dirpath, clear=False):
    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    elif clear:
        clear_dir(dirpath)

def open_archive(dir, mode="r"):
    vpz_files = glob.glob(dir + '*.vpz')
    assert len(vpz_files) == 1
    return zipfile.ZipFile(vpz_files[0], mode)

def open_vpz_r(dir, mode="r"):
    archive = open_archive(dir, mode)
    xml_file = [f for f in archive.namelist() if ".xml" in f]
    assert len(xml_file) == 1
    root = ET.fromstring(archive.read(xml_file[0]))
    return root, archive

def get_xml_layers(xml_root):
    sites = xml_root.find('sites')
    assert len(sites) == 1
    return sites[0].find('layers')

def append_vpz_shapes(vpz_dir, shapes_fp_l, col_rgb_l=None):
    xml_root, archive_r = open_vpz_r(vpz_dir, "r")
    labels = [fp.split('/')[-1].split('.')[0] for fp in shapes_fp_l]
    tmp_archive_fp = vpz_dir+'.tmp.vpz'
    archive_w = zipfile.ZipFile(tmp_archive_fp, 'w')
    layers = get_xml_layers(xml_root)
    # Copy all files which are not xml or gpkg to new archive
    for file in archive_r.filelist:
        if not any(k in file.filename for k in ['.xml', '.gpkg']):
            archive_w.writestr(file.filename, archive_r.read(file.filename))
    # copy over all shape files not being overwritten to new archive,
    # delete xml entries of files which are being overwritten
    sf_idx = 0
    for layer in list(layers):
        if not layer.get('type') == 'shapes':
            continue
        shp_fp = layer.find('data').get('path')
        if layer.get('label') in labels:
            layers.remove(layer)
        else:
            new_fn = 'shapes{}.gpkg'.format(sf_idx)
            archive_w.writestr(new_fn, archive_r.read(shp_fp))
            layer.find('data').set('path', new_fn)
            sf_idx += 1
    # Append updated shapes to vpz archive and xml file
    cols_hex = ['#ffffff'] * len(shapes_fp_l) if col_rgb_l is None else ['#%02x%02x%02x' % c for c in col_rgb_l]
    for label, fp, col in list(zip(labels, shapes_fp_l, cols_hex)):
        new_fn = 'shapes{}.gpkg'.format(sf_idx)
        archive_w.write(fp, new_fn)
        content='''\
                <layer type="shapes" label="{}" enabled="true">
                    <data path="{}"/>
                    <meta>
                        <property name="style/color" value="{}"/>
                    </meta>
                </layer>
                '''.format(label, new_fn, col)
        layers.append(ET.XML(content))
        sf_idx += 1
    archive_w.writestr('doc.xml', ET.tostring(xml_root))
    old_archive_fp = archive_r.fp.name
    archive_w.close()
    archive_r.close()
    shutil.move(tmp_archive_fp, old_archive_fp)

def extract_vpz_shapes(vpz_dir):
    xml_root, archive = open_vpz_r(vpz_dir)
    tmpshps_dir = vpz_dir+'gpkg_files/.tmpshps/'
    os.mkdir(tmpshps_dir)
    archive.extractall(tmpshps_dir, members=[f for f in archive.namelist() if ".gpkg" in f])
    layers = get_xml_layers(xml_root)
    xml_shapes_layers = [layer for layer in layers if layer.get('type') == 'shapes' and layer.find('data') is not None]
    for layer in xml_shapes_layers:
        fn_nospace = '_'.join(layer.get('label').split(' '))
        shutil.move(tmpshps_dir+layer.find('data').get('path'), vpz_dir+'gpkg_files/'+fn_nospace+'.gpkg')
    shutil.rmtree(tmpshps_dir)

def get_vpz_dataset_paths(vpz_dir):
    vpz_files = glob.glob(vpz_dir + '*.vpz')
    assert len(vpz_files) == 1
    archive = zipfile.ZipFile(vpz_files[0])
    zipped_shape_files = [f for f in archive.namelist() if ".gpkg" in f]
    file_paths = []
    for zipped_fn in zipped_shape_files:
        file_paths.append('zip://' + 'Station_3_grid.vpz' + '!' + zipped_fn) # vpz_files[0]
        print(file_paths[-1])
    return file_paths
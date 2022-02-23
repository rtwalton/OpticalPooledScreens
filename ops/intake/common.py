import numpy as np
import pandas as pd
from scipy.spatial.kdtree import KDTree
from scipy.stats import mode
from itertools import product
from ops.filenames import name_file as name
from ops.filenames import parse_filename as parse
from ops.io import save_stack as save

try:
    from pims import ND2Reader_SDK
except:
    from pims import ND2_Reader as ND2Reader_SDK
from nd2reader import ND2Reader

from ops.constants import *

ND2_EXPORT_FILE_PATTERN = ('.*'
        r'Well(?P<well_ix>\d+)_.*'
        r'Seq(?P<seq>\d+).*?'
        r'(?P<mag>\d+X).'    
        r'(?:(?P<cycle>[^_\.]*)_)?.*'
        r'(?P<m>\d{4})'                   
        r'(?:\.(?P<tag>.*))*\.(?P<ext>.*)')

ND2_EXPORT_FILE_PATTERN_96 = ('.*'
        r'Well(?P<well>.\d\d)_.*'
        r'Seq(?P<seq>\d+).*?'
        r'(?P<mag>\d+X).'    
        r'(?:(?P<cycle>[^_\.]*)_)?.*'
        r'(?P<m>\d{4})'                   
        r'(?:\.(?P<tag>.*))*\.(?P<ext>.*)')

ND2_FILE_PATTERN_LF = [(r'(?P<cycle>c[0-9]+)?/?'
    r'(?P<dataset>.*)?/?'
    r'Well(?P<well>[A-H][0-9]*)_'
    r'(Point[A-H][0-9]+_(?P<site>[0-9]*)_)?'
    r'Channel((?P<channel_1>[^_,]+)(_[^,]*)?)?,?'
    r'((?P<channel_2>[^_,]+)(_[^,]*)?)?,?'
    r'((?P<channel_3>[^_,]+)(_[^,]*)?)?,?'
    r'((?P<channel_4>[^_,]+)(_[^,]*)?)?,?'
    r'((?P<channel_5>[^_,]+)(_[^,]*)?)?,?'
    r'((?P<channel_6>[^_,]+)(_[^_]*)?)?'
    r'_Seq([0-9]+).(?P<ext>.*)')]

def add_neighbors(df_info, num_neighbors=9, radius_leniency=10):
    xy = ['x_um', 'y_um']
    xy = [GLOBAL_X, GLOBAL_Y]
    site = SITE
    
    df = df_info.drop_duplicates(xy)
    kdt = KDTree(df[xy].values)

    distance, index = kdt.query(df[xy].values, k=num_neighbors)

    # convert to m
    index = np.array(df[site])[index]

    m = mode(distance.max(axis=1).astype(int)).mode[0]
    filt = (distance < (m + radius_leniency)).all(axis=1)

    it = zip(df.loc[filt, site], index[filt])
    arr = []
    for m, ix in it:
        arr += [{site: m, 'ix': sorted(ix)}]

    return df_info.merge(pd.DataFrame(arr), how='left')

def parse_nd2_filename(f):
    description = parse(f,custom_patterns=ND2_FILE_PATTERN_LF)

    if 'site' in description.keys():
        description['site']=int(description['site'])

    return description

## ND2Reader (pure python)
# initial tests show this is faster at metadata extraction, slower at image reading

def extract_nd2_metadata_py(f,variables=['frames','x_data','y_data','z_data','t_data'],f_description=None):
    if f_description is None:
        f_description = parse_nd2_filename(f)

    with ND2Reader(f) as nd2:
        df = pd.DataFrame({key:val for key,val in nd2.metadata.items() if key in variables})
        if 'z_data' in variables:
            df = pd.concat([df,
                            pd.DataFrame(np.array(df['z_data'].tolist()),
                                         columns=['z_'+str(level) for level in nd2.metadata['z_levels']])
                           ],axis=1) 
            df = df.drop(columns='z_data')
        if 't_data' in variables:
            t_data = np.array(nd2.timesteps).reshape(-1,len(nd2.metadata['z_levels']))
            df = pd.concat([df,
                            pd.DataFrame(t_data,columns=['t_'+str(step) for step in range(t_data.shape[1])])
                           ],axis=1)
        if 'v' in nd2.axes:
            df['site'] = nd2.metadata['fields_of_view']
        else:
            df = df.assign(site = f_description['site'])
            
    (df
     .assign(well = f_description['well'])
     .to_pickle(name(f_description,ext='pkl'))
    )

## pims_nd2 (Nikon SDK based)
# initial tests show this is slower at metadata extraction, faster at image reading

def get_metadata_at_coords(nd2, **coords):
    import pims_nd2
    h = pims_nd2.ND2SDK

    _coords = {'t': 0, 'c': 0, 'z': 0, 'o': 0, 'm': 0}
    _coords.update(coords)
    c_coords = h.LIMUINT_4(int(_coords['t']), int(_coords['m']), 
                           int(_coords['z']), int(_coords['o']))
    i = h.Lim_GetSeqIndexFromCoords(nd2._lim_experiment,
                                c_coords)

    h.Lim_FileGetImageData(nd2._handle, i, 
                           nd2._buf_p, nd2._buf_md)


    return {'x_um': nd2._buf_md.dXPos,
                    'y_um': nd2._buf_md.dYPos,
                    'z_um': nd2._buf_md.dZPos,
                    't_ms': nd2._buf_md.dTimeMSec,
                }

def get_axis_size(nd2,axis):
    try:
        size = list(range(nd2.sizes[axis]))
    except:
        size = [0]
    return size

def extract_nd2_metadata_sdk(f, interpolate=True, progress=None):
    """Interpolation fills in timestamps linearly for each well; x,y,z positions 
    are copied from the first time point. 
    """
    with ND2Reader_SDK(f) as nd2:
        ts = get_axis_size(nd2,'t')
        ms = get_axis_size(nd2,'m')
        zs = get_axis_size(nd2,'z')   

        if progress is None:
            progress = lambda x: x

        if len(ts)==len(ms)==len(zs)==0:
            arr = [get_metadata_at_coords(nd2)]
        else:
            arr = []
            for t, m, z in progress(list(product(ts, ms, zs))):
                if len(ms)>1:
                    boundaries = [0, nd2.sizes['m'] - 1]
                    skip = m not in boundaries and t > 0
                else:
                    skip=False
                if interpolate and skip:
                    metadata = {}
                else:
                    metadata = get_metadata_at_coords(nd2, t=t, m=m, z=z)
                metadata['t'] = t
                metadata['m'] = m
                metadata['z'] = z
                metadata['file'] = f
                metadata.update()
                arr += [metadata]
    
        
    df_info = pd.DataFrame(arr)
    if interpolate:
        return (df_info
         .sort_values(['m', 't'])
         .assign(x_um=lambda x: x['x_um'].fillna(method='ffill'))
         .assign(y_um=lambda x: x['y_um'].fillna(method='ffill'))        
         .assign(z_um=lambda x: x['z_um'].fillna(method='ffill'))         
         .sort_values(['t', 'm'])
         .assign(t_ms=lambda x: x['t_ms'].interpolate())
                )
    else:
        return df_info



def build_file_table(f_nd2, f_template, wells):
    """
    Example:
    
    wells = 'A1', 'A2', 'A3', 'B1', 'B2', 'B3'
    f_template = 'input/10X_Hoechst-mNeon/10X_Hoechst-mNeon_A1.live.tif'
    build_file_table(f_nd2, f_template, wells)
    """
    rename = lambda x: name(parse(f_template), **x)

    get_well = lambda x: wells[int(re.findall('Well(\d)', x)[0]) - 1]
    df_files = (common.extract_nd2_metadata_sdk(f_nd2, progress=tqdn)
     .assign(well=lambda x: x['file'].apply(get_well))
     .assign(site=lambda x: x['m'])
     .assign(file_=lambda x: x.apply(rename, axis=1))
    )
    
    return df_files

def export_nd2_sdk_file_table(f_nd2, df_files):

    df = df_files.drop_duplicates('file_')

    with ND2Reader_SDK(f_nd2) as nd2:

        nd2.iter_axes = 'm'
        nd2.bundle_axes = ['t', 'c', 'y', 'x']

        for m, data in tqdn(enumerate(nd2)):
            f_out = df.query('m == @m')['file_'].iloc[0]
            save(f_out, data)

## user-defined nd2 reader backend

def read_nd2(f,slicer=slice(None),backend='ND2SDK'):
    if backend == 'ND2SDK':
        reader = ND2Reader_SDK
        axes = list('mtzcyx')
    elif backend == 'python':
        reader = ND2Reader
        axes = list('vtzcyx')
    else:
        raise ValueError('Only "ND2SDK" and "python" backends are available.')

    with reader(f) as nd2:
        exist_axes = [ax for ax in axes if ax in nd2.axes]
        nd2.iter_axes = exist_axes[0]
        nd2.bundle_axes = exist_axes[1:]
        data = np.array(nd2[slicer])

    return data

def export_nd2(f, iter_axes='v', project_axes=False, slicer=slice(None), f_description=None, split=False, backend='ND2SDK',transpose=None,**kwargs):
    if f_description is None:
        f_description = parse_nd2_filename(f)

    if backend == 'ND2SDK':
        reader = ND2Reader_SDK
        axes = list('mtzcyx')
        iter_axes = 'm' if iter_axes=='v' else iter_axes
    elif backend == 'python':
        reader = ND2Reader
        axes = list('vtzcyx')
        iter_axes = 'v' if iter_axes=='m' else iter_axes
    else:
        raise ValueError('Only "ND2SDK" and "python" backends are available.')

    if not iter_axes in axes:
        raise ValueError(f'Supplied iter_axes \'{iter_axes}\' not in axes options for backend \'{backend}\' ({axes})')

    with reader(f) as nd2:
        if split:
            nd2.iter_axes = iter_axes
            nd2.bundle_axes = [ax for ax in axes if (ax in nd2.axes) & (ax != iter_axes)]
            if project_axes:
                g = lambda x: x.max(axis=nd2.bundle_axes.index(project_axes))
                axes.remove(project_axes)
            else:
                g = lambda x: x

            axes.remove(nd2.iter_axes[0])
            # preserve inner singleton dimensions
            axes = [ax in nd2.bundle_axes for ax in axes]
            axes = axes[axes.index(True):]
            for site,data in enumerate(nd2[slicer]):
                data = g(data)
                im = data[tuple([slice(None) if ax else None for ax in axes])]
                if transpose is not None:
                    im = np.transpose(im,transpose)
                save(name(f_description,site=site, ext='tif'),
                    im, 
                    **kwargs)
        else:
            axes_exist = [ax for ax in axes if ax in nd2.axes]
            nd2.iter_axes = axes_exist[0]
            nd2.bundle_axes = axes_exist[1:]
            if project_axes:
                data = np.max(nd2,axis=axes_exist.index(project_axes))
                axes_exist.remove(project_axes)
            else:
                data = np.array(nd2)
            # preserve inner singleton dimensions
            axes = [ax in axes_exist for ax in axes]
            axes = axes[axes.index(True):]
            im = data[tuple([slice(None) if ax else None for ax in axes])]
            if transpose is not None:
                im = np.transpose(im,transpose)
            save(name(f_description, ext='tif'),
                im, 
                **kwargs)
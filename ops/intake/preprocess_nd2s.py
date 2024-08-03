from glob import glob
from joblib import Parallel, delayed
from ops.intake.common_luke import parse_nd2_filename, export_nd2
import os
from tqdm.auto import tqdm
import fire

def process_files(files, **kwargs):
    if not isinstance(files, list):
        files = glob(files)

    def process_site(f, remove=False, iter_axes='m', project_axes=False, split=True, backend='ND2SDK', **kwargs):
        description = parse_nd2_filename(f)
        print(description)
        kwargs_ = kwargs.copy()
        description['mag'] = kwargs_.pop('mag', '10X')
        description['subdir'] = kwargs_.pop('subdir', description['dataset'].split('/', 1)[0])
        description['dataset'] = kwargs_.pop('dataset', 'preprocess')
        description['cycle'] = kwargs_.pop('cycle', description['cycle'])
        export_nd2(f, iter_axes=iter_axes, project_axes=project_axes, slicer=slice(None), f_description=description,
                   split=split, backend=backend, **kwargs)
        if remove:
            os.remove(f)

#     Parallel(backend='loky', n_jobs=kwargs.pop('n_jobs', -2))(
#         delayed(process_site)(file, **kwargs) for file in files)

    for file in files:
        process_site(file, **kwargs)
        
    return "Files processed"

if __name__ == '__main__':
    fire.Fire(process_files)

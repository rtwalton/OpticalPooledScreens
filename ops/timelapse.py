import ops.utils

import networkx as nx
import pandas as pd
import numpy as np
import scipy.spatial.kdtree
from collections import Counter
from scipy.spatial.distance import cdist

from scipy.interpolate import UnivariateSpline
from statsmodels.stats.multitest import multipletests
    
def format_stats_wide(df_stats):
    index = ['gene_symbol']
    columns = ['stat_name', 'stimulant']
    values = ['statistic', 'pval', 'pval_FDR_10']

    stats = (df_stats
     .pivot_table(index=index, columns=columns, values=values)
     .pipe(ops.utils.flatten_cols))

    counts = (df_stats
     .pivot_table(index=index, columns='stimulant', values='count')
     .rename(columns=lambda x: 'cells_' + x))

    return pd.concat([stats, counts], axis=1)


def distribution_difference(df):
    col = 'dapi_gfp_corr_early'
    y_neg = (df
      .query('gene_symbol == "non-targeting"')
      [col]
    )
    return df.groupby('gene_symbol').apply(lambda x:
      scipy.stats.wasserstein_distance(x[col], y_neg))


def add_est_timestamps(df_all):
    s_per_frame = 24 * 60
    sites_per_frame = 2 * 364
    s_per_site = s_per_frame / sites_per_frame
    starting_time = 3 * 60

    cols = ['frame', 'well', 'site']
    df_ws = df_all[cols].drop_duplicates().sort_values(cols)

    est_timestamps = [(starting_time + i*s_per_site) / 3600
                      for i in range(len(df_ws))]

    df_ws['timestamp'] = est_timestamps

    return df_all.join(df_ws.set_index(cols), on=cols)


def add_dapi_diff(df_all):
    index = ['well', 'site', 'cell_ph']
    dapi_diff = (df_all
     .pivot_table(index=index, columns='frame', 
                  values='dapi_max')
     .pipe(lambda x: x/x.mean())
     .pipe(lambda x: x.max(axis=1) - x.min(axis=1))
     .rename('dapi_diff')
    )
    
    return df_all.join(dapi_diff, on=index)


def add_spline_diff(df, s=25):

    T_neg, Y_neg = (df
     .query('gene_symbol == "non-targeting"')
     .groupby('timestamp')
     ['dapi_gfp_corr'].mean()
     .reset_index().values.T
    )

    ix = np.argsort(T_neg)
    spl = UnivariateSpline(T_neg[ix], Y_neg[ix], s=s)

    return (df
     .assign(splined=lambda x: spl(df['timestamp']))
     .assign(spline_diff=lambda x: x.eval('dapi_gfp_corr - splined'))
    )


def get_stats(df, col='spline_diff'):
    df_diff = (df
     .groupby(['gene_symbol', 'cell'])
     [col].mean()
     .sort_values(ascending=False)
     .reset_index())

    negative_vals = (df_diff
     .query('gene_symbol == "non-targeting"')
     [col]
    )

    test = lambda x: scipy.stats.ttest_ind(x, negative_vals).pvalue

    stats = (df_diff.groupby('gene_symbol')
     [col]
     .pipe(ops.utils.groupby_reduce_concat, 'mean', 'count', 
           pval=lambda x: x.apply(test))
     .assign(pval_FDR_10=lambda x: 
            multipletests(x['pval'], 0.1)[1]))
    
    return stats

# track nuclei nearest neighbor

def initialize_graph(df):
    arr_df = [x for _, x in df.groupby('frame')]
    nodes = df[['frame', 'label']].values
    nodes = [tuple(x) for x in nodes]

    G = nx.DiGraph()
    G.add_nodes_from(nodes)

    edges = []
    for df1, df2 in zip(arr_df, arr_df[1:]):
        edges = get_edges(df1, df2)
        G.add_weighted_edges_from(edges)
    
    return G


def get_edges(df1, df2):
    neighboring_points = 3
    get_label = lambda x: tuple(int(y) for y in x[[2, 3]])

    x1 = df1[['i', 'j', 'frame', 'label']].values
    x2 = df2[['i', 'j', 'frame', 'label']].values
    
    kdt = scipy.spatial.kdtree.KDTree(df1[['i', 'j']])
    points = df2[['i', 'j']]

    result = kdt.query(points, neighboring_points)
    edges = []
    for i2, (ds, ns) in enumerate(zip(*result)):
        end_node = get_label(x2[i2])
        for d, i1 in zip(ds, ns):
            start_node = get_label(x1[i1])
            w = d
            edges.append((start_node, end_node, w))

    return edges


def displacement(x):
    d = np.sqrt(np.diff(x['x'])**2 + np.diff(x['y'])**2)
    return d


def analyze_graph(G, cutoff=100):
    """Trace a path forward from each nucleus in the starting frame. Only keep 
    the paths that reach the final frame.
    """
    start_nodes = [n for n in G.nodes if n[0] == 0]
    max_frame = max([frame for frame, _ in G.nodes])
    
    cost, path = nx.multi_source_dijkstra(G, start_nodes, cutoff=cutoff)
    cost = {k:v for k,v in cost.items() if k[0] == max_frame}
    path = {k:v for k,v in path.items() if k[0] == max_frame}
    return cost, path


def filter_paths(cost, path, threshold=35):
    """Remove intersecting paths. 
    returns list of one [(frame, label)] per trajectory
    """
    # remove intersecting paths (node in more than one path)
    node_count = Counter(sum(path.values(), []))
    bad = set(k for k,v in node_count.items() if v > 1)
    print('bad', len(bad), len(node_count))

    # remove paths with cost over threshold
    too_costly = [k for k,v in cost.items() if v > threshold]
    bad = bad | set(too_costly)
    
    relabel = [v for v in path.values() if not (set(v) & bad)]
    assert(len(relabel) > 0)
    return relabel


def relabel_nuclei(nuclei, relabel):
    nuclei_ = nuclei.copy()
    max_label = nuclei.max() + 1
    for i, nodes in enumerate(zip(*relabel)):
        labels = [n[1] for n in nodes]
        table = np.zeros(max_label).astype(int)
        table[labels] = range(len(labels))
        nuclei_[i] = table[nuclei_[i]]

    return nuclei_

# track nuclei trackmate

def call_TrackMate_centroids(input_path, output_path='trackmate_output.csv', fiji_path=None, threads=1, tracker_settings=dict()):
    '''warnings:    - `threads` is probably not actually setting the max threads for fiji. 

                    - to allow multiple instances of fiji to run concurrently (e.g., launched from snakemake pipeline), likely have 
                    to set `allowMultiple` parameter in Fiji.app/Contents/Info.plist to true.

    `CUTOFF_PERCENTILE` parameter in tracker_settings changes the alternative cost to gap closing/merging/splitting. Higher values -> 
    more gap closures/merges/splits.
    '''
    import subprocess, json

    if fiji_path is None:
        import sys
        if sys.platform == "darwin":
            fiji_path = '/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx'
        elif sys.platform == "linux":
            fiji_path = '~/Fiji.app/ImageJ-linux64'
        else:
            raise ValueError("Currently only OS X and linux systems can infer Fiji install location.")

    tracker_defaults = {"LINKING_MAX_DISTANCE":60.,"GAP_CLOSING_MAX_DISTANCE":60.,
                        "ALLOW_TRACK_SPLITTING":True,"SPLITTING_MAX_DISTANCE":60.,
                        "ALLOW_TRACK_MERGING":True,"MERGING_MAX_DISTANCE":60.,
                        "MAX_FRAME_GAP":2,"CUTOFF_PERCENTILE":0.90}


    for key, val in tracker_defaults.items():
        _ = tracker_settings.setdefault(key,val)
    
    trackmate_call = ('''{fiji_path} --ij2 --headless --console --run {ops_path}/external/TrackMate/track_centroids.py'''
                        .format(fiji_path=fiji_path,ops_path=ops.__path__[0]))

    variables = ('''"input_path='{input_path}',output_path='{output_path}',threads={threads},tracker_settings='{tracker_settings}'"'''
                    .format(input_path=input_path,output_path=output_path,
                        threads=int(threads),tracker_settings=json.dumps(tracker_settings)))
    
    output = subprocess.check_output(' '.join([trackmate_call,variables]), shell=True)
    print(output.decode("utf-8"))

def format_trackmate(df):
    import ast

    df = (pd.concat([df,
        pd.DataFrame(df['parent_ids'].apply(lambda x: ast.literal_eval(x)).tolist(),
            index = df.index,columns=['parent_id_0','parent_id_1'])
        ],axis=1)
        .fillna(value=-1)
        .drop(columns=['parent_ids'])
        .assign(relabel=-1,parent_cell_0=-1,parent_cell_1=-1)
        .astype(int)
        .set_index('id')
    )

    lookup = np.zeros((df.index.max()+2,3),dtype=int)

    lookup[df.index] = (df
        [['cell','parent_id_0','parent_id_1']]
        .values
        )

    lookup[-1] = np.array([-1,-1,-1])

    set_cols = ['relabel','parent_cell_0','parent_cell_1']

    current = 1
    
    arr_frames = []
    for frame,df_frame in df.groupby('frame'):
        df_frame = df_frame.copy()
        
        if frame==0:
            arr_frames.append(df_frame.assign(relabel = list(range(current,current+df_frame.pipe(len))),
                                              parent_cell_0 = -1,
                                              parent_cell_1 = -1))
            current += df_frame.pipe(len)
            continue

        # unique child from single parent
        idx_propagate = ((df_frame.duplicated(['parent_id_0','parent_id_1'],keep=False)==False)
                &
              ((df_frame[['parent_id_0','parent_id_1']]==-1).sum(axis=1)==1)
             ).values

        lookup[df_frame[idx_propagate].index.values] = df_frame.loc[idx_propagate,set_cols] = lookup[df_frame.loc[idx_propagate,'parent_id_0'].values]

        # split, merge, or new
        idx_new = ((df_frame.duplicated(['parent_id_0','parent_id_1'],keep=False))
                |
              ((df_frame[['parent_id_0','parent_id_1']]==-1).sum(axis=1)!=1)
             ).values

        lookup[df_frame[idx_new].index.values] = df_frame.loc[idx_new,set_cols] = np.array([list(range(current,current+idx_new.sum())),
                                                                                            lookup[df_frame.loc[idx_new,'parent_id_0'].values,0],
                                                                                            lookup[df_frame.loc[idx_new,'parent_id_1'].values,0]
                                                                                            ]).T
        current += idx_new.sum()
        arr_frames.append(df_frame)
    
    return pd.concat(arr_frames).reset_index()

# recover parent relationships
## during some iterations of trackmate, saving of parent cell identities was unintentionally
## commented out. these functions infer these relationships. For a single tile, correctly assigned
## same parent-child relationships as trackmate for >99.8% of cells. Well-constrained problem.

def recover_parents(df_tracked,threshold=60, cell='cell', ij=('i','j'), keep_cols=['well','tile','track_id','cell']):
    # to be run on a table from a single tile

    # get junction cells
    df_pre_junction = (df_tracked
                       .groupby(['track_id',cell],group_keys=False)
                       .apply(lambda x: x.nlargest(1,'frame'))
                      )
    df_post_junction = (df_tracked
                        .groupby(['track_id',cell],group_keys=False)
                        .apply(lambda x: x.nsmallest(1,'frame'))
                       )
    
    arr = []
    
    # assign frame 0 cells or un-tracked cells with no parents
    arr.append(df_post_junction
               .query('frame==0 | track_id==-1')
               [keep_cols]
               .assign(parent_cell_0=-1,parent_cell_1=-1)
              )
    
    # clean up tables
    last_frame = int(df_tracked['frame'].nlargest(1))
    df_pre_junction = df_pre_junction.query('frame!=@last_frame & track_id!=-1')
    df_post_junction = df_post_junction.query('frame!=0 & track_id!=-1')
    
    # categorize frames to avoid issues with no-cell junction frames
    df_pre_junction.loc[:,'frame'] = pd.Categorical(df_pre_junction['frame'],
                                                      categories=np.arange(0,last_frame),
                                                      ordered=True)
    df_post_junction.loc[:,'frame'] = pd.Categorical(df_post_junction['frame'],
                                                      categories=np.arange(1,last_frame+1),
                                                      ordered=True)
    
    for (frame_pre,df_pre),(frame_post,df_post) in zip(df_pre_junction.groupby('frame'),
                                                       df_post_junction.groupby('frame')):
        if df_post.pipe(len)==0:
            continue
        elif df_pre.pipe(len)==0:
            arr.append(df_post[keep_cols].assign(parent_cell_0=-1,parent_cell_1=-1))
        else:
            arr.extend(junction_parent_assignment(pd.concat([df_pre,df_post]),
                                               frame_0=frame_pre,
                                               threshold=threshold,
                                               ij=ij,
                                               cell=cell,
                                               keep_cols=keep_cols
                                              )
                      )
        
    return pd.concat(arr,ignore_index=True)

def junction_parent_assignment(df_junction, frame_0, threshold, ij, cell, keep_cols):
    arr = []
    i,j = ij
    for track,df_track_junction in df_junction.groupby('track_id'):
        if (df_track_junction['frame'].nunique()==1):
            if df_track_junction.iloc[0]['frame']==(frame_0+1):
                # only post-junction cells -> parents = -1
                arr.append(df_track_junction[keep_cols].assign(parent_cell_0=-1,parent_cell_1=-1))
            elif df_track_junction.iloc[0]['frame']==frame_0:
                # only pre-junction cells -> ends of tracks, don't have to assign
                continue
        else:
            before,after = (g[[cell,i,j]].values 
                            for _,g 
                            in df_track_junction.groupby('frame')
                           )
            distances = cdist(after[:,1:],before[:,1:])
            edges = distances<threshold

            edges = resolve_conflicts(edges,distances, conflict_type='extra')
            edges = resolve_conflicts(edges,distances,conflict_type='tangle')

            parents = tuple(before[edge,0] 
                            if edge.sum()>0 else np.array([-1,-1]) 
                            for edge in edges)

            if len(parents) != edges.shape[0]:
                raise ValueError('Length of parents tuple does not match number of post-junction cells')

            if max([len(p) for p in parents])>2:
                raise ValueError(f'''Conflict resolution error; too many parents selected for at least one cell 
                for track {track} in frame {frame_0}
                                 ''')

            parents = np.array([np.concatenate([p,np.array([-1])])
                                if len(p)==1
                                else p
                                for p in parents
                               ]
                              )
            
            arr.append(df_track_junction.query('frame==@frame_0+1')
                       [keep_cols]
                       .assign(parent_cell_0=parents[:,0],parent_cell_1=parents[:,1])
                      )

    return arr 

def resolve_conflicts(edges,distances,conflict_type='tangle'):
    if conflict_type=='tangle':
        # any cell with more than one edge is a potential conflict
        edge_threshold = 1
        conflict_threshold = 1
    elif conflict_type=='extra':
        # any cell with more than 2 edges is a potential conflict
        edge_threshold = 2
        conflict_threshold = 0

    def evaluate_conflicts(edges,edge_threshold):
        # check for conflicting edges
        # conflicts matrix for `tangle`: 1 is an edge, >1 is a conflict edge
        # for `extra`: >0 is a conflict edge: more than 2 edges to a single cell
        conflicts = np.zeros(edges.shape)
        conflicts += (edges.sum(axis=0)>edge_threshold)
        conflicts += (edges.sum(axis=1)>edge_threshold)[:,None]
        conflicts[~edges] = 0
        return conflicts

    conflicts = evaluate_conflicts(edges,edge_threshold)
    
    while (conflicts>conflict_threshold).sum()>0:
        # remove longest edge
        edges[distances==distances[conflicts>conflict_threshold].max()] = False
        # re-evaluate conflicts
        conflicts = evalutate_conflicts(edges,edge_threshold)
        
    return edges

# plot traces

def plot_traces_gene_stim(df, df_neg, gene):
    import ops.figures.plotting
    fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(12, 12), 
                        sharex=True, sharey=True)  
    for stim, df_1 in df.groupby('stimulant'):
        if stim == 'TNFa':
            axs_ = axs[:2]
            color = ops.figures.plotting.ORANGE
        else:
            axs_ = axs[2:]
            color = ops.figures.plotting.BLUE

        x_neg, y_neg = (df_neg
         .query('stimulant == @stim')
         .groupby(['frame'])
         [['timestamp', 'dapi_gfp_corr']].mean()
         .values.T)

        for ax, (sg, df_2) in zip(axs_.flat[:], 
                                 df_1.groupby('sgRNA_name')):
            plot_traces(df_2, ax, sg, color)
            ax.plot(x_neg, y_neg, c='black')
            
    return fig


def plot_traces(df, ax, sgRNA_label, color):
    
    index = ['well', 'tile', 'cell', 'sgRNA_name']
    values = ['timestamp', 'dapi_gfp_corr']
    wide = (df
     .pivot_table(index=index, columns='frame', 
                   values=values)
    )
    x = wide['timestamp'].values
    y = wide['dapi_gfp_corr'].values
    
    ax.plot(x.T, y.T, c=color, alpha=0.2)
    ax.set_title(sgRNA_label)
    
# timelapse montages

def subimage_timelapse(filename, bounds, frames=None, max_frames=None):
    # from ops.io import read_hdf_image
    import tables
    # maximum of subimages from a single timelapse frame
    max_frame_bounds = max([len(bound) for bound in bounds])
    
    # maximum shape of subimage
    max_bound_shape = tuple(np.array([(bound[2]-bound[0],bound[3]-bound[1]) 
                                for frame_bounds 
                                in bounds 
                                for bound 
                                in frame_bounds])
                            .max(axis=0)
                           )
    
    if frames is None:
        frames = list(range(len(bounds)))
    if max_frames is None:
        max_frames = len(frames)
        
    I = np.zeros((max_frames,1,max_bound_shape[0],max_bound_shape[1]*max_frame_bounds),dtype=np.uint16)

    hdf_file = tables.file.open_file(filename,mode='r')
    image_node = hdf_file.get_node('/',name='image')
    
    for frame_count,(frame,frame_bounds) in enumerate(zip(frames,bounds)):
        leading_dims = (slice(frame,frame+1),slice(None))
        
        for bound_count,bound in enumerate(frame_bounds):
            i0, j0 = max(bound[0], 0), max(bound[1], 0)
            i1, j1 = min(bound[2], image_node.shape[-2]), min(bound[3], image_node.shape[-1])

            try:
                data = image_node[leading_dims+(slice(i0,i1),slice(j0,j1))]
            except:
                print('error')
                data = None

            I[(slice(frame_count,frame_count+1),
               slice(None),
               slice(0,data.shape[-2]),
               slice(max_bound_shape[1]*bound_count,(max_bound_shape[1]*bound_count)+data.shape[-1]))] = data

    hdf_file.close()
    return I

def timelapse_montage_guide(df_guide, cell_width=60, montage_width=25, max_frames=None, tqdm=False, 
    file_pattern='{plate}/process_ph/images/20X_{well}_mCherry_Tile-{tile}.aligned.hdf'):
    from ops.annotate import add_rect_bounds

    df_guide = (df_guide
                .drop_duplicates(['plate','well','tile','track_id','tracked_cell','frame'])
                .sort_values(['tracked_length','track_id','frame','tracked_cell'])
                .pipe(add_rect_bounds,width=cell_width,ij=['i_tracked','j_tracked'],bounds_col='bounds')
               )
    
    arr = []
    
    if max_frames is None:
        max_frames = df_guide['frame'].nunique()

    if tqdm:
        import tqdm.notebook
        work = tqdm.notebook.tqdm(df_guide.groupby(['plate','well','tile','track_id']))
    else:
        work = df_guide.groupby(['plate','well','tile','track_id'])
    
    for (plate,well,tile,track_id),df_track in work:
        arr.append(subimage_timelapse(file_pattern.format(plate=plate,well=well,tile=tile),
                                    frames=sorted(df_track['frame'].unique()),
                                    max_frames=max_frames,
                                    bounds=[df['bounds'].tolist() for _,df in df_track.groupby('frame')]#,
                                   )
                  )
    fill = montage_width-(int(sum([track.shape[-1] for track in arr])/(cell_width*2))%montage_width)
    arr.append(np.zeros((max_frames,1,(cell_width*2),(cell_width*2*fill)),dtype=np.uint16))
    montage = np.concatenate(arr,axis=-1)
    return np.concatenate(np.split(montage,montage.shape[-1]/(montage_width*cell_width*2),axis=-1),axis=-2)

def timelapse_montage_guide_track(df_guide, track_width=100, montage_width=10, max_frames=None, tqdm=False,
    file_pattern='{plate}/process_ph/images/20X_{well}_mCherry_Tile-{tile}.aligned.hdf'):
    from ops.annotate import add_rect_bounds
    df_tracks = (df_guide
                .drop_duplicates(['plate','well','tile','track_id','tracked_cell','frame'])
                .sort_values(['tracked_length','track_id','frame','tracked_cell'])
                .groupby(['plate','well','tile','track_id','frame'])
                [['i_tracked','j_tracked']]
                .mean()
                .reset_index()
                .pipe(add_rect_bounds,width=track_width,ij=['i_tracked','j_tracked'],bounds_col='bounds')
               )
    
    arr = []
    
    if max_frames is None:
        max_frames = df_guide['frame'].nunique()

    if tqdm:
        import tqdm.notebook
        work = tqdm.notebook.tqdm(df_guide.groupby(['plate','well','tile','track_id']))
    else:
        work = df_guide.groupby(['plate','well','tile','track_id'])
    
    for (plate,well,tile,track_id),df_track in work:
        arr.append(tracked_subimage(file_pattern.format(plate=plate,well=well,tile=tile),
                                    frames=sorted(df_track['frame'].unique()),
                                    max_frames=max_frames,
                                    bounds=[df['bounds'].tolist() for _,df in df_track.groupby('frame')]
                                   )
                  )
    fill = montage_width-(int(sum([track.shape[-1] for track in arr])/(track_width*2))%montage_width)
    arr.append(np.zeros((max_frames,1,(track_width*2),(track_width*2*fill)),dtype=np.uint16))
    montage = np.concatenate(arr,axis=-1)
    return np.concatenate(np.split(montage,montage.shape[-1]/(montage_width*track_width*2),axis=-1),axis=-2)

def timelapse_montage_gene(df_gene,cell_width=40,montage_width=25,groupby='sgRNA',
                           file_pattern='{plate}/process_ph/images/20X_{well}_mCherry_Tile-{tile}.aligned.hdf'):
    arr = []
    max_frames = df_gene['frame'].nunique()
    
    for guide_count,(_,df_guide) in enumerate(df_gene.groupby('sgRNA')):
        guide_montage = timelapse_montage_guide(df_guide,
                                                cell_width=cell_width,
                                                montage_width=montage_width,
                                                max_frames=max_frames,
                                                file_pattern=file_pattern
                                               )
        if guide_count != 0:
            guide_montage = np.concatenate([np.zeros(guide_montage.shape[:-2]+(cell_width*2,guide_montage.shape[-1]),dtype=np.uint16),
                                       guide_montage],
                                      axis=-2
                                     )
        arr.append(guide_montage)
        
    return np.concatenate(arr,axis=-2)


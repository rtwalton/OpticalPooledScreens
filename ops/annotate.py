import numpy as np
import pandas as pd
import skimage.morphology
import warnings
from itertools import count
import os
import PIL.Image
import PIL.ImageFont

from ops.constants import *
import ops.filenames
import ops
import ops.io


# load font
def load_truetype(truetype='visitor1.ttf',size=10):
    """
    Note that `size` here is the "em" size in pixels, which is different than 
    the actual height of the letters for most fonts.
    """
    PATH = os.path.join(os.path.dirname(ops.__file__), truetype)
    try:
        return PIL.ImageFont.truetype(PATH,size=size)
    except OSError as e:
        warnings.warn('TrueType font not found at {0}'.format(PATH))

VISITOR_FONT = load_truetype()

def annotate_labels(df, label, value, label_mask=None, tag='cells', outline=False):
    """Transfer `value` from dataframe `df` to a saved integer image mask, using 
    `label` as an index. 

    The dataframe should contain data from a single image, which is loaded from
    `label_mask` if provided, or else guessed based on descriptors in the first 
    row of `df` and `tag`. 
    """
    if df[label].duplicated().any():
        raise ValueError('duplicate rows present')

    label_to_value = df.set_index(label, drop=False)[value]
    index_dtype = label_to_value.index.dtype
    value_dtype = label_to_value.dtype
    if not np.issubdtype(index_dtype, np.integer):
        raise ValueError('label column {0} is not integer type'.format(label))

    if not np.issubdtype(value_dtype, np.number):
        label_to_value = label_to_value.astype('category').cat.codes
        warnings.warn('converting value column "{0}" to categorical'.format(value))

    if label_to_value.index.duplicated().any():
        raise ValueError('duplicate index')

    top_row = df.iloc[0]
    if label_mask is None:
        filename = ops.filenames.guess_filename(top_row, tag)
        labels = ops.io.read_stack(filename)
    elif isinstance(label_mask, str):
        labels = ops.io.read_stack(label_mask)
    else:
        labels = label_mask
    
    if outline:
        labels = outline_mask(labels, 'inner')
    
    phenotype = relabel_array(labels, label_to_value)
    
    return phenotype


def annotate_points(df, value, ij=('i', 'j'), width=3, shape=(1024, 1024)):
    """Create a mask with pixels at coordinates `ij` set to `value` from 
    dataframe `df`. 
    """

    if shape=='1x1':
        shape = (2048,2048)
    elif shape=='2x2':
        shape = (1024,1024)
        
    ij = df[list(ij)].values.astype(int)
    n = ij.shape[0]
    mask = np.zeros(shape, dtype=df[value].dtype)
    mask[ij[:, 0], ij[:, 1]] = df[value]

    selem = np.ones((width, width))
    mask = skimage.morphology.dilation(mask, selem)

    return mask


def relabel_array(arr, new_label_dict):
    """Map values in integer array based on `new_labels`, a dictionary from
    old to new values.
    """
    n = arr.max()
    arr_ = np.zeros(n+1)
    for old_val, new_val in new_label_dict.items():
        if old_val <= n:
            arr_[old_val] = new_val
    return arr_[arr]


def outline_mask(arr, direction='outer', width=1):
    """Remove interior of label mask in `arr`.
    """
    selem = skimage.morphology.disk(width)
    arr = arr.copy()
    if direction == 'outer':
        mask = skimage.morphology.erosion(arr, selem)
        arr[mask > 0] = 0
        return arr
    elif direction == 'inner':
        mask1 = skimage.morphology.erosion(arr, selem) == arr
        mask2 = skimage.morphology.dilation(arr, selem) == arr
        arr[mask1 & mask2] = 0
        return arr
    else:
        raise ValueError(direction)
    

def bitmap_label(labels, positions, colors=None):
    positions = np.array(positions).astype(int)
    if colors is None:
        colors = [1] * len(labels)
    i_all, j_all, c_all = [], [], []
    for label, (i, j), color in zip(labels, positions, colors):
        if label == '':
            continue
        i_px, j_px = np.where(lasagna.io.bitmap_text(label))
        i_all += list(i_px + i)
        j_all += list(j_px + j)
        c_all += [color] * len(i_px)
        
    shape = max(i_all) + 1, max(j_all) + 1
    arr = np.zeros(shape, dtype=int)
    arr[i_all, j_all] = c_all
    return arr


def build_discrete_lut(colors):
    """Build ImageJ lookup table for list of discrete colors. 

    If the values to  label are in the range 0..N, N + 1 colors should be 
    provided (zero value is usually black). Color values should be understood 
    by `sns.color_palette` (e.g., "blue", (1, 0, 0), or "#0000ff").
    """
    try:
        import seaborn as sns
        colors = sns.color_palette(colors)
    except:
        pass
    colors = 255 * np.array(colors)

    # try to match ImageJ LUT rounding convention
    m = len(colors)
    n = int(256 / m)
    p = m - (256 - n * m)
    color_index_1 = list(np.repeat(range(0, p), n))
    color_index_2 = list(np.repeat(range(p, m), n + 1))
    color_index = color_index_1 + color_index_2
    return colors_to_imagej_lut(colors[color_index, :])

def bitmap_draw_line(image,coords,width=1,dashed=False):
    """Draw horizontal line, returning an image of same shape.
    Dashed if requested.
    """
    import PIL.ImageDraw

    if (len(coords)>2)&(dashed is not False):
        raise ValueError('Drawing a dashed line between more than 2 points not supported.')
    if (coords[0][1]!=coords[1][1])&(dashed is not False):
        raise ValueError('Drawing a dashed non-horizontal line not supported')

    if image.dtype==np.uint16:
        mode='I;16'
        fill = 2**16-1
    elif image.dtype==np.uint8:
        mode='L'
        fill = 2**8-1
    else:
        mode='1'
        fill = True

    img = PIL.Image.new(mode, image.shape[:-3:-1])
    draw = PIL.ImageDraw.Draw(img,mode=mode)

    if dashed:
        y = coords[0][1]
        if not isinstance(dashed,list):
            dashed = [100,50] # dash, gap
        xs = []
        x = coords[0][0]
        counter = count(start=0,step=1)
        while x<coords[1][0]:
            xs.append(x)
            c = next(counter)
            if c%2==0:
                x+=dashed[0]
            else:
                x+=dashed[1]
        xs.append(coords[1][0])
        for x_0,x_1 in zip(xs[::2],xs[1::2]):
            draw.line([(x_0,y),(x_1,y)],width=width,fill=fill)
    else:
        draw.line(coords,width=width,fill=fill)

    return np.array(img)

def bitmap_text_overlay(image,anchor_point,text,size=10,font=VISITOR_FONT):
    """Draw text in the shape of the given image.
    """
    import PIL.ImageDraw

    if image.dtype==np.uint16:
        mode='L' # PIL has a bug with drawing text on uint16 images
    elif image.dtype==np.uint8:
        mode='L'
    else:
        mode='1'

    img = PIL.Image.new(mode, image.shape[:-3:-1])
    draw = PIL.ImageDraw.Draw(img)

    if isinstance(font,PIL.ImageFont.FreeTypeFont):
        FONT = font
        if FONT.size != size:
            warnings.warn(f'Size of supplied FreeTypeFont object is {FONT.size}, '
            f'but input argument size = {size}.'
            )
    else:
        FONT = load_truetype(truetype=font,size=size)
    offset = FONT.getoffset(text)

    draw.text(np.array(anchor_point)-np.array(offset),text,font=FONT,fill='white')

    if image.dtype==np.uint16:
        return skimage.img_as_uint(np.array(img))
    else:
        return np.array(img,dtype=image.dtype)

def bitmap_line(s,crop=True):
    """Draw text using Visitor font (characters are 5x5 pixels).
    """
    import PIL.Image
    import PIL.ImageDraw
    img = PIL.Image.new("RGBA", (len(s) * 8, 10), (0, 0, 0))
    draw = PIL.ImageDraw.Draw(img)
    draw.text((0, 0), s, (255, 255, 255), font=VISITOR_FONT)
    draw = PIL.ImageDraw.Draw(img)

    n = np.array(img)[2:7, :, 0]
    if (n.sum() == 0)|(~crop):
        return n
    return (n[:, :np.where(n.any(axis=0))[0][-1] + 1] > 0).astype(int)


def bitmap_lines(lines, spacing=1,crop=True):
    """Draw multiple lines of text from a list of strings.
    """
    bitmaps = [bitmap_line(x,crop=crop) for x in lines]
    height = 5
    shapes = np.array([x.shape for x in bitmaps])
    shape = (height + 1) * len(bitmaps), shapes[:, 1].max()

    output = np.zeros(shape, dtype=int)
    for i, bitmap in enumerate(bitmaps):
        start, end = i * (height + 1), (i + 1) * (height + 1) - 1
        output[start:end, :bitmap.shape[1]] = bitmap

    return output[:-1, :]


def colors_to_imagej_lut(lut_values):
    """ImageJ header expects 256 red values, then 256 green values, then 
    256 blue values.
    """
    return tuple(np.array(lut_values).T.flatten().astype(int))


def build_GRMC():
    import seaborn as sns
    colors = (0, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 1)
    lut = []
    for color in colors:
        lut.append([0, 0, 0, 1])
        lut.extend(sns.dark_palette(color, n_colors=64 - 1))
    lut = np.array(lut)[:, :3]
    RGCM = np.zeros((256, 3), dtype=int)
    RGCM[:len(lut)] = (lut * 255).astype(int)
    return tuple(RGCM.T.flatten())


def add_rect_bounds(df, width=10, ij='ij', bounds_col='bounds'):
    arr = []
    for i,j in df[list(ij)].values.astype(int):
        arr.append((i - width, j - width, i + width, j + width))
    return df.assign(**{bounds_col: arr})

def make_sq_bounds(
        df,
        input_bounds=['bounds_0','bounds_1','bounds_2','bounds_3'],
        bounds_col='bounds'):

    def split_pad(pad):
            return (pad//2,pad//2+pad%2)

    arr = []
    for bounds in df[input_bounds].values.astype(int):
        width,height = (bounds[2]-bounds[0]),(bounds[3]-bounds[1])
        diff = height-width
        pad_width, pad_height = split_pad(np.clip(diff,0,None)),split_pad(np.clip(-diff,0,None))
        arr.append(tuple(bounds+np.array([-pad_width[0],-pad_height[0],pad_width[1],pad_height[1]])))
    return df.assign(**{bounds_col: arr})

# BASE LABELING

colors = (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 0, 1), (0, 1, 1)
GRMC = build_discrete_lut(colors)


def add_base_codes(df_reads, bases, offset, col):
    n = len(df_reads[col].iloc[0])
    df = (df_reads[col].str.extract('(.)'*n)
          .applymap(bases.index)
          .rename(columns=lambda x: 'c{0}'.format(x+1))
         )
    return pd.concat([df_reads, df + offset], axis=1)


def annotate_bases(df_reads, col='barcode', bases='GTAC', offset=1, **kwargs):
    """
    from ops.annotate import add_base_codes, GRMC
    labels = annotate_bases(df_reads)
    # labels = annotate_bases(df_cells, col='cell_barcode_0')

    data = read('process/10X_A1_Tile-7.log.tif')
    labeled = join_stacks(data, (labels[:, None], '.a'))

    luts = GRAY, GREEN, RED, MAGENTA, CYAN, GRMC 
    save('test/labeled', labeled, luts=luts)
    """
    df_reads = add_base_codes(df_reads, bases, offset, col)
    n = len(df_reads[col].iloc[0])
    cycles = ['c{0}'.format(i+1) for i in range(n)]
    labels = np.array([annotate_points(df_reads, c, **kwargs) for c in cycles])
    return labels



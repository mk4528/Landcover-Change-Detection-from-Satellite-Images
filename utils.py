import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from skimage import io
import numpy as np
import geopandas
from tqdm import trange
import seaborn as sns


NLCD_CLASS = {
     0: "No Data",
    11: "Open Water",
    12: "Perennial Ice/Snow",
    21: "Developed, Open Space",
    22: "Developed, Low Intensity",
    23: "Developed, Medium Intensity",
    24: "Developed, High Intensity",
    31: "Barren Land (Rock/Sand/City)",
    41: "Deciduous Forest",
    42: "Evergreen Forest",
    43: "Mixed Forest",
    52: "Shrub/Scrub",
    71: "Grassland/Herbaceous",
    81: "Pasture/Hay",
    82: "Cultivated Crops",
    90: "Woody Wetlands",
    95: "Emergent Herbaceous Wetlands"
}


MAPPING = {
    "Water": [11],
    "Tree Canopy": [41, 42, 43, 52, 90, 95],
    "Low Vegetation": [21, 71, 81, 82],
    "Impervious": [22, 23, 24, 31],
    "No Data": [0, 12],
}


def read_nlcd_img_as_np(df, img_number: int, img_year: int):
    """Read NLCD image given image number and year as np.ndarray
    
    Args:
        df: a dataframe that contains the url to the TIF image
        img_number: int, 3716
        img_year: int, 2013/2016
    
    Returns:
        np.ndarray
    """
    for i in range(len(df)):
        if f"{img_number}_nlcd-{img_year}" in df.loc[i, "label_fn"]:
            return io.imread(df.loc[i, "label_fn"])


def get_percentages(geojson_path) -> dict:
    gdf = geopandas.read_file(geojson_path)
    gdf = gdf.to_crs("EPSG:3395")
    gdf["area (sq meters)"] = gdf.area
    total_area = gdf['area (sq meters)'].sum()
    all_classes = gdf['default'].unique()
    res = {}
    for c in all_classes:
        gdf_sub = gdf.loc[gdf['default'] == c]
        sub_class_area_perc = gdf_sub['area (sq meters)'].sum() / total_area
        res[c] = sub_class_area_perc
    return res


def get_dist(img, pixel2label: dict, classes: list) -> dict:
    """Calculate class percentages, super fast
    
    Args:
        img: numpy.ndarray
        pixel2label: mapping of pixel: class labels
        classes: a list of classes, for example 
            ["Water", "Impervious", "Tree Canopy", "Low Vegetation"]
    
    Returns:
        1. dict: {"Water": 0.xx, "Impervious": 0.xx, ...}
        2. dict only restricts to pixels whose labels changed from old year to 
            new year
    """
    res = {k: 0 for k in classes}
    unique_pixels = np.unique(img)
    pixesl_counts = np.bincount(img.flatten())
    for p in unique_pixels:
        label = pixel2label[p]
        res[label] += pixesl_counts[p]

    sum1 = sum(res.values())
    for k, v in res.items():
        res[k] = v / sum1
    return res


def get_dist_diffpixel(old_img, new_img, pixel2label: dict, classes: list) -> dict:
    """Calculate class percentages, super fast
    
    Args:
        old_img: numpy.ndarray
        new_img: numpy.ndarray
        pixel2label: mapping of pixel: class labels
        classes: a list of classes, for example 
            ["Water", "Impervious", "Tree Canopy", "Low Vegetation"]
    
    Returns:
        dict only restricts to pixels whose labels changed from old year to new year
    """
    res = {k: 0 for k in classes}
    old_img_diff = old_img[old_img!=new_img]
    unique_pixels = np.unique(old_img_diff)
    for p in unique_pixels:
        res[pixel2label[p]] += sum(old_img_diff==p)

    sum1 = sum(res.values())
    for k, v in res.items():
        res[k] = v / sum1
    return res


def find_dist_change(old_img, new_img, pixel2label: dict, classes: list) -> dict:
    """Find distribution change in classes from 2013 to 2016 
    figure out for all the pixels in old_img, what have they changed to in new img
    """
    dist_change = {k: {m: 0 for m in classes} for k in classes}
    unique_old = np.unique(old_img)
    for p in unique_old:
        new_class_ary = new_img[old_img==p]
        unique_new = np.unique(new_class_ary)
        for q in unique_new:
            dist_change[pixel2label[p]][pixel2label[q]] += sum(new_class_ary==q)
            
    for k in dist_change.keys():
        sum_value = sum(dist_change[k].values())
        for m in dist_change[k].keys():
            dist_change[k][m] /= sum_value

    return dist_change


def find_dist_change_restricted(old_img, new_img, pixel2label: dict, classes: list) -> dict:
    dist_change = {k: {m: 0 for m in classes} for k in classes}
    unique_old = np.unique(old_img)
    for p in unique_old:
        new_class_ary = new_img[(old_img!=new_img)&(old_img==p)]
        unique_new_pixels = np.unique(new_class_ary)
        for q in unique_new_pixels:
            dist_change[pixel2label[p]][pixel2label[q]] += sum(new_class_ary==q)

    for k in dist_change.keys():
        sum_value = sum(dist_change[k].values()) - dist_change[k][k]
        for m in dist_change[k].keys():
            if sum_value == 0:
                dist_change[k][m] = 0
            else:
                dist_change[k][m] /= sum_value
    
    return dist_change
        

def create_direct_map() -> dict:
    """Create a mapping of NLCD pixel value: class labels
    e.g. 11: 'Water'. """
    res = {}
    for k in NLCD_CLASS.keys():
        for m, n in MAPPING.items():
            if k in n:
                res[k] = m
    return res


def get_percentages(geojson_path) -> dict:
    gdf = geopandas.read_file(geojson_path)
    gdf = gdf.to_crs("EPSG:3395")
    gdf["area (sq meters)"] = gdf.area
    total_area = gdf['area (sq meters)'].sum()
    all_classes = gdf['default'].unique()
    res = {}
    for c in all_classes:
        gdf_sub = gdf.loc[gdf['default'] == c]
        sub_class_area_perc = gdf_sub['area (sq meters)'].sum() / total_area
        res[c] = sub_class_area_perc
    return res


def plot_bars(dist_left: dict, dist_right: dict, 
              labels: list, title: str, output_path: str) -> None:
    """Make bar plots 
    
    Args:
        dist_left: distribution on the left in the bar plot
        dist_right: ... on the right in the bar plot
        labels: a list of names, for example ["NLCD", "high-res"]
        title: title of the image
        classes: list of classes
        output_path: e.g. "xxx.png"
    """
    df = pd.DataFrame.from_records([dist_left, dist_right], 
                                    index=labels)
    df = df.reset_index().rename(columns={"index": "Annotation"})
    df = df.melt(id_vars=["Annotation"], 
                 value_vars=list(dist_left.keys()), 
                 var_name="Class", value_name="Percentage")
    df["Percentage"] = df["Percentage"] * 100
    # round to 4 decimal places
    df = df.round({"Percentage": 4})
    ax = sns.barplot(data=df, x="Class", y="Percentage", hue="Annotation")
    ax.set_title(title)
    ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    # add texts above bars
    ax.bar_label(ax.containers[0], fontsize=9)
    ax.bar_label(ax.containers[1], fontsize=9)
    # iterate over the texts and format to percentages
    for t in ax.texts: 
        t.set_text(f"{float(t.get_text()):.2f}%")
    plt.savefig(f"{output_path}",bbox_inches='tight')


def plot_heatmaps(dist_old: dict, distribution_change: dict, 
                  classes: list, title: str, xlabel: str, ylabel: str, figname: str) -> None:
    """Plot heatmaps of class distributions in 2013 vs 2017 and save the image. """
    index_col = []
    for c in classes:
        index_col.append(f"{c} ({dist_old[c] * 100:.2f}%)")

    heatmap_df = pd.DataFrame.from_records(data=[{m: distribution_change[k][m] for m in classes} for k in classes], 
                                           index=index_col)
    heatmap_df = heatmap_df * 100
    heatmap_df = heatmap_df.round(1)
    
    # plot heatmap
    ax = sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu")
    # set the colorbar to percentage format
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # iterate over the texts and format to percentages
    for t in ax.texts: 
        t.set_text(f"{float(t.get_text())}%")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(figname, bbox_inches='tight')


def plot_heatmaps_no_diagonal(dist_old: dict, distribution_change: dict, 
                              classes: list, title: str, xlabel: str, ylabel: str, figname: str) -> None:
    # construct the index column
    index_col = []
    for c in classes:
        index_col.append(f"{c} ({dist_old[c] * 100:.2f}%)")
        
    heatmap_df = pd.DataFrame.from_records(data=[{m: distribution_change[k][m] for m in classes} for k in classes], 
                                           index=index_col)
    heatmap_df = heatmap_df * 100
    heatmap_df = heatmap_df.round(1)
    
    # plot heatmap
    mask = np.eye(len(heatmap_df))
    ax = sns.heatmap(heatmap_df, annot=True, cmap="YlGnBu", mask=mask)
    # set the colorbar to percentage format
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # iterate over the texts and format to percentages
    for t in ax.texts: 
        t.set_text(f"{float(t.get_text())}%")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(figname, bbox_inches='tight')


# borrowed from Masa and Yuki
def store_np_and_png(nlcd_path, hr_label_path, output_path, image_id_year):
    """
    Please export the high-res label(s) from GroundWork using "export data" tab.
    You only need to select the check-box of tile layer for the image options.
    Then you can DL catalog.zip from GroundWork.
    In this function, you will import the geojason file where high-res label data are stored.
    We also import the original NLCD low-res label, but this is only to get the size of the *label*.
    
    Note that we need to at least label the upper left and the lower right corner of the original NAIP image,
    because the largest and smallest coordinates are required to accurately transform to a png image file.

    Input:
        nlcd_path: File path of the downloaded NLCD image
        hr_label_path: File path of the exported label data (/catalog/labels/data/xxx.geojson) from GroundWork
        output_path: Folder path to save outputs
        image_id_year: 'Image_file_ID + "_" + year' to name outputs

    Output:
        image_gray_np: np.array of high-resolution label in gray scale
        image: rgb image with three channels (light blue: Water, dark green: Tree Canopy, light green: Low Vegetation, red: Impervious).
               Use this to visually check the curated high-res label
        image_gray: gray-scale image with only one channel (0: Water, 1: Tree Canopy, 2: Low Vegetation, 3: Impervious).
                Note that these are the same with Codalab https://codalab.lisn.upsaclay.fr/competitions/7908). Use this to test our models
    """
    import sys
    from osgeo import gdal,ogr,osr
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    import json
    import pandas as pd
    import numpy as np
    import re

    def calc_map_LonLat(nlcd_path):
        def intermidiate_calc(extracted_xy):
            ms  = re.search('\..*"',  extracted_xy).group()[1:-1]
            sec = re.search('\'.*\.', extracted_xy).group()[1:-1]
            min = re.search('d.*\'',  extracted_xy).group()[1:-1]
            deg = re.search('.*d',    extracted_xy).group()[0:-1]
            return(int(deg) + int(min) / 60 + int(sec) / 3600 + int(ms) / 3600000)
        
        def calc_mapXY(extracted_UL_LR):
            mapX = - intermidiate_calc(re.search('\).*W', extracted_UL_LR).group()[3:-1])
            mapY =   intermidiate_calc(re.search('W,.*N', extracted_UL_LR).group()[2:-1])
            return(mapX, mapY)

        txt = gdal.Info(nlcd_path)
        extracted_UL = re.search('Upper Left.*\nLower Left', txt).group()
        extracted_LR = re.search('Lower Right.*\nCenter', txt).group()
        _mapXmin_, _mapYmax_ = calc_mapXY(extracted_UL)
        _mapXmax_, _mapYmin_ = calc_mapXY(extracted_LR)

        return(_mapXmin_, _mapYmax_, _mapXmax_, _mapYmin_)

    def make_save_image_np(nlcd_path, hr_label_path, output_path, image_id_year):
        # Open the data source and read in the extent
        data_source = ogr.Open(hr_label_path)
        if data_source is None:
            print ('Could not open file')
            sys.exit(1)

        mapXmin, mapXmax, mapYmin, mapYmax = data_source.GetLayer().GetExtent()
        # http://gdal.org/python/osgeo.ogr.Layer-class.html#GetExtent

        _mapXmin_, _mapYmax_, _mapXmax_, _mapYmin_ = calc_map_LonLat(nlcd_path)

        print("X and Y-coordinates are longitude and latitude respectively",
              "\nmapXmin: ", mapXmin, "\nmapXmax: ", mapXmax, "\nmapYmin: ", mapYmin, "\nmapYmax: ", mapYmax,
              "\n\nfrom corresponding NLCD",
              "\nmapXmin: ", _mapXmin_, "\nmapXmax: ", _mapXmax_, "\nmapYmin: ", _mapYmin_, "\nmapYmax: ", _mapYmax_,
              "\nNote that it seems to be better to use the one from the exported high-res label from GroundWork",
              "\n\nW:", mapXmax - mapXmin, "\nH:", mapYmax - mapYmin)

        # Define pixel_size 
        # pixel_size = 0.5 # meters are one pixel
        # Create the target data source
        gdal_nlcd = gdal.Open(nlcd_path, gdal.GA_ReadOnly)
        target_Width = gdal_nlcd.RasterXSize
        target_Height = gdal_nlcd.RasterYSize

        pixel_size_x = abs(mapXmax - mapXmin) / target_Width
        pixel_size_y = abs(mapYmax - mapYmin) / target_Height
        print("pixel_size_x:", pixel_size_x, "\npixel_size_y:", pixel_size_y)

        image_TC = Image.new('RGB', (target_Width, target_Height))
        image_gray_TC = Image.new('L', (target_Width, target_Height))
        image_Im = Image.new('RGB', (target_Width, target_Height))
        image_gray_Im = Image.new('L', (target_Width, target_Height))
        image_W = Image.new('RGB', (target_Width, target_Height))
        image_gray_W = Image.new('L', (target_Width, target_Height))
        image_LV = Image.new('RGB', (target_Width, target_Height))
        image_gray_LV = Image.new('L', (target_Width, target_Height))

        draw_TC = ImageDraw.Draw(image_TC)
        draw_gray_TC = ImageDraw.Draw(image_gray_TC)
        draw_Im = ImageDraw.Draw(image_Im)
        draw_gray_Im = ImageDraw.Draw(image_gray_Im)
        draw_W = ImageDraw.Draw(image_W)
        draw_gray_W = ImageDraw.Draw(image_gray_W)
        draw_LV = ImageDraw.Draw(image_LV)
        draw_gray_LV = ImageDraw.Draw(image_gray_LV)

        # Loop through the features in the layer
        json_source = json.load(open(hr_label_path))
        for ftr in json_source.get('features'):
            att = ftr.get('properties')['default']

            for multipolygon in ftr['geometry']['coordinates']: #4D coordata
                for ply in multipolygon:
                    ply = np.array(ply)
                    loc = np.argmax(ply[:, 0])
                    v1 = ply[loc] - ply[loc - 1]
                    v2 = ply[loc + 1] - ply[loc - 1]

                    ply = (ply - [mapXmin, mapYmax]) * [1, -1] / [pixel_size_x, pixel_size_y]
                    ply = [(a[0], a[1]) for a in ply.tolist()]

                    if v1[0] * v2[1] - v1[1] * v2[0] >= 0:
                        if(att == 'Tree Canopy'):
                            color = (0, 128, 0); color2 = 1 #dark green
                            draw_TC.polygon(ply, fill = color, outline = None)
                            draw_gray_TC.polygon(ply, fill = color2, outline = None)
                        elif(att == 'Impervious'):
                            color = (255, 0, 0); color2 = 3 #red
                            draw_Im.polygon(ply, fill = color, outline = None)
                            draw_gray_Im.polygon(ply, fill = color2, outline = None)
                        elif(att == 'Water'):
                            color = (0, 225, 225); color2 = 0 #light blue
                            draw_W.polygon(ply, fill = color, outline = None)
                            draw_gray_W.polygon(ply, fill = color2, outline = None)
                        elif(att == 'Low Vegetation'):
                            color = (0, 255, 0); color2 = 2 #light green
                            draw_LV.polygon(ply, fill = color, outline = None)
                            draw_gray_LV.polygon(ply, fill = color2, outline = None)
                        else: raise ValueError('Wrong target class!')
                    else:
                        if(att == 'Tree Canopy'):
                            draw_TC.polygon(ply, fill = (0, 0, 0), outline = None)
                            draw_gray_TC.polygon(ply, fill = 0, outline = None)
                        elif(att == 'Impervious'):
                            draw_Im.polygon(ply, fill = (0, 0, 0), outline = None)
                            draw_gray_Im.polygon(ply, fill = 0, outline = None)
                        elif(att == 'Water'):
                            draw_W.polygon(ply, fill = (0, 0, 0), outline = None)
                            draw_gray_W.polygon(ply, fill = 0, outline = None)
                        elif(att == 'Low Vegetation'):
                            draw_LV.polygon(ply, fill = (0, 0, 0), outline = None)
                            draw_gray_LV.polygon(ply, fill = 0, outline = None)
                        else: raise ValueError('Wrong target class!')

        image_TC_np = np.array(image_TC)
        image_Im_np = np.array(image_Im)
        image_W_np = np.array(image_W)
        image_LV_np = np.array(image_LV)
        image_gray_TC_np = np.array(image_gray_TC)
        image_gray_Im_np = np.array(image_gray_Im)
        image_gray_W_np = np.array(image_gray_W)
        image_gray_LV_np = np.array(image_gray_LV)

        #prioritize in order of Im ->  LV ->  TC ->  W
        image_LV_np[np.sum(image_Im_np, axis = -1) > 0, :] = 0
        image_TC_np[np.sum(image_Im_np, axis = -1) > 0, :] = 0
        image_W_np[np.sum(image_Im_np, axis = -1) > 0, :] = 0
        image_gray_LV_np[image_gray_Im_np > 0] = 0
        image_gray_TC_np[image_gray_Im_np > 0] = 0
        image_gray_W_np[image_gray_Im_np > 0] = 0

        image_TC_np[np.sum(image_LV_np, axis = -1) > 0, :] = 0
        image_W_np[np.sum(image_LV_np, axis = -1) > 0, :] = 0
        image_gray_TC_np[image_gray_LV_np > 0] = 0
        image_gray_W_np[image_gray_LV_np > 0] = 0

        image_W_np[np.sum(image_TC_np, axis = -1) > 0, :] = 0
        image_gray_W_np[image_gray_TC_np > 0] = 0

        image_np = image_TC_np + image_Im_np + image_W_np + image_LV_np
        image_gray_np = image_gray_TC_np + image_gray_Im_np + image_gray_W_np + image_gray_LV_np
        np.save(output_path + "image_rgb-" + image_id_year, image_np)
        np.save(output_path + "image_gray-" + image_id_year, image_gray_np)

        return(image_np, image_gray_np)

    
    image_np, image_gray_np = make_save_image_np(nlcd_path, hr_label_path, output_path, image_id_year)
    image = Image.fromarray(image_np)
    image_gray = Image.fromarray(image_gray_np)

    print(f'\nsize of image: {image.size}', f'\nsize of image_gray: {image_gray.size}',
          f'\nsize of image -- rbg: {image_np.shape} as numpy array',
          f'\nsize of image_gray -- gray-scale: {image_gray_np.shape} as numpy array')
    print('\n---Distribution-------------------',
          '\n0: Water', '\n1: Tree Canopy', '\n2: Low Vegetation', '\n3: Impervious\n\n',
          pd.Series(image_gray_np.flatten()).value_counts())

    gdal_nlcd = gdal.Open(nlcd_path, gdal.GA_ReadOnly)
    nlcd_np = np.array([gdal_nlcd.GetRasterBand(i + 1).ReadAsArray() for i in range(gdal_nlcd.RasterCount)])

    fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    ax[0].imshow(image)
    ax[0].set_title("high-res label (rgb)")
    ax[1].imshow(image_gray, cmap = 'gray')
    ax[1].set_title("high-res label (gray-scale)")
    ax[2].imshow(nlcd_np[0], cmap = 'gray')
    ax[2].set_title("NLCD low-res label (gray-scale)")
    plt.show()

    image.save(output_path + "image_rgb-" + image_id_year + '.png')
    image.save(output_path + "image_gray-" + image_id_year + '.png')
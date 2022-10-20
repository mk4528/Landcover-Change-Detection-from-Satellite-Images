import pandas as pd 
import matplotlib.pyplot as plt
from skimage import io
import numpy as np
import geopandas

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


NLCD_IDX_TO_REDUCED_LC_MAP = np.array([
    4,#  0 No data 0
    0,#  1 Open Water
    4,#  2 Ice/Snow
    2,#  3 Developed Open Space
    3,#  4 Developed Low Intensity
    3,#  5 Developed Medium Intensity
    3,#  6 Developed High Intensity
    3,#  7 Barren Land
    1,#  8 Deciduous Forest
    1,#  9 Evergreen Forest
    1,# 10 Mixed Forest
    1,# 11 Shrub/Scrub
    2,# 12 Grassland/Herbaceous
    2,# 13 Pasture/Hay
    2,# 14 Cultivated Crops
    1,# 15 Woody Wetlands
    1,# 16 Emergent Herbaceious Wetlands
])


def get_image_ndarray(csv_path, col, row):
    df = pd.read_csv(csv_path)
    a = io.imread(df.loc[row, col])
    return a


def display_image(csv_path, col, row):
    """Display one tif image from csv file
    
    Args:
        csv_path: the path to the csv data file
        col: str, the name of the column
        row: int, row number

    Example:
        display_image("data/splits/val_inference_both.csv", "label_fn", 3)
    """
    a = get_image_ndarray(csv_path, col, row)
    plt.imshow(a)
    plt.show()


def get_tif_dimension(csv_path, col, row):
    """Get the dimension of a TIF image
    
    Args:
        csv_path: the path to the csv data file
        col: str, the column name 
        row: int, the row number

    Returns:
        the dimension of the TIF image
    """
    a = get_image_ndarray(csv_path, col, row)
    return a.shape


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
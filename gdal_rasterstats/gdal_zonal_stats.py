from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import functools
from osgeo import gdal, ogr
from osgeo.gdalconst import GA_ReadOnly

gdal.PushErrorHandler("CPLQuietErrorHandler")
gdal.UseExceptions()


def bbox_to_pixel_offsets(gt, bbox):
    originX = gt[0]
    originY = gt[3]
    pixel_width = gt[1]
    pixel_height = gt[5]
    x1 = int((bbox[0] - originX) / pixel_width)
    x2 = int((bbox[1] - originX) / pixel_width) + 1

    y1 = int((bbox[3] - originY) / pixel_height)
    y2 = int((bbox[2] - originY) / pixel_height) + 1

    xsize = x2 - x1
    ysize = y2 - y1
    return (x1, y1, xsize, ysize)


def gdal_zonalstats(
    raster_path: Path,
    vector_path: Path,
    nodata_value: bool = True,
    global_src_extent: bool = False,
    band_nb: int = 1,
):
    """
    Extracts the median value over all polygons from the vector layer on a band from the raster file.

    Parameters
    ----------
    raster_path : Path
        Path to the raster (tif) file
    vector_path : Path
        Path to the vector file (handles parquet files)
    nodata_value : bool, optional
        Defaults to True. If true, removes the no-data value from the raster file from the statistics
    global_src_extent : bool, optional
        Defaults to False. If True, reads the raster file in a single time (more pressure on the memory, but faster if reading is slow)
    band_nb : int, optional
        Defaults to 1. Band to read from the raster file.

    Returns
    -------
    dict
        Dictionnary containing the results on the band.
    """

    rds = gdal.Open(raster_path, GA_ReadOnly)
    assert rds
    rb = rds.GetRasterBand(band_nb)
    rgt = rds.GetGeoTransform()

    if nodata_value:
        nodata_value = float(rb.GetNoDataValue())
    else:
        nodata_value = None

    vds = ogr.Open(vector_path, GA_ReadOnly)
    assert vds
    vlyr = vds.GetLayer(0)

    # create an in-memory numpy array of the source raster data
    # covering the whole extent of the vector layer
    if global_src_extent:
        # use global source extent
        # useful only when disk IO or raster scanning inefficiencies are your limiting factor
        # advantage: reads raster data in one pass
        # disadvantage: large vector extents may have big memory requirements
        src_offset = bbox_to_pixel_offsets(rgt, vlyr.GetExtent())
        src_array = rb.ReadAsArray(*src_offset)

        # calculate new geotransform of the layer subset
        new_gt = (
            (rgt[0] + (src_offset[0] * rgt[1])),
            rgt[1],
            0.0,
            (rgt[3] + (src_offset[1] * rgt[5])),
            0.0,
            rgt[5],
        )

    mem_drv = ogr.GetDriverByName("Memory")
    driver = gdal.GetDriverByName("MEM")

    # Loop through vectors
    stats = []
    feat = vlyr.GetNextFeature()
    while feat is not None:
        if not global_src_extent:
            # use local source extent
            # fastest option when you have fast disks and well indexed raster (ie tiled Geotiff)
            # advantage: each feature uses the smallest raster chunk
            # disadvantage: lots of reads on the source raster
            src_offset = bbox_to_pixel_offsets(rgt, feat.geometry().GetEnvelope())
            src_array = rb.ReadAsArray(*src_offset)

            # calculate new geotransform of the feature subset
            new_gt = (
                (rgt[0] + (src_offset[0] * rgt[1])),
                rgt[1],
                0.0,
                (rgt[3] + (src_offset[1] * rgt[5])),
                0.0,
                rgt[5],
            )

        # Create a temporary vector layer in memory
        mem_ds = mem_drv.CreateDataSource("out")
        mem_layer = mem_ds.CreateLayer("poly", None, ogr.wkbPolygon)
        mem_layer.CreateFeature(feat.Clone())

        # Rasterize it
        rvds = driver.Create("", src_offset[2], src_offset[3], 1, gdal.GDT_Byte)
        rvds.SetGeoTransform(new_gt)
        gdal.RasterizeLayer(rvds, [1], mem_layer, burn_values=[1])
        rv_array = rvds.ReadAsArray()

        # Mask the source data array with our current feature
        # we take the logical_not to flip 0<->1 to get the correct mask effect
        # we also mask out nodata values explictly
        masked = np.ma.MaskedArray(
            src_array,
            mask=np.logical_or(src_array == nodata_value, np.logical_not(rv_array)),
        )

        feature_stats = {
            "count": int(masked.count()),
            "tot_count": int(rv_array.sum()),
            "median": float(np.median(masked.data[~masked.mask])),
            "pol_id": feat.GetField(0),
        }

        stats.append(feature_stats)

        rvds = None
        mem_ds = None
        feat = vlyr.GetNextFeature()

    vds = None
    rds = None
    return stats


def _gdal_extract_band(
    band_nb: int,
    raster_path: Path,
    vector_path: Path,
    band_names: list,
    min_data_perc: float,
    **kwargs,
) -> pd.DataFrame:
    """
    Extracts median statistics on a single band for a given tif file and shapefile, using gdal for speed.

    Parameters
    ----------
    band_nb : int
        band number to open (from 1 to total bands), corresponding value in list handled.
    raster_path : Path
        Path to the tif file
    vector_path : Path
        Path to shapefile containing zones of interest
    band_names : list
        List of band names to associate with the tif file.
    min_data_perc : float
        Mimimum fraction of available data over the polygon, by default 0.6.
        If a polygon has less available data, it is transformed into a nan.

    Returns
    -------
    pd.DataFrame
        Median values over the polygon for the given band.
    """

    col_name = band_names[band_nb - 1]

    stats = gdal_zonalstats(
        raster_path,
        vector_path,
        nodata_value=True,
        band_nb=int(band_nb),
        **kwargs,
    )

    # Store results in a dataframe
    df_temp = pd.DataFrame(stats).set_index("pol_id")
    # Add a pixel percentage and select only parcels above threshold
    df_temp["pix_perc"] = df_temp["count"] / df_temp["tot_count"]
    df_temp.loc[df_temp["pix_perc"] < min_data_perc, "median"] = np.nan
    # Keep only median value and rename it
    stats_df = df_temp[["median"]].rename(columns={"median": col_name})
    stats_df.index.name = None

    return stats_df


def gdal_extract_median_stats(
    tif_path: Path,
    shp_path: Path,
    band_names: list,
    min_data_perc: float = 0.6,
    num_workers: int = 16,
    **kwargs,
) -> pd.DataFrame:
    """
    Extracts median statistics for each band of a tif file.
    Multi-threaded to increase speed.
    Parameters
    ----------
    tif_path : Path
        Path to the tif file
    shp_path : Path
        Path to the segmentation file
    band_names : list
        List of names to the bands in the raster file.
    min_data_perc : float, optional
        Mimimum fraction of available data over the polygon, by default 0.6.
        If a polygon has less available data, it is transformed into a nan.
    num_workers : int, optional
        Number of concurrent workers, defaults to 16
    Returns
    -------
    pd.DataFrame
        DataFrame with median statistics for each polygon and band/date.
    """

    raster = gdal.Open(tif_path)
    num_bands = raster.RasterCount

    processor = functools.partial(
        _gdal_extract_band,
        raster_path=tif_path,
        vector_path=shp_path,
        band_names=band_names,
        min_data_perc=min_data_perc,
        **kwargs,
    )

    stats_list = []
    loop_over = np.arange(1, num_bands + 1)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for df_meds in tqdm(
            executor.map(processor, loop_over),
            total=loop_over.size,
            desc="Extracting parcel statistics",
            position=0,
        ):
            stats_list.append(df_meds)

    out_df = pd.concat(stats_list, axis=1)

    return out_df

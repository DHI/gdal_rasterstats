### General information

This package is built upon a code snippet written by Perrygeo at https://gist.github.com/perrygeo/5667173

This package contains simple functions to extract median statistics over polygons on the bands of a rasterfile, using only GDAL implementations, as well as parralelization over the raster bands. This package is a faster alternative to the rasterstats package, going much faster, especially on large datasets.

So far, only the median over each polygon is returned, but the list of available statistics can be expanded. Importantly, this package requires the raster and vector files to be defined in the same geographic projection. An automatic reprojection will be added soon.

### How to use

```python
from gdal_rasterstats import gdal_extract_median_stats

band_names =  ["2024-01-01", "2024-02-01"]  # The length of this vector must be the same as the number of bands in the tif file, and defines the column names in the output dataframe.
df_stats = gdal_extract_median_stats("path/to/raster.tif", "path/to/vector.shp", band_names, num_workers=16)
```

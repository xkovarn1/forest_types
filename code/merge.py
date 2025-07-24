import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import numpy as np

# File paths
input_raster_path = '/dataset.tif'
output_raster_path = '/merged.tif'

# Load TIFF
with rasterio.open(input_raster_path) as src:
    meta = src.meta.copy()
    transform = src.transform
    raster_crs = src.crs
    width, height = src.width, src.height
    data = src.read()

# Load shp with target classes
gdf = gpd.readfile('final_roi_data.shp')

# Align CRS (gdf is shapefile with ROI)
gdf = gdf.to_crs(raster_crs)

# Map classess
classes = gdf['SLT2_grouped'].unique()
class_to_int = {cls: i + 1 for i, cls in enumerate(classes)}  # 0 is background

# Prepare shapefiles
shapes = [(geom, class_to_int[attr]) for geom, attr in zip(gdf.geometry, gdf['SLT2_grouped'])]

# Rasterize shape to mask
mask = rasterize(
    shapes=shapes,
    out_shape=(height, width),
    transform=transform,
    fill=0,  # pozadí
    dtype='uint8'
)

# Add new band as the last layer
new_data = np.vstack([data, mask[np.newaxis, :, :]])

# Update metadata
meta.update({
    'count': new_data.shape[0],
    'dtype': new_data.dtype
})

# Save new tiff
with rasterio.open(output_raster_path, 'w', **meta) as dst:
    dst.write(new_data)

print(f'Done! Output saved as: {output_raster_path}')
print(f'Class mapping: {class_to_int}')

// =======================================================
// 1. Define Area of Interest (AOI)
// =======================================================
var roi = table;  // Your region of interest (a FeatureCollection)

// =======================================================
// 2. Load Canopy Height Map (1m Global)
// =======================================================
var canopy = ee.ImageCollection("projects/meta-forest-monitoring-okw37/assets/CanopyHeight")
              .mosaic()
              .clip(roi)
              .rename('CanopyHeight');

// =======================================================
// 3. Load Sentinel-2 SR and Cloud Mask
// =======================================================
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(roi)
  .filterDate('2020-01-01', '2020-12-31')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10));

// Cloud mask using Scene Classification Layer (SCL)
function maskS2(image) {
  var scl = image.select('SCL');
  // Mask cloud shadows (3), clouds (8), cirrus (9), snow (10)
  var mask = scl.neq(3).and(scl.neq(8)).and(scl.neq(9)).and(scl.neq(10));
  return image.updateMask(mask);
}
s2 = s2.map(maskS2);

// =======================================================
// 4. Vegetation Indices (including red edge indices)
// =======================================================
function addIndices(image) {
  var ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI');
  var evi = image.expression(
    '2.5 * ((NIR - RED) / (NIR + 6 * RED - 7.5 * BLUE + 1))', {
      'NIR': image.select('B8'),
      'RED': image.select('B4'),
      'BLUE': image.select('B2')
    }).rename('EVI');
  var nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR');
  var ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI');
  var ndmi = image.normalizedDifference(['B8', 'B11']).rename('NDMI');
  var savi = image.expression(
    '((NIR - RED) / (NIR + RED + 0.5)) * 1.5', {
      'NIR': image.select('B8'),
      'RED': image.select('B4')
    }).rename('SAVI');
  var gndvi = image.normalizedDifference(['B8', 'B3']).rename('GNDVI');
  var reci = image.expression(
    '(NIR / RED) - 1', {
      'NIR': image.select('B8'),
      'RED': image.select('B4')
    }).rename('RECI');
  var ndre = image.normalizedDifference(['B8', 'B5']).rename('NDRE');
  var rendvi = image.normalizedDifference(['B8A', 'B5']).rename('RENDVI');
  var mtci = image.expression(
    '(B6 - B5) / (B6 - B4)', {
      'B4': image.select('B4'),
      'B5': image.select('B5'),
      'B6': image.select('B6')
    }).rename('MTCI');
  var msr = image.expression(
    '(NIR / RED - 1) / sqrt(NIR / RED + 1)', {
      'NIR': image.select('B8'),
      'RED': image.select('B4')
    }).rename('MSR');

  return image.addBands([ndvi, evi, nbr, ndwi, ndmi, savi, gndvi, reci,
                         ndre, rendvi, mtci, msr]);
}
var s2_indices = s2.map(addIndices);

// =======================================================
// 5. Median & Percentile NDVI
// =======================================================
var s2_median = s2.median().clip(roi);
var indices_median = s2_indices.median().clip(roi);
var ndvi_p90 = s2_indices.select('NDVI').reduce(ee.Reducer.percentile([90])).rename('NDVI_p90');
var ndvi_std = s2_indices.select('NDVI').reduce(ee.Reducer.stdDev()).rename('NDVI_stdDev');

// =======================================================
// 6. Seasonal Composites for Sentinel-2 Indices
// =======================================================
function seasonComposite(start, end, name) {
  var composite = s2_indices
    .filterDate(start, end)
    .median()
    .select([
      'NDVI', 'EVI', 'NBR', 'NDWI', 'NDMI', 'SAVI', 'GNDVI', 'RECI',
      'NDRE', 'RENDVI', 'MTCI', 'MSR'
    ])
    .rename([
      name + '_NDVI', name + '_EVI', name + '_NBR', name + '_NDWI',
      name + '_NDMI', name + '_SAVI', name + '_GNDVI', name + '_RECI',
      name + '_NDRE', name + '_RENDVI', name + '_MTCI', name + '_MSR'
    ]);
  return composite;
}
var spring = seasonComposite('2020-03-01', '2020-05-31', 'spring');
var summer = seasonComposite('2020-06-01', '2020-08-31', 'summer');
var autumn = seasonComposite('2020-09-01', '2020-11-30', 'autumn');

// =======================================================
// 7. Load Sentinel-1 SAR (VV, VH), merge orbits, add indices
// =======================================================
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(roi)
  .filterDate('2020-01-01', '2020-12-31')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH'));

var s1_asc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'));
var s1_desc = s1.filter(ee.Filter.eq('orbitProperties_pass', 'DESCENDING'));
var s1_combined = s1_asc.merge(s1_desc);

// Add SAR indices to an image
// Sentinel-1 VV and VH are in dB scale (10*log10 sigma0)
// Computed indices are in dB or ratio
function addS1Indices(image) {
  var vv = image.select('VV');
  var vh = image.select('VH');

  var vh_vv_ratio = vh.divide(vv).rename('VH_VV_ratio');
  var vh_minus_vv = vh.subtract(vv).rename('VH_minus_VV');
  var vh_plus_vv = vh.add(vv).rename('VH_plus_VV');

  return image.addBands([vh_vv_ratio, vh_minus_vv, vh_plus_vv]);
}

// Create seasonal composites for Sentinel-1
function seasonCompositeS1(start, end, name) {
  var composite = s1_combined
    .filterDate(start, end)
    .select(['VV', 'VH'])  // <== Ensure only these bands
    .median()
    .clip(roi);
  return addS1Indices(composite).rename([
    name + '_VV', name + '_VH', name + '_VH_VV_ratio',
    name + '_VH_minus_VV', name + '_VH_plus_VV'
  ]);
}


var s1_spring = seasonCompositeS1('2020-03-01', '2020-05-31', 's1_spring');
var s1_summer = seasonCompositeS1('2020-06-01', '2020-08-31', 's1_summer');
var s1_autumn = seasonCompositeS1('2020-09-01', '2020-11-30', 's1_autumn');

var s1_median = s1_combined
  .select(['VV', 'VH'])
  .median()
  .clip(roi);
var s1_median_withIndices = addS1Indices(s1_median);

// =======================================================
// 8. Terrain (DEM, slope, aspect)
// =======================================================
var dem = ee.Image('USGS/SRTMGL1_003').clip(roi);
var terrain = ee.Terrain.products(dem);
var elevation = terrain.select('elevation');
var slope = terrain.select('slope');
var aspect = terrain.select('aspect');

// =======================================================
// 9. Texture Features (NDVI, Canopy Height, Sentinel-1 VV/VH)
// =======================================================
// NDVI: scale -1..1 → -1000..1000
var ndvi_int = indices_median.select('NDVI').multiply(1000).toInt32();
var ndviTexture = ndvi_int.glcmTexture({size: 3});

// CanopyHeight: 0-40 m → ×100
var canopy_int = canopy.select('CanopyHeight').multiply(100).toInt32();
var canopyTexture = canopy_int.glcmTexture({size: 3});

// Sentinel-1 VV and VH textures (scale dB to int)
var vv_int = s1_median_withIndices.select('VV').multiply(1000).toInt32();
var vh_int = s1_median_withIndices.select('VH').multiply(1000).toInt32();
var vvTexture = vv_int.glcmTexture({size: 3});
var vhTexture = vh_int.glcmTexture({size: 3});

// =======================================================
// 10. Stack all layers including Sentinel-1 bands and indices
// =======================================================
var stacked = canopy
  .addBands(s2_median.select(['B2','B3','B4','B5','B6','B7','B8','B8A','B11','B12']))
  .addBands(indices_median.select(['NDVI', 'EVI', 'NBR', 'NDWI', 'NDMI', 'SAVI', 'GNDVI', 'RECI', 'NDRE', 'RENDVI', 'MTCI', 'MSR']))
  .addBands(ndvi_p90)
  .addBands(ndvi_std)
  .addBands(spring)
  .addBands(summer)
  .addBands(autumn)
  .addBands(s1_median_withIndices)          // Sentinel-1 median + indices
  .addBands(s1_spring)
  .addBands(s1_summer)
  .addBands(s1_autumn)
  .addBands(elevation)
  .addBands(slope)
  .addBands(aspect)
  .addBands(canopyTexture)
  .addBands(ndviTexture)
  .addBands(vvTexture)
  .addBands(vhTexture)
  .unmask(0)      // Fill NaNs with zero
  .toFloat()      // Uniform data type
  .clip(roi);

// =======================================================
// 11. Visualization
// =======================================================
Map.centerObject(roi, 13);
Map.addLayer(stacked.select('NDVI'), {min: 0, max: 1, palette: ['white', 'green']}, 'NDVI');
Map.addLayer(canopy, {min: 0, max: 40, palette: ['yellow', 'green']}, 'Canopy Height');
Map.addLayer(stacked.select(['VV', 'VH']), {min: -25, max: 0}, 'Sentinel-1 VV & VH');

// Print band names for inspection
print('Band names:', stacked.bandNames());

// =======================================================
// 12. Export example (uncomment and customize as needed)
// =======================================================
Export.image.toDrive({
  image: stacked.clip(roi),
  description: 'Sentinel2_S1_SLP_KOMPLEX_2020',
  folder: 'GEE_exports',
  region: roi.geometry(),
  scale: 10,  // Sentinel-2 resolution
  maxPixels: 1e13,
  fileFormat: 'GeoTIFF'
});

# sar-landslide-seg

Fine-tunes an **ImageNet-pretrained segmentation network on multi-band satellite data** for two jobs:

1. **Landslide mapping** on [Landslide4Sense](https://huggingface.co/datasets/ibm-nasa-geospatial/Landslide4sense): 14 bands (Sentinel-2 B1–B12 + ALOS slope + DEM), 128×128 patches, pixel labels.
2. **Bi-temporal SAR change detection** on your own GeoTIFF pairs (Sentinel-1 or NISAR, before/after an event), which feeds directly into landslide early-warning and SAR change-detection work.

The first conv layer is adapted from 3 to N bands, the encoder warms up frozen for 2 epochs, and then the whole network fine-tunes with a lower encoder learning rate. Loss is BCE + Dice with a positive-class weight, because landslide pixels are rare.

## Run on Kaggle

Push to `github.com/Sidvortex/sar-landslide-seg`, import `kaggle_notebook.ipynb`, enable GPU and Internet, and run all. The first run downloads Landslide4Sense (a few GB). Training takes about 1 h for 40 epochs on a T4.

```bash
python train.py --config configs/landslide4sense.yaml
python train.py --set model.encoder=efficientnet-b3 train.output_dir=outputs/l4s-effb3
python train.py --set model.arch=UnetPlusPlus train.output_dir=outputs/l4s-upp
python train.py --set "data.bands=[1,2,3,7,12,13]" train.output_dir=outputs/l4s-rgbnir-slope-dem   # band ablation
```

`outputs/<run>/history.json` holds per-epoch val IoU, F1, precision and recall. `best.pt` is the best-IoU checkpoint, and `band_stats.json` holds the normalisation used at inference.

## Predict (works on CPU)

```bash
pip install --break-system-packages -r requirements.txt
python predict.py --ckpt outputs/l4s-unet-r34/best.pt --h5 image_12.h5 --out pred.png
python predict.py --ckpt outputs/l4s-unet-r34/best.pt --tif scene_14band.tif --out mask.tif
python predict.py --ckpt outputs/sar-change-unet-r34/best.pt --pre pre.tif --post post.tif --out change.tif
```

Large scenes are processed with overlapping, blended tiles. GeoTIFF outputs keep the input's CRS and transform (band 1 is probability, band 2 is the binary mask), so they open directly in QGIS or a web map.

## Bring your own SAR pairs

```
data/pairs/train/<scene_id>/pre.tif    # C bands, e.g. VV,VH linear backscatter, before the event
data/pairs/train/<scene_id>/post.tif   # same bands, same grid, after
data/pairs/train/<scene_id>/mask.tif   # 1 band, 1 = changed
data/pairs/val/<scene_id>/...
```

- Export co-registered Sentinel-1 GRD (VV, VH) before/after pairs from Google Earth Engine or ASF. NISAR L-band products follow the same pattern once you resample them to a common grid.
- Masks can come from mapped landslide or flood inventories (e.g. state disaster agency polygons rasterised onto the grid).
- With `data.sar: true`, inputs are converted to dB and a log-ratio channel (post − pre in dB) is added, which is the classic SAR change indicator. So the network sees `[pre, post, change]` = 3×C channels.
- Training samples 256×256 tiles and biases half of them toward changed areas.

```bash
python train.py --config configs/sar_change_pairs.yaml --set data.root=/kaggle/input/<your-dataset>
```

## Metrics to report

IoU and F1 for the landslide class, plus precision and recall. For early warning, recall usually matters more, so try `--threshold 0.3` in `predict.py`. Compare against the published Landslide4Sense competition results for context.

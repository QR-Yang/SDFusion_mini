# SDFusion Mini

This is a simplified SDFusion-style pipeline for class-conditioned ShapeNet SDF generation.

```text
ShapeNet mesh -> SDF preprocessing -> VQ-VAE -> latent diffusion -> marching cubes obj
```

## Classes

```text
chair    03001627 -> 0
table    04379243 -> 1
car      02958343 -> 2
rifle    04090263 -> 3
airplane 02691156 -> 4
```

## Prepare SDF Data

Raw ShapeNet meshes should be available through:

```text
data/ShapeNet/ShapeNetCore.v1/<synset>/<model_id>/model.obj
```

Run preprocessing from the `preprocess` directory. Example:

```bash
cd /data/yangqr/test/SDFusion_mini/preprocess
conda activate final
LD_LIBRARY_PATH=$CONDA_PREFIX/lib:./isosurface:./isosurface/tbb/tbb2018_20180822oss/lib/intel64/gcc4.7 \
python create_sdf.py --dset shapenet --category chair --reduce 4
```

Replace `chair` with `table`, `car`, `rifle`, or `airplane` for other classes.

Generated SDF files are saved under:

```text
data/ShapeNet/SDF_v1/resolution_64/
```

## Check Data

```bash
cd /data/yangqr/test/SDFusion_mini
conda activate final
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python datasets.py
```

Expected SDF tensor shape:

```text
torch.Size([1, 64, 64, 64])
```

## Train

```bash
cd /data/yangqr/test/SDFusion_mini
conda activate final
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python train_vqvae.py
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python train_diffusion.py
```

Checkpoints are written to:

```text
checkpoints/vqvae.pth
checkpoints/class_diffusion.pth
```

## Sample

Set `class_name` in `sample.py` to one of:

```text
chair, table, car, rifle, airplane
```

Then run:

```bash
cd /data/yangqr/test/SDFusion_mini
conda activate final
LD_LIBRARY_PATH=$CONDA_PREFIX/lib python sample.py
```

Generated meshes are saved to `outputs/`.

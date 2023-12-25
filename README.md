# CoordNet: Data Generation and Visualization Generation for Time-Varying Volumes via a Coordinate-Based Neural Network
Pytorch implementation for CoordNet: Data Generation and Visualization Generation for Time-Varying Volumes via a Coordinate-Based Neural Network

## Prerequisites
- Linux
- CUDA >= 10.0
- Python >= 3.7
- Numpy
- Skimage
- Pytorch >= 1.0

## Data format

The volume at each time step is saved as a .dat file with the little-endian format. The data is stored in column-major order, that is, z-axis goes first, then y-axis, finally x-axis.

## Training models
```
cd Code 
```

- training
```
python3 main.py --mode 'train' --dataset 'Vortex' --applicaion 'temporal'
```

- inference
```
python3 main.py --mode 'inf' --dataset 'Vortex' --application 'temporal'
```

## Citation 
```
@article{han2023coordnet,
  title={CoordNet: Data Generation and Visualization Generation for Time-Varying Volumes via a Coordinate-Based Neural Network},
  author={Han, Jun and Wang, Chaoli},
  journal={IEEE Transactions on Visualization and Computer Graphics},
  volume={29},
  number={12},
  pages={4951--4963},
  year={2023}
}

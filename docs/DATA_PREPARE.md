<img src="./figs/ResLPR_logo.png" align="right" width="20%">

# Data Preparation

### Overall Structure

```
└── ResLPR 
    └── WeatherKITTI    
         │── Snow      
         │   │── train                
         │   │    │── light
         │   │    │     ├── 03
         │   │    │     │   ├── snow_velodyne
         │   │    │     │   └── snow_label
         │   │    │     ├── 04
         │   │    │     ├──...
         │   │    │     └── 10
         │   │    │── mod 
         │   │    └── heavy
         │   └── test
         │        │── light
         │        │     └── 00
         │        │         └── snow_velodyne
         │        ├── mod
         │        └── heavy                                                             
         │── Fog                          
         └── Rain
    └── WeatherNCLT
         ├── Snow
         │   ├── train
         │   │    └── 2012-01-22
         │   │          ├── snow_velodyne
         │   │          └── snow_label  
         │   └── test
         │        ├── light
         │        │     └── 2012-02-12
         │        │           └── snow_velodyne
         │        ├── mod
         │        └── heavy
         ├── Fog 
         └── Rain
    └── Clean(Not provided)
         ├── KITTI
         │    ├── 00
         │    │    └── velodyne
         │    ├── 03
         │    ├── 04
         │    ├──...
         │    └── 10
         └── NCLT
              ├── 2012-01-08_vel
              │    └── velodyne_sync
              ├── 2012-01-22_vel 
              ├── 2012-02-12_vel
              ├── 2012-06-15_vel
              ├── 2012-08-04_vel
              ├── 2012-11-04_vel
              └── 2012-11-16_vel
```
## KITTI
WeatherKITTI is simulated based on sequence 00 and sequences 03 to 10 in the odometry dataset of KITTI. To reproduce the results of ResLPR, it is necessary to use the original **sequence 00 as the database** and the point clouds of adverse weather conditions as the query for experiments. Please download the original dataset of sequence 00 by yourself: [KITTI odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php).

## NCLT
WeatherNCLT is simulated based on multiple sequences in the NCLT dataset. To reproduce the results of ResLPR, it is necessary to use the original 2012 - 01 - 08 sequence as the database and the point clouds of adverse weather conditions as the query for experiments. To obtain the corresponding LPR results under clear weather conditions, you also need to download other corresponding sequences. Please download the original NCLT dataset by yourself: [NCLT download](https://robots.engin.umich.edu/nclt/index.html#download).

## WeatherKITTI & WeatherNCLT
One can download WeatherKITTI and WeatherNCLT via the provided link: [ResLPR datasets](https://example_baidu.com). It is advisable to organize these datasets in accordance with the format specified in the Overall Structure.


## References

Please note that you should cite the corresponding paper(s) once you use these datasets.


### KITTI

```bibtex
@inproceedings{geiger2012kitti,
    author = {A. Geiger and P. Lenz and R. Urtasun},
    title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
    pages = {3354--3361},
    year = {2012}
}
```

### NCLT
```bibtex
@article{carlevaris2016university,
  title={University of Michigan North Campus long-term vision and lidar dataset},
  author={Carlevaris-Bianco, Nicholas and Ushani, Arash K and Eustice, Ryan M},
  journal={The International Journal of Robotics Research},
  volume={35},
  number={9},
  pages={1023--1035},
  year={2016},
  publisher={Sage Publications Sage UK: London, England}
}
```

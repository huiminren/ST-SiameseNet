# ST-SiameseNet
<p align="center">
<img src="img/intro_driver_id.png" width="55%" height="80%">
<img src="img/overall_frame_driver_id.png" width="40%" height="20%">
</p>

## About
Source code of the paper: [ST-SiameseNet: Spatio-Temporal Siamese Networks for Human Mobility Signature Identification](https://dl.acm.org/doi/pdf/10.1145/3394486.3403183)
## Requirements
* Python >= 3.6
* `tensorflow >= 2.0.0`
* `keras >= 2.0.0`

## Usage
### Installation
#### Clone this repo:
```bash
git clone https://github.com/huiminren/ST-SiameseNet.git
cd ST-SiameseNet
```
#### Install Packages
For pip users, please type the command `pip install -r requirements.txt`.
#### Dataset
We provide 500 drivers as sample data. If you need full dataset, please feel free to contact us.
#### Running
  `python main.py`

## Citation
If you find this repo useful and would like to cite it, citing our paper as the following will be really appropriate: <br>
```
@inproceedings{ren2020st,
  title={ST-SiameseNet: Spatio-Temporal Siamese Networks for Human Mobility Signature Identification},
  author={Ren, Huimin and Pan, Menghai and Li, Yanhua and Zhou, Xun and Luo, Jun},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={1306--1315},
  year={2020}
}
```

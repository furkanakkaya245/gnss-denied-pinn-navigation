# GNSS-Denied Navigation with Physics-Informed Neural Networks

A physics-informed neural network (PINN) framework for aerial navigation in GNSS-denied environments, using IMU, magnetometer, and PPK ground truth data from the [MUN-FRL Bell 412 dataset](https://mun-frl-vil-dataset.readthedocs.io/en/latest/).

## Problem

In GNSS-denied environments (urban canyons, jamming, tunnels), inertial navigation systems suffer from unbounded drift. This work investigates whether physics-informed neural networks can maintain positional accuracy using only IMU and magnetometer data during GNSS outages.

## Architectures

| Model | File | Description |
|-------|------|-------------|
| PINN Single | `models/pinn_fast.py` | Single dataset, GRU+physics loss |
| PINN Multi | `models/pinn_multi.py` | Multi-dataset training (DS1+DS3→DS4) |
| PINN Delta | `models/pinn_delta2.py` | Coordinate-independent delta position |
| GRU Fusion | `models/gru_test.py` | GRU with GNSS/IMU fusion |
| Advanced Fusion | `models/new_model.py` | GRU+Attention with physics loss |

## Dataset

[MUN-FRL: Aerial Visual-Inertial-LiDAR Dataset](https://mun-frl-vil-dataset.readthedocs.io/en/latest/)

- **Platform:** Bell 412 Helicopter + DJI M600
- **Sensors:** IMU (400Hz), Magnetometer (100Hz), RTK-GNSS (5Hz)
- **Ground Truth:** Post-processed kinematic (PPK) GNSS
- **License:** CC BY 4.0

## Project Structure# gnss-denied-pinn-navigation
gnss-denied-pinn-navigation/
├── models/
│   ├── pinn_fast.py          # Single dataset PINN (10Hz, CPU optimized)
│   ├── pinn_multi.py         # Multi-dataset PINN
│   ├── pinn_delta2.py        # Delta position PINN (coordinate-independent)
│   ├── gru_test.py           # GRU fusion baseline
│   └── new_model.py          # Advanced GRU+Attention model
├── utils/
│   ├── extract_bag_v2.py     # ROS bag → CSV extractor
│   └── adjusment.py          # IMU bias estimation (block adjustment)
├── results/                  # Output figures
├── data/                     # Dataset placeholder
├── requirements.txt
└── README.md
##  Installation

```bash
git clone https://github.com/furkanakkaya245/gnss-denied-pinn-navigation
cd gnss-denied-pinn-navigation
pip install -r requirements.txt
```

##  Usage

### 1. Extract ROS Bag

```bash
python utils/extract_bag_v2.py --bag /path/to/bell412_dataset1.bag
```

Outputs: `*_imu.csv`, `*_mag.csv`, `*_ppk_gnss.csv`

### 2. IMU Bias Estimation

```bash
python utils/adjusment.py
```

### 3. Train Single Dataset PINN

```bash
python models/pinn_fast.py
```

### 4. Train Multi-Dataset PINN

```bash
python models/pinn_multi.py
```

### 5. Train Delta Position PINN

```bash
python models/pinn_delta2.py
```

##  Results

### Single Dataset (DS1 train → DS1 test)

| Metric | Value |
|--------|-------|
| ATE (full trajectory) | 8.19 m |
| ATE (GNSS outage) | 40.96 m |
| RMSE E | — |
| RMSE N | — |

### Multi-Dataset (DS1+DS3 train → DS4 test)

| Metric | Value |
|--------|-------|
| ATE (full trajectory) | 480.58 m |
| ATE (GNSS outage) | 1885.43 m |

The significant performance drop in cross-dataset generalization reveals the core research challenge: **PINN-based inertial navigation models trained on specific flight dynamics fail to generalize across different flight profiles.**

## Physics Constraints

All PINN models incorporate the following physical constraints:

- **Velocity-position consistency:** `Δpos = vel × Δt`
- **Speed limit:** helicopter max speed ~80 m/s
- **Coriolis effect:** `f = 2Ω sin(lat)`
- **Newton's law:** velocity change proportional to acceleration

## Known Limitations

- Models overfit to training flight dynamics
- Cross-dataset generalization degrades significantly
- Coordinate-dependent models (absolute position) generalize poorly
- Delta position approach (pinn_delta2.py) partially addresses generalization

## Future Work

- [ ] Domain adaptation for cross-flight generalization
- [ ] Uncertainty estimation (Bayesian PINN)
- [ ] Quantum inertial sensor simulation and comparison
- [ ] Extended Kalman Filter hybrid architecture
- [ ] Real-time deployment optimization

## Data Paths

Update dataset paths in each script before running:

```python
IMU_PATHS = {i: f'YOUR_PATH/bell412_dataset{i}_imu.csv' for i in [1,3,4]}
MAG_PATHS = {i: f'YOUR_PATH/bell412_dataset{i}_mag.csv' for i in [1,3,4]}
PPK_PATHS = {i: f'YOUR_PATH/dataset{i}/bell412_dataset{i}_ppk.pos' for i in [1,3,4]}
```

## Citation

If you use this work, please also cite the MUN-FRL dataset:

```bibtex
@article{thalagala2024munfrl,
  title={MUN-FRL: A Visual-Inertial-LiDAR Dataset for Aerial Autonomous Navigation and Mapping},
  author={Thalagala, Ravindu G and De Silva, Oscar and Mann, George KI and Gosine, Raymond G},
  journal={The International Journal of Robotics Research},
  volume={43},
  number={12},
  pages={1853--1866},
  year={2024},
  doi={10.1177/02783649241238318}
}
```

## Author

**Furkan Akkaya**
GitHub: [@furkanakkaya245](https://github.com/furkanakkaya245)
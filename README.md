# Boundary DoU++
This repo holds code for Boundary Difference Over Union Loss++ For Lung Nodule Segmentation.

## Usage

### Environment

#### Poetry

Make sure to use Poetry version 1.8.3

```bash
poetry shell
poetry install
```

#### Python venv

Make sure to use Python 3.11 or higher

```bash
pip install .
```

### Run Example

```bash
python example.py
```

## Results

The performance metrics are computed on the LIDC dataset across three different subsets:
1. **Whole dataset** – includes all samples.
2. **GE MEDICAL SYSTEMS without contrast subset** – includes only images acquired with GE MEDICAL SYSTEMS scanner but without contrast enhancement.
3. **GE MEDICAL SYSTEMS with contrast subset** – includes images acquired with GE MEDICAL SYSTEMS scanner and with contrast enhancement.

### Performance Metrics on the Whole Dataset

| Loss Function           | IoU      | SDSC     | Best Threshold |
|-------------------------|---------|---------|----------------|
| Boundary DoU++         | -       | -       | -              |
| Boundary DoU + Focal   | 0.484745 | **0.449673** | **0.6**  |
| Boundary DoU           | **0.488049** | 0.443682 | **0.6**  |
| Jaccard + BCE         | 0.465766 | 0.443107 | 0.9          |
| Dice                 | 0.468999 | 0.439383 | 0.8          |
| Dice + BCE           | 0.468966 | 0.437939 | 0.9          |
| Jaccard              | 0.451531 | 0.428241 | 0.9          |
| Focal                | 0.323004 | 0.263585 | 0.3          |

### Performance Metrics on GE MEDICAL SYSTEMS Without Contrast Subset

| Loss Function                           | IoU      | SDSC     | Best Threshold |
|-----------------------------------------|---------|---------|----------------|
| Boundary DoU                           | **0.478895** | **0.394080** | 0.7          |
| Boundary DoU++                         | 0.456137 | 0.382882 | **0.6**  |
| Dice + BCE                             | 0.410037 | 0.371589 | 0.9          |
| Jaccard + BCE                          | 0.420577 | 0.371133 | 0.9          |
| Boundary DoU++ (trained on contrast)   | 0.410795 | 0.349280 | **0.4**  |
| Boundary DoU (trained on contrast)     | 0.434197 | 0.346120 | **0.4**  |
| Boundary DoU + Focal                   | 0.450610 | 0.341621 | 0.9          |
| Jaccard + BCE (trained on contrast)    | 0.436914 | 0.338295 | 0.7          |
| Dice + BCE (trained on contrast)       | 0.376973 | 0.323151 | 0.9          |
| Boundary DoU + Focal (trained on contrast) | 0.426341 | 0.307982 | 0.7          |

**Note:** Models marked as "(trained on contrast)" were trained using the GE MEDICAL SYSTEMS with contrast subset.

### Performance Metrics on GE MEDICAL SYSTEMS With Contrast Subset

| Loss Function                           | IoU      | SDSC     | Best Threshold |
|-----------------------------------------|---------|---------|----------------|
| Boundary DoU++ (trained on no contrast) | 0.445058 | **0.486868** | **0.4**  |
| Boundary DoU++                         | **0.465802** | 0.474011 | **0.4**  |
| Boundary DoU + Focal (trained on no contrast) | 0.432991 | 0.470615 | **0.6**  |
| Boundary DoU                           | 0.457118 | 0.467075 | **0.4**  |
| Boundary DoU (trained on no contrast)   | 0.392872 | 0.451000 | **0.6**  |
| Dice + BCE (trained on no contrast)    | 0.393544 | 0.449920 | 0.9          |
| Jaccard + BCE                          | 0.436778 | 0.447341 | 0.2          |
| Boundary DoU + Focal                   | 0.428950 | 0.443293 | **0.4**  |
| Dice + BCE                             | 0.410969 | 0.442941 | 0.1          |
| Jaccard + BCE (trained on no contrast) | 0.410136 | 0.429173 | 0.9          |

**Note:** Models marked as "(trained on no contrast)" were trained using the GE MEDICAL SYSTEMS without contrast subset.

## References

[1] Sun, F., Luo, Z., Li, S.: "Boundary Difference over Union Loss for Medical Image Segmentation." In: International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 292–301 (2023). Springer.

[2] Yeung, M., Rundo, L., Nan, Y., Sala, E., Schönlieb, C.-B., Yang, G.: "Calibrating the Dice Loss to Handle Neural Network Overconfidence for Biomedical Image Segmentation." Journal of Digital Imaging 36(2), 739–752 (2023).

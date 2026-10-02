# Street Mapper: Automated Pothole Detection using YOLO

Street Mapper is a deep learning system that detects potholes in road images and video using a fine-tuned **YOLO11s** model. For every detection it logs the confidence score and a (simulated) GPS location, which makes it a starting point for smart-city road monitoring and proactive maintenance.

> College project, Department of Engineering Sciences, **IIIT Gwalior**.

---

## Features

- Detects potholes in images and video streams in real time
- Fine-tunes a pre-trained YOLO11s model with transfer learning
- Prints a location-based alert with a confidence score for each detection
- Saves detections (timestamp, latitude, longitude, issue type, confidence) to a CSV file
- Includes an optional TFLite export step for mobile and edge deployment

## Results

Evaluated on the validation set (395 images, 1,394 pothole instances) after 80 epochs:

| Metric | Value |
|---|---|
| Precision | 0.802 |
| Recall | 0.719 |
| mAP@50 | 0.805 |
| mAP@50-95 | 0.543 |
| Inference speed | ~3.6 ms per image (Tesla T4) |

## Tech Stack

- Python 3
- [Ultralytics YOLO11](https://docs.ultralytics.com/)
- PyTorch
- Google Colab (Tesla T4 GPU)
- OpenCV, PyYAML, CSV

## Project Structure

```
street-mapper/
├── Street_mapper.ipynb      # Full pipeline: setup, training, detection, export
├── docs/
│   └── Street_Mapper_Report.pdf
├── README.md
└── .gitignore
```

## Dataset

The dataset contains road images labelled with bounding boxes in **YOLO format**, with one class (`pothole`). It is split into a training set and a validation set:

```
datasets/
└── potholes/
    ├── train/    # .jpg images + .txt labels
    └── valid/
```

The dataset is not included in this repository because of its size. Add your source or download link here: `<dataset link>`

## How to Run

The notebook is designed for **Google Colab** with a GPU runtime.

1. Open `Street_mapper.ipynb` in Colab and select **Runtime → Change runtime type → T4 GPU**.
2. Put your dataset in Google Drive at `MyDrive/Street/train` and `MyDrive/Street/valid` (each with images and matching `.txt` labels), or edit the paths in the notebook.
3. Run the cells in order:
   - Mount Drive and install Ultralytics (`pip install -U ultralytics`)
   - Link the dataset and create `pothole_config.yaml`
   - Train the model
   - Run detection on a video
   - Log detections to CSV

### Training configuration

| Parameter | Value |
|---|---|
| Model | YOLO11s (pre-trained) |
| Epochs | 80 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Device | GPU (Tesla T4) |

Training takes about 1.2 hours. The best weights are saved to `runs/detect/street_mapper_mvp/weights/best.pt`.

### Run detection

```python
from ultralytics import YOLO

model = YOLO('runs/detect/street_mapper_mvp/weights/best.pt')
results = model.predict(source='sample_video.mp4', conf=0.5, stream=True)

for result in results:
    if len(result.boxes) > 0:
        print("DAMAGE DETECTED")
        print(f"Confidence: {result.boxes.conf[0].item():.2f}")
```

Example output:

```
!!! DAMAGE DETECTED !!!
Location: 22.7196, 75.8577
Confidence: 0.86
```

### Export for mobile (optional)

```python
model = YOLO('runs/detect/street_mapper_mvp/weights/best.pt')
model.export(format='tflite', int8=True)
```

## Limitations

- GPS locations are **simulated**. Real GPS integration is not implemented yet.
- The dataset is limited in size, which affects generalisation.
- Accuracy drops in low light, blurred frames and extreme weather.
- Shadows and road patches can cause false positives.

## Future Work

- Real-time GPS integration for accurate location tracking
- Deployment on mobile or edge devices
- Larger and more diverse datasets
- Dashboard integration for reporting to authorities
- Multi-class detection (cracks, bumps, debris)

## Report

The full project report is available in [`docs/Street_Mapper_Report.pdf`](docs/Street_Mapper_Report.pdf).

## Authors

- **Naveen P Raj** (2025MS013)
- **Arkadyuti Bhattacharjee** (2025MS006)

IIIT Gwalior

## References

1. Redmon et al., *You Only Look Once: Unified, Real-Time Object Detection*
2. [Ultralytics YOLO Documentation](https://docs.ultralytics.com/)
3. Recent research on road damage detection using deep learning

## License

Add a license here if you want one (for example MIT), or remove this section.

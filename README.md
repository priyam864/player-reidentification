
# 🏃‍♂️ Player Re-Identification in Sports Footage

This project implements a Player Re-Identification system designed for sports video analysis. It automatically detects players in each frame, assigns them unique IDs, and maintains identity consistency—even when players leave and re-enter the scene. The system leverages a fine-tuned YOLOv11 model for detection and DeepSORT for real-time, appearance-aware tracking. The modular design allows individual component testing, feature extraction, and flexible customization.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ⚙️ Tech Stack

| Component             | Technology / Library                                                |
| ---------------------|----------------------------------------------------------------------|
| 🧠 Detection          | [YOLOv11](https://github.com/ultralytics/ultralytics) (Ultralytics) |
| 🔁 Tracking           | [DeepSORT](https://github.com/nwojke/deep_sort)                     |
| 🔬 Feature Extraction | Color histograms, spatial & texture descriptors                     |
| 🖼️ Visualization     | OpenCV                                                               |
| 🐍 Language           | Python 3.8+                                                          |
| 📦 Dependencies       | PyTorch, OpenCV, NumPy, scikit-learn, matplotlib, tqdm, requests     |

---

## 📌 Features

- 🎯 **Player Detection** using a fine-tuned YOLOv11 model.
- 🔁 **Player Tracking** using DeepSORT for maintaining consistent IDs.
- 🧠 **Feature Extraction** including color, texture, and spatial features for re-identification.
- 📊 **ID Management** to match re-appearing players with previous IDs.
- 🖼️ **Visualization** with bounding boxes and track IDs.
- 🧪 **Modular Testing** of all components (detection, tracking, extraction, etc.).
- ✅ Easy setup via a single `setup.py` script.

---

## 🗂️ Project Structure

```

player-reidentification/
│
├── data/                   # Input videos (e.g., 15sec\_input\_720p.mp4)
├── models/                 # YOLOv11 model weights (best.pt)
├── output/                 # Output videos with annotated tracking
├── src/                    # Core source modules
│   ├── detector.py             # Player detection (YOLOv11 wrapper)
│   ├── feature\_extractor.py   # Color, texture, spatial features
│   ├── tracker.py             # DeepSORT-based tracking
│   ├── visualizer.py          # Bounding box & track ID rendering
│   └── main.py                # Main pipeline runner
│
├── download\_model.py        # Script to download the YOLOv11 model
├── test\_system.py           # Unit tests for pipeline and modules
├── setup.py                 # Installation, setup, and testing script
├── requirements.txt         # Python dependencies
├── .gitignore               # Files and folders to ignore in Git
└── README.md                # Project overview and documentation

````

---

## ⚙️ Setup Instructions

### ✅ Step 1: Clone the Repository

```bash
git clone https://github.com/priyam864/player-reidentification.git
cd player-reidentification
````

### ✅ Step 2: Run the Setup Script

This will:

* Check Python version
* Create required directories
* Install dependencies
* Download YOLOv11 weights
* Run a basic test

```bash
python setup.py
```

📥 If the model fails to download automatically, you can download it manually:

[Manual Model Link](https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view)
Save it as: `models/best.pt`

---

## 🎥 Running the System

### 🏁 Basic Command

```bash
python -m src.main data/your_video.mp4
```

### ⚙️ Optional Arguments

```bash
--output output/your_result.mp4       # Custom output path
--save-comparison                     # Save side-by-side comparison view
--show-detections                     # Show raw detections only (no tracking)
```

### 💡 Example

```bash
python -m src.main data/15sec_input_720p.mp4 --output output/tracked.mp4 --save-comparison
```

---

## 🧪 Testing

You can verify that all components (detector, tracker, extractor, visualizer) are working by running:

```bash
python test_system.py
```

This runs:

* Frame detection
* Feature extraction
* Dummy frame tracking
* Drawing visualizations

---

## 💡 Notes

* The YOLOv11 model used is fine-tuned to detect players (`class 0`).
* The system is modular, allowing individual testing or replacement of parts.
* Videos are saved automatically in `output/`.

---

## 🔧 Dependencies

Dependencies are listed in [`requirements.txt`](requirements.txt) and installed via `setup.py`. Major libraries include:

* `ultralytics` (YOLOv11)
* `opencv-python`
* `torch`, `torchvision`
* `numpy`, `scikit-learn`
* `tqdm`, `requests`, `matplotlib`, `pillow`, `scipy`

---

## 📬 Contact

For questions or collaboration, feel free to reach out via [GitHub](https://github.com/priyam864).

# Chart Info Detector 📈🕵️‍♂️

<!-- Add a banner here like: https://github.com/StephanAkkerman/fintwit-bot/blob/main/img/logo/fintwit-banner.png -->

---
<!-- Adjust the link of the first and second badges to your own repo -->
<p align="center">
  <img alt="GitHub Actions Workflow Status" src="https://img.shields.io/github/actions/workflow/status/StephanAkkerman/chart-info-detector/pyversions.yml?label=python%203.10%20%7C%203.11%20%7C%203.12%20%7C%203.13&logo=python&style=flat-square">
  <img src="https://img.shields.io/github/license/StephanAkkerman/chart-info-detector.svg?color=brightgreen" alt="License">
  <a href="https://github.com/psf/black"><img src="https://img.shields.io/badge/code%20style-black-000000.svg" alt="Code style: black"></a>
</p>

## Introduction

This project provides a tool to detect and extract key information from trading charts, such as the symbol title and last price pill, using computer vision techniques. The tool leverages the YOLO (You Only Look Once) object detection model to accurately identify and label these elements in chart images.

## Labelling Process (optional) 🏷️
I have already labelled a dataset of trading chart images using Label Studio which are availble on this Hugginface dataset repo: https://huggingface.co/datasets/StephanAkkerman/chart-info-yolo. If you want to label your own dataset, follow the instructions below.

After installing Label Studio using the command in the Installation section, you can start a new labelling project by running the following command in your terminal:

```bash
label-studio
```

Use the following XML code snippet as the labeling configuration:

```xml
<View>
  <Image name="img" value="$image"/>
  <RectangleLabels name="label" toName="img">
    <Label value="symbol_title" background="#1f77b4"/>
    <Label value="last_price_pill" background="#ff7f0e"/>
  </RectangleLabels>
</View>
```

Then, import your images into the project and start labelling the `symbol_title` and `last_price_pill` regions in each image. After you have finished labelling, you can export the annotations in YOLO format.
To do this, go to the "Export" tab in Label Studio, select "YOLO format", and download the annotations.

Afterwards you need to clean up the exported files. Make sure that the directory structure matches the expected format for training. I have made some scripts to help with this process in the `dataset_creation` folder.

1. Move the images and their corresponding `.txt` annotation files into a single directory structure like this:

```
datasets/
└── tradingview/
    ├── images/
    │   ├── train/
    │   └── val/
    └── labels/
        ├── train/
        └── val/
```
2. Ensure that the `data.yml` file in the `datasets/tradingview/` directory is correctly set up to point to the training and validation image and label directories.
3. Verify that the class names in the `data.yml` file match the labels you used during annotation (`symbol_title` and `last_price_pill`).
4. Run the `align_label_files.py` script in the `dataset_creation` folder to ensure that all images have corresponding label files and vice versa.
5. Run `check_yolo_dataset.py` script in the `dataset_creation` folder to verify the integrity of the dataset.

## Table of Contents 🗂

- [Installation](#installation)
- [Usage](#usage)
- [Citation](#citation)
- [Contributing](#contributing)
- [License](#license)

## Installation ⚙️
<!-- Adjust the link of the second command to your own repo -->

The required packages to run this code can be found in the requirements.txt file. To run this file, execute the following code block after cloning the repository:

```bash
pip install -r requirements.txt
```

or

```bash
pip install git+https://github.com/StephanAkkerman/chart-info-detector.git
```

## Usage ⌨️
To train the YOLO model on your dataset, run the following command in your terminal:

```bash
python src/main.py
```

To set up wandb for it, simply run the following command before training:

```bash 
wandb login
```

And then enable wandb in the settings with:

```bash
yolo settings wandb=True
```

## Citation ✍️
<!-- Be sure to adjust everything here so it matches your name and repo -->
If you use this project in your research, please cite as follows:

```bibtex
@misc{chart_info_detector_2025,
  author  = {Stephan Akkerman},
  title   = {Chart Info Detector},
  year    = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/StephanAkkerman/chart-info-detector}}
}
```

## Contributing 🛠
<!-- Be sure to adjust the repo name here for both the URL and GitHub link -->
Contributions are welcome! If you have a feature request, bug report, or proposal for code refactoring, please feel free to open an issue on GitHub. We appreciate your help in improving this project.\
![https://github.com/StephanAkkerman/chart-info-detector/graphs/contributors](https://contributors-img.firebaseapp.com/image?repo=StephanAkkerman/chart-info-detector)

## License 📜

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

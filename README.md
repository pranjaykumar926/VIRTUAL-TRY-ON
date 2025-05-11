# Virtual Try-On for Jewellery

This repository presents an AI-powered solution to revolutionize the online jewellery shopping experience through virtual try-on capabilities. By integrating computer vision and deep learning techniques, users can try on jewellery items virtually using either uploaded images or their webcam.

## Features

* **Real-Time Jewellery Detection**
  Utilizes a custom-trained YOLO (You Only Look Once) model to detect and localize jewellery such as earrings, necklaces, and rings in real time.

* **Interactive Web Application**
  Offers a user-friendly interface for uploading images or using webcam feeds to simulate jewellery try-ons.

* **End-to-End AI Pipeline**
  Covers all stages from model training to deployment, providing a seamless user experience.

## Technologies Used

* **Python** – Core programming language for model development and application logic.
* **YOLO** – Object detection model used for jewellery localization.
* **OpenCV** – Library used for real-time image processing and integration.

## Setup Instructions

1. Clone the repository:

   ```bash
   git clone https://github.com/pranjaykumar926/VIRTUAL-TRY-ON.git
   ```

2. Navigate to the project directory:

   ```bash
   cd VIRTUAL-TRY-ON
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the application:

   ```bash
   python app.py
   ```

## Resources

* **YOLO Training Notebook** – Details the training process, dataset preparation, and configuration.
* **Presentation** – Highlights the project scope, technical approach, and results.
* **Sample Outputs** – Includes example images demonstrating the try-on feature.

## Future Work

* Integration with more jewellery types.
* Support for 3D rendering and AR-based previews.
* Enhanced facial/jewellery landmark detection for improved placement accuracy.

---

For additional information and updates, please visit the [GitHub Repository](https://github.com/pranjaykumar926/VIRTUAL-TRY-ON).

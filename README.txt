Hybrid CAPTCHA Recognition Project

Overview

This project aims to generate and solve hybrid CAPTCHA images that blend text-based CAPTCHAs with object images. We utilize a CRNN (Convolutional Recurrent Neural Network) to recognize CAPTCHA text and a YOLOv8 model for object detection with semantic similarity analysis using SBERT.

The project comprises:

CAPTCHA generation with noise and object blending.

CRNN model training and evaluation for text recognition.

YOLOv8 model evaluation for object detection.

Combined testing of the models on hybrid CAPTCHA images.

Required Libraries and Dependencies

To run the project, install the required libraries by running the following commands:

Installations

# Basic Libraries and Tools
!pip install captcha
!pip install torch torchvision torchaudio
!pip install matplotlib numpy pillow
!pip install opencv-python

# For YOLOv8 Object Detection
!pip install ultralytics

# For High-Resolution Image Generation
!pip install diffusers

# For Semantic Similarity
!pip install sentence-transformers

# Visualization Tools
!pip install torchviz

File Structure

├── captcha_wo_noise_sample           # Contains approx. 5000 clean CAPTCHA images.
├── CRNN model                        # CRNN model architecture, checkpoints, and training code.
│   ├── CRNN model.ipynb              # CRNN model architecture and training.
│   ├── crnn_epoch_40.pth             # Model checkpoint at epoch 40.
│   ├── crnn_epoch_200.pth            # Model checkpoint at epoch 200.
│
├── final_dataset                     # Contains 1000 blended hybrid CAPTCHA images.
│
├── Execute.ipynb                     # Main notebook to test both CRNN and YOLO models.
├── Getting_final_blended_CAPTCHA_image_right.ipynb
│                                    # Notebook for adding noise and generating hybrid CAPTCHAs.
│
├── Hybrid_model_results.ipynb        # Testing YOLO+SBERT for object detection and CRNN for text.
├── Stable_diffusion_and_captcha_images_generation.ipynb
│                                    # Generating high-res images and CAPTCHA text.
│
├── README.md                         # Project documentation.

Instructions for Execution

Step 1: Setting Up the Environment

Ensure Python (>= 3.8) is installed.

Install the required libraries using the commands mentioned above.

Step 2: Dataset Preparation

captcha_wo_noise_sample: Clean text CAPTCHA images without noise.

final_dataset: Hybrid CAPTCHAs with blended object images and added noise.

Step 3: Running the CRNN Model

Open the Execute.ipynb notebook.

Use the function to pass an image file path and test the prediction for both CRNN and YOLO models.

image_path = '/path/to/your/image.png'
predict_image(image_path, crnn_model, yolo_model, symbols, device)

If you wish to test only the CRNN model, use the function:

predict_crnn(image_path, crnn_model, symbols, device)

Step 4: Results Visualization

The predictions for CRNN and YOLO models are displayed along with the input image.

The combined results of both models are printed, including:

Exact Match (YOLO)

Semantic Match (YOLO)

Text Recognition Success (CRNN)

Step 5: Model Weights

Pretrained Weights:

CRNN checkpoints: crnn_epoch_40.pth and crnn_epoch_200.pth

Located in the CRNN model folder.

Alternatively, download from the provided Drive link: https://drive.google.com/drive/folders/1f5rppfhgJ1NnU8kXNf32AjlCiAzulnMP?usp=sharing

Step 6: Training CRNN Model

To retrain the CRNN model:

Open the CRNN model.ipynb notebook.

Run the training loop with your dataset.

train_crnn(model, train_loader, val_loader, optimizer, criterion, epochs=200, checkpoint=True, checkpoint_path='checkpoints/')

Training and validation loss will be plotted.

Results

Performance Metrics

Training Set Evaluation:

Average Loss: 0.0103

Sequence-Level Accuracy: 93.34%

Testing Set Evaluation:

Average Loss: 0.0503

Sequence-Level Accuracy: 87.59%

Notes

Noise and Object Blending: Noise was carefully tuned to balance human readability and CAPTCHA difficulty.

Model Testing: The CRNN model was trained on base CAPTCHA images (without noise or background) and tested on blended images.

YOLO+SBERT: YOLOv8m predictions were enhanced using semantic similarity to allow approximate matches.

Future Work

Explore adaptive noise generation to align with human readability.

Enhance CRNN generalization by training it on hybrid CAPTCHA images.

Evaluate robustness of YOLO and CRNN with additional preprocessing techniques.

References
Stable Diffusion: CompVis/stable-diffusion

YOLOv8: Ultralytics YOLOv8 for state-of-the-art object detection.

CRNN: Convolutional Recurrent Neural Networks for text recognition.

SBERT: Sentence-BERT for semantic similarity.


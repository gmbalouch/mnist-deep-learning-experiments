# MNIST Deep Learning Experiments
 
A hands-on deep learning project exploring handwritten digit classification using progressively more advanced neural network architectures: a Perceptron baseline, an Artificial Neural Network (ANN) with hidden layers, and a Convolutional Neural Network (CNN).
 
## Objectives
 
- Establish a Perceptron baseline for MNIST classification.
- Investigate an ANN with hidden layers and the Adam optimizer.
- Implement a CNN for image classification.
- Compare model accuracy, training time, and parameter count under identical data and evaluation settings.
## Dataset
 
The project uses the MNIST handwritten digit dataset (Kaggle CSV format), consisting of grayscale images of digits from 0 to 9.
 
- Total samples: 42,000 (no missing values)
- Train/test split: 80/20 (33,600 train / 8,400 test), `random_state=42`
- Image dimensions: 28 × 28 pixels, pixel values scaled to [0, 1]
- Number of classes: 10 (one-hot encoded labels)
- Input format: `(samples, 28, 28)` for the Perceptron and ANN, `(samples, 28, 28, 1)` for the CNN
## Models
 
| Model | Architecture | Optimizer | Epochs | Parameters |
|---|---|---|---:|---:|
| Perceptron | Flatten → Dense(10, softmax) | SGD | 5 | 7,850 |
| ANN | Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, softmax) | Adam | 10 | 109,386 |
| CNN | Conv2D(32, 3×3, ReLU) → MaxPool(2×2) → Conv2D(64, 3×3, ReLU) → MaxPool(2×2) → Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(10, softmax) | Adam | 5 | 225,034 |
 
All models use categorical cross-entropy loss and a batch size of 32.
 
## Results
 
| Model | Test Accuracy | Misclassified (of 8,400) | Approx. Training Time (Colab) |
|---|---:|---:|---:|
| Perceptron | 89.24% | ~904 | ~14 s |
| ANN | 97.13% | ~241 | ~50 s |
| CNN | **98.92%** | **~91** | ~190 s |
 
Accuracies come from the latest clean run of the full notebook (`03_cnn_experiments.ipynb`). Training times are approximate and depend on the Colab hardware assigned.
 
### Key Observations
 
- **Hidden layers gave the largest jump.** Moving from the Perceptron to the two-hidden-layer ANN improved accuracy by about 7.9 percentage points, since the ANN can learn non-linear patterns that a single linear layer cannot.
- **The CNN is the best model.** It beats the ANN by about 1.8 percentage points and makes roughly 62% fewer errors (about 91 vs. 241 on the test set). This gap is far larger than the run-to-run variation observed for the ANN, so it reflects a real architectural advantage: convolutions exploit the spatial structure of images.
- **The CNN costs more compute.** It trains in about 190 s versus about 50 s for the ANN, roughly 4× slower, for a smaller but meaningful accuracy gain.
- **The ANN overfits.** Training accuracy reaches about 99.4% while validation accuracy plateaus near 97% after the first few epochs, so additional epochs do not improve generalization. The CNN, which uses Dropout(0.5), shows a much smaller train/validation gap and was still improving at epoch 5.
- **Run-to-run noise is about ±0.2%.** Repeated runs of the same ANN configuration scored between 97.10% and 97.40%, and the Perceptron between 88.92% and 89.24%. On an 8,400-image test set this is a difference of only 10–25 images, so differences of that size should not be read as the effect of a configuration change.
- **Re-running only the `fit` cell continues training from existing weights.** Each experiment must rebuild and compile the model for a fair comparison. An earlier ANN run was invalid for this reason and was re-run cleanly.
### Sample Predictions
 
The notebook visualizes predictions from all three models on 5 randomly selected test images. In the sample shown, the Perceptron and ANN both misread a "1" (with an extra stroke beside it) as an "8", while the CNN classified it correctly. All three models agreed and were correct on the other four images. This is a qualitative illustration, not a benchmark.
 
## Limitations
 
- The test set is also passed as `validation_data` during training. No model selection or early stopping was based on it, but a separate held-out set would make the reported numbers more rigorous.
- Random seeds are not fixed, so exact numbers vary slightly between runs.
- Results are from single runs per model; multi-seed averages (mean ± std) are not yet reported.
## Planned Next Steps
 
- Fix random seeds and report mean ± std over multiple runs per model.
- Compare activation functions (ReLU, tanh, sigmoid) under identical settings.
- Add early stopping and dropout to the ANN to reduce overfitting.
- Add confusion matrices and per-class metrics, and analyze the most commonly confused digits.
- Experiment with CNN depth, filter counts, and more epochs.
## Technologies
 
- Python
- NumPy and Pandas
- TensorFlow / Keras
- Matplotlib
- Scikit-learn
- Google Colab
## Project Structure
 
- `notebooks/` — Experiments and model implementations
  - `02_ann_activation_adam_experiments.ipynb` — Perceptron baseline and ANN
  - `03_cnn_experiments.ipynb` — Full pipeline: Perceptron, ANN, CNN, and prediction comparison
- `results/` — Model comparison tables and evaluation outputs
## How to Run
 
1. Download the MNIST CSV (`mnist.csv`, 42,000 rows × 785 columns) and place it next to the notebook, or upload it to Colab.
2. Open a notebook in Google Colab (or locally with Python 3 and TensorFlow installed).
3. Run all cells from top to bottom. Always run the model build, compile, and fit cells together so each experiment starts from fresh weights.
## Reproducibility
 
Experiments are developed in Google Colab. Dataset preparation, preprocessing, model configurations, and evaluation procedures are documented in the notebooks. Single-run results may vary by roughly ±0.2% due to random weight initialization and batch shuffling; multi-seed averages will be added as experiments are finalized.
 
## Author
 
Ghulam Mustafa
 
Software Engineering Student | Machine Learning & Deep Learning
 
[GitHub](https://github.com/gmbalouch) · [LinkedIn](https://linkedin.com/in/ghulammustafakhoso)
 
---
 
This is an ongoing educational and experimental project. Results reflect the configurations documented in each experiment.
 

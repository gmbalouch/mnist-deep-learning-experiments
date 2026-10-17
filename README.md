# MNIST Deep Learning Experiments

A hands-on deep learning project exploring handwritten digit classification using progressively more advanced neural network architectures.

## Objectives

- Establish a Perceptron baseline for MNIST classification.
- Investigate Artificial Neural Networks (ANNs) with hidden layers and different activation functions.
- Implement Convolutional Neural Networks (CNNs) for image classification.
- Compare model accuracy, training time, and other evaluation metrics under controlled experimental conditions.

## Dataset

The project uses the MNIST handwritten digit dataset (Kaggle CSV format), consisting of grayscale images of handwritten digits from 0 to 9.

- Total samples: 42,000 (no missing values)
- Train/test split: 80/20 (33,600 train / 8,400 test), `random_state=42`
- Image dimensions: 28 × 28 pixels, pixel values scaled to [0, 1]
- Number of classes: 10 (one-hot encoded labels)
- Input format for CNNs: (samples, 28, 28, 1)

## Experiments and Results

| Model | Configuration | Epochs | Test Accuracy |
|---|---|---:|---:|
| Perceptron | Flatten → Dense(10, softmax), SGD | 5 | 88.92% |
| ANN | Flatten → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, softmax), Adam | 10 | 97.23% |
| CNN | Planned | – | Not yet evaluated |

All models use categorical cross-entropy loss and batch size 32.

### Key Observations

- Adding two hidden layers with ReLU and switching to Adam improved accuracy by about 8.3 percentage points over the Perceptron baseline.
- Across repeated runs of the same ANN configuration, test accuracy varied between 97.10% and 97.40%. Differences of this size (about ±0.2%, roughly 10–25 images on the 8,400-image test set) are within run-to-run noise from random initialization, so they should not be read as the effect of a configuration change.
- The ANN overfits: training accuracy reaches about 99.7% while validation accuracy plateaus near 97.2% after the first few epochs. Training for more epochs does not improve generalization.
- Re-running only the `fit` cell continues training from existing weights. Each experiment must rebuild and compile the model for a fair comparison.

## Planned Next Steps

- Fix random seeds and report mean ± std over multiple runs per configuration.
- Compare activation functions (ReLU, tanh, sigmoid) under identical settings.
- Reduce overfitting with early stopping and dropout.
- Implement and evaluate CNN models.

## Technologies

- Python
- NumPy and Pandas
- TensorFlow / Keras
- Matplotlib
- Scikit-learn
- Google Colab

## Project Structure

- `notebooks/` — Experiments and model implementations
- `results/` — Model comparison tables and evaluation outputs

## Current Progress

- Perceptron baseline completed: 88.92% test accuracy.
- ANN with hidden layers completed: 97.23% test accuracy (single run; observed range 97.10–97.40%).
- Next: activation-function comparison, multi-seed evaluation, then CNN experiments.

## Reproducibility

Experiments are developed in Google Colab. Dataset preparation, preprocessing, model configurations, and evaluation procedures are documented in the notebooks. Results from single runs may vary slightly due to random weight initialization; multi-seed averages will be added as experiments are finalized.

## Author

Ghulam Mustafa

Software Engineering Student | Machine Learning & Deep Learning

[GitHub](https://github.com/gmbalouch) · [LinkedIn](https://linkedin.com/in/ghulammustafakhoso)

---

This is an ongoing educational and experimental project. Results reflect the configurations documented in each experiment.

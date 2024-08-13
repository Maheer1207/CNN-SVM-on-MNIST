# ResolutionNet CNN vs. SVM on MNIST

This project focuses on training Convolutional Neural Networks (CNN) and Support Vector Machines (SVM) on the MNIST dataset. The goal is to analyze the performance of these models on both the original and downscaled low-resolution datasets, calculating confidence intervals to assess their stability and generalization.

## Project Overview

The MNIST dataset of numbers is widely available, as is the Fashion MNIST dataset, which contains images of clothing. For this project, we also explore the effects of downscaling the images to lower resolutions, such as 7x7 pixels, and comparing the performance of our models across different resolutions.

### Tasks Addressed

1. **Training a CNN on the MNIST Dataset**:
   - **Original Fashion MNIST (28x28 resolution)**:
     - Code implementation for training.
     - Explanation of the model architecture, including pooling methods and parameter selection.
     - Graphical comparison of training and test results with computed confidence intervals.
   - **Low-Resolution MNIST (e.g., 7x7 resolution)**:
     - Code implementation for training.
     - Explanation of architectural modifications to handle low-resolution data.
     - Graphical comparison of training and test results with computed confidence intervals.
   - **Performance Analysis**:
     - Detailed explanation of model performance, comparing the results across different resolutions and dataset types.

2. **Training an SVM on the MNIST Dataset**:
   - **Original MNIST (28x28 resolution)**:
     - Code implementation for training.
     - Explanation of the SVM configuration and parameter selection.
     - Graphical comparison of training and test results with computed confidence intervals.
   - **Low-Resolution MNIST (e.g., 7x7 resolution)**:
     - Code implementation for training.
     - Explanation of architectural modifications to handle low-resolution data.
     - Graphical comparison of training and test results with computed confidence intervals.
   - **Performance Analysis**:
     - Detailed explanation of model performance, comparing the results across different resolutions and dataset types.

## Requirements

To run the code in this project, ensure that you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- TensorFlow
- Keras
- scikit-learn
- NumPy
- Matplotlib

You can install these packages using `pip`:

```bash
pip install jupyter tensorflow keras scikit-learn numpy matplotlib
```

## Usage

### Running the Classifiers

All the code for training the CNN and SVM models, as well as plotting the results, is contained in a Jupyter Notebook. Follow the steps below to run the notebook:

1. Clone the repository:

```bash
git clone https://github.com/Maheer1207/ResolutionNet-CNN-vs.-SVM-on-MNIST.git
```

2. Navigate to the project directory:

```bash
cd ResolutionNet-CNN-vs.-SVM-on-MNIST
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open the notebook named `classifiers.ipynb`:

   This notebook contains all the necessary code to:
   - Train the CNN on both the original and low-resolution MNIST datasets.
   - Train the SVM on both the original and low-resolution MNIST datasets.
   - Plot the training and testing accuracy along with confidence intervals for comparison.

5. Execute the cells in the notebook to run the models and generate the plots.

### Viewing the Results

Once you've run the notebook, the results, including accuracy plots and confidence intervals, will be displayed inline within the notebook. You can also save these plots from the notebook interface for further analysis.

## Results

The results of the training processes, including the accuracy and confidence intervals for both CNN and SVM models, are documented within the Jupyter Notebook. Graphs are provided to visualize the performance differences between the original and low-resolution datasets.

### Key Observations

- **CNN Performance**:
  - The CNN model performed better on the original 28x28 dataset compared to the low-resolution 7x7 dataset.
  - As the resolution decreases, the test accuracy drops, indicating the importance of image detail for CNNs.

- **SVM Performance**:
  - The SVM model showed consistent performance across different resolutions, though it required more time to train and validate.
  - Interestingly, the SVM performed relatively better on the low-resolution dataset compared to CNN, though overall performance was still better on the original resolution.

## Conclusion

This project highlights the impact of image resolution on model performance. CNNs, which are inherently designed to capture spatial hierarchies in images, perform better with higher-resolution inputs, whereas SVMs, while more resilient to resolution changes, require significant computational resources.

The comparison underscores the need to choose the right model based on the specific use case, data characteristics, and computational constraints.

## Contributing

If you'd like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

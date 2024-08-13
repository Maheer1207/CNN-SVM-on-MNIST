# ResoNet: CNN vs. SVM on Multi-Resolution MNIST

This project investigates the impact of image resolution on the performance of Convolutional Neural Networks (CNN) and Support Vector Machines (SVM) using the MNIST dataset. We employ both original and downsampled low-resolution datasets to evaluate the models' accuracy, stability, and generalization by calculating and visualizing confidence intervals.

## Project Overview

The MNIST dataset, consisting of handwritten digits, is a standard benchmark in the machine learning community. We extend this dataset by exploring its downsampled versions, such as reducing the resolution from 28x28 to 7x7 pixels. The project leverages CNNs, known for their ability to capture spatial hierarchies, and SVMs, which are effective for binary and multiclass classification, to compare model robustness across varying image resolutions.

### Objectives

1. **CNN Training on MNIST**:
   - **Original Fashion MNIST (28x28 resolution)**:
     - Implementation of a deep convolutional neural network (CNN).
     - Detailed rationale for architectural choices, including convolutional layers, pooling strategies (e.g., MaxPooling), and hyperparameter selection.
     - Visualization of training and validation accuracy, accompanied by confidence intervals.
   - **Low-Resolution MNIST (e.g., 7x7 resolution)**:
     - Adaptation of the CNN architecture to handle lower-resolution inputs.
     - Comparative analysis of the model's performance on downsampled images.
     - Graphical representation of the training and test results with confidence intervals.
   - **Performance Evaluation**:
     - Comparative study of model performance across different resolutions, with a focus on overfitting, underfitting, and generalization capabilities.

2. **SVM Training on MNIST**:
   - **Original MNIST (28x28 resolution)**:
     - Implementation of a Support Vector Machine (SVM) classifier using a linear kernel.
     - Discussion on the choice of hyperparameters such as `C` (regularization) and `gamma`.
     - Generation of confidence intervals to evaluate the model's reliability.
   - **Low-Resolution MNIST (e.g., 7x7 resolution)**:
     - Modification of the input pipeline for SVM to accommodate lower-dimensional feature vectors.
     - Comparative analysis of classification performance on reduced-resolution data.
     - Visualization of accuracy metrics with associated confidence intervals.
   - **Performance Evaluation**:
     - Comparative study of SVM performance across different resolutions, highlighting its sensitivity to input dimensionality and dataset complexity.

## Requirements

To execute the code in this project, ensure you have the following dependencies installed:

- Python 3.x
- Jupyter Notebook
- TensorFlow
- Keras
- scikit-learn
- NumPy
- Matplotlib

Install these dependencies using `pip`:

```bash
pip install jupyter tensorflow keras scikit-learn numpy matplotlib
```

## Usage

### Running the Classifiers

All the code for training the CNN and SVM models, as well as visualizing the results, is encapsulated in a Jupyter Notebook. Follow the instructions below to run the notebook:

1. Clone the repository:

```bash
git clone https://github.com/Maheer1207/Resonet.git
```

2. Navigate to the project directory:

```bash
cd Resonet
```

3. Launch Jupyter Notebook:

```bash
jupyter notebook
```

4. Open the notebook named `classifiers.ipynb`:

   This notebook contains:
   - End-to-end implementation for training CNNs on both the original and downsampled MNIST datasets.
   - End-to-end implementation for training SVMs on both the original and downsampled MNIST datasets.
   - Generation of plots for training and validation accuracy, along with confidence intervals to assess model reliability.

5. Execute the cells in the notebook sequentially to train the models and generate the visualizations.

### Viewing the Results

The notebook will display the results, including accuracy plots and confidence intervals, inline. You can save the generated plots directly from the notebook for further analysis or reporting purposes.

## Results

The training processes yield insights into the accuracy and stability of CNN and SVM models across different resolutions. The key performance indicators (KPIs) such as accuracy, confidence intervals, and loss curves are visualized within the Jupyter Notebook.

### Key Observations

- **CNN Performance**:
  - The CNN exhibits superior performance on the original 28x28 dataset, leveraging the spatial hierarchies inherent in the image data.
  - A significant drop in test accuracy is observed as the resolution decreases, indicating that CNNs are highly sensitive to the level of detail in input images.

- **SVM Performance**:
  - The SVM demonstrates relatively consistent performance across different resolutions, though it is computationally intensive, especially when handling higher-dimensional data.
  - On low-resolution datasets, the SVM's performance is more resilient compared to CNNs, though the overall accuracy remains higher for the original dataset.

## Conclusion

This project underscores the importance of image resolution in machine learning model performance. CNNs, with their ability to exploit spatial structures, are more affected by reduced resolution, while SVMs, though less sensitive, require careful hyperparameter tuning to handle varying input dimensions. The results highlight the trade-offs between model complexity, input resolution, and computational resources.

## Contributing

We welcome contributions to this project. If you have suggestions or improvements, please fork the repository and submit a pull request. For significant changes, open an issue to discuss the proposed modifications.

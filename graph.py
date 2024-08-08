import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def graph_mnist(classifier, num_runs, num_epoch, model, x_train, y_train, x_test, y_test):
  if (classifier == "CNN"):
    accs, val_accs = collect_data(num_runs, num_epoch, model, x_train, y_train, x_test, y_test)
  elif (classifier == "SVM"):
    accs, val_accs = collect_data_svm(num_runs, num_epoch, model, x_train, y_train, x_test, y_test)

  mean_accuracy, std_accuracy, mean_val_accuracy, std_val_accuracy = process_data(accs, val_accs)
  
  plot_data(classifier, num_epoch, num_runs, mean_accuracy, std_accuracy, mean_val_accuracy, std_val_accuracy)


def collect_data(num_runs, num_epoch, model, x_train, y_train, x_test, y_test):
    # Initialize lists to store metrics
    accs, val_accs = [], []
    for _ in range(num_runs):
        history = model.fit(x_train, y_train, epochs=num_epoch, validation_data=(x_test, y_test), verbose=0)
        accs.append(history.history['accuracy'])
        val_accs.append(history.history['val_accuracy'])
    return accs, val_accs

def collect_data_svm(num_runs, num_epoch, model, x_train, y_train, x_test, y_test):
  accs, val_accs = [], []
  for run in range(num_runs):
    run_accs, run_val_accs = [], []
    for step in range(1, num_epoch + 1):
      subset_size = int(len(x_train) * (step / num_epoch))
      x_train_subset, x_val, y_train_subset, y_val = train_test_split(x_train, y_train, train_size=((step / num_epoch)*0.75), shuffle=True)
      model.fit(x_train_subset, y_train_subset)
      train_accuracy = model.score(x_train_subset, y_train_subset)
      test_accuracy = model.score(x_test, y_test)
      run_accs.append(train_accuracy)
      run_val_accs.append(test_accuracy)
    accs.append(run_accs)
    val_accs.append(run_val_accs)
  return accs, val_accs

def process_data(accs, val_accs):
  # Convert lists to numpy arrays for calculation
  accs, val_accs = np.array(accs), np.array(val_accs)

  # Calculate means and standard deviations
  mean_accuracy = np.mean(accs, axis=0)
  std_accuracy = np.std(accs, axis=0)
  mean_val_accuracy = np.mean(val_accs, axis=0)
  std_val_accuracy = np.std(val_accs, axis=0)

  print(mean_accuracy, std_accuracy, mean_val_accuracy, std_val_accuracy)

  return mean_accuracy, std_accuracy, mean_val_accuracy, std_val_accuracy

def plot_data(classifier, num_epoch, num_runs, mean_accuracy, std_accuracy, mean_val_accuracy, std_val_accuracy):
  epochs = range(1, num_epoch+1)

  # Plot with confidence intervals
  plt.figure(figsize=(10, 6))

  # Training and Validation Accuracy
  plt.plot(epochs, mean_accuracy, label='Training Accuracy')
  plt.fill_between(epochs, mean_accuracy - 1.96*(std_accuracy/np.sqrt(num_runs)), mean_accuracy + 1.96*(std_accuracy/np.sqrt(num_runs)), alpha=0.1)
  plt.plot(epochs, mean_val_accuracy, label='Testing Accuracy')
  plt.fill_between(epochs, mean_val_accuracy - 1.96*(std_val_accuracy/np.sqrt(num_runs)), mean_val_accuracy + 1.96*(std_val_accuracy/np.sqrt(num_runs)), alpha=0.1)

  if (classifier == "CNN"):
    plt.title('Training and Testing Accuracy over Epochs')
    plt.xlabel('Epoch')
  elif (classifier == "SVM"):
    plt.title('Training and Testing Accuracy over Varying Subsets')
    plt.xlabel('Subset Size as "Epochs"')

  plt.ylabel('Accuracy')
  plt.legend()

  plt.show()
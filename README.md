# ML_project_2024

We created a model based on the ResNet-18 pre-trained architecture to predict blinker states from cropped images of cars. We used images from the Rear Signal Dataset ([link](http://vllab1.ucmerced.edu/~hhsu22/rear_signal/rear_signal)) and Roboflow ([link](https://universe.roboflow.com/ilham-winar/venom/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)) for training. After sorting the training images, we manually labeled them and created a balanced dataset with 1000 images per combination (8 combinations). We applied multi-class binary labels for three classes.

In this GitHub repository, we provide the codebase we created for the project, a CSV file with correct labels, and 1000 images for which we predicted labels using our model.

Trained model is available ([here](https://drive.google.com/drive/folders/1JFBR16hBg1FINO1TkWYj_qMNfphB-cEg?usp=sharing))

* In the codebase, lines 1-82 are for training.
* The function **evaluate_model(model, dataloader, threshold=0.5, device="cpu")** is used for predicting and evaluating the model. It returns all predictions and ground truths and prints out accuracy, precision, recall, and F1-score for each class.
* The function **confusion_matrix(labels, predictions)** creates a confusion matrix for each class and plots it.

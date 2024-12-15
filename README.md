# ML_project_2024

We created a model based on the ResNet-18 pre-trained architecture to predict blinker states from cropped images of cars. We used images from the Rear Signal Dataset ([link](http://vllab1.ucmerced.edu/~hhsu22/rear_signal/rear_signal)) and Roboflow ([link](https://universe.roboflow.com/ilham-winar/venom/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)) for training. After sorting the training images, we manually labeled them and created a balanced dataset with 1000 images per combination (8 combinations). We applied multi-class binary labels for three classes.

In this GitHub repository, we provide the codebase we created for the project, a CSV file with correct labels, and 1000 images for which we predicted labels using our model.

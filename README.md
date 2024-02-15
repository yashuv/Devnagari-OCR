## Handwritten Devnagari Optical Character Recognition

### Project Description
The Handwritten Devanagari Optical Character Recognition (OCR) project aimed to develop a system capable of recognizing handwritten Devanagari characters from a dataset of 46 classes, including 36 consonants and 10 numerals. The system utilized a Convolutional Neural Network (CNN) for feature extraction from the handwritten characters and employed OpenCV for image pre-processing.

### Dataset Information
The dataset is taken from the *UCI Machine Learning Repository*.

The Devanagari Handwritten Character Dataset is a collection of 92,000 grayscale images of handwritten characters and comprising of 46 different classes (10 numerals and 36 consonants). Each character is represented by a 32x32 pixel image. The dataset is divided into a training set of 85% of the images and a test set of 15% of the images. It is useful for training and evaluating machine learning algorithms for handwritten character recognition. 

### Methodology

#### i) Model Training

Firstly, the dataset was gathered which will serve as the foundation for training and evaluating the model. The dataset consists of a train set and a seperate test set. 

Before training the model, the dataset undergoes preprocessing steps which included normalizing the pixel values from image to standardize the images and improve the overall performance of the model.

Once the dataset is preprocessed, the next step is to design and  create the OCR model. A custom CNN architecture is used to effectively learn the features and patterns present in the handwritten characters. The CNN model is built using appropriate linear stack of layers which comprise of convolutional layers, activation functions (ReLU), pooling layers, dense layers, and dropout layers, and suitable parameters to optimize its performance.
The preprocessed train set is fed into the CNN model for training. During the training process, the model learns from the input data, adjusting its internal parameters to minimize the loss function and improve its accuracy in recognizing handwritten Nepali words. The training involves iterative epochs, where the model updates its weights based on the optimization algorithm.

The model performance is also evaluated on the preprocessed test set after each epoch of training to assess the model's effectiveness in recognizing handwritten characters. 

The final model is the trained OCR model that has the highest accuracy on the testset and can effectively recognize handwritten Nepali characters.

#### ii) Character recognition and Prediction

Image acquisition --> preprocess --> segmentation --> detection --> saved model (from final model) --> Prediction --> Localization --> character label --> localized character image with prediction and accuracy

-	**Image Acquisition**: The first step is to acquire the image of the handwritten Nepali word. Image acquisition can be done by using a picture of the document from local storage or using a webcam.
-	**Preprocess**: The image is then preprocessed to enhance their quality and remove any noise or artifacts that may interfere with the character recognition process. Preprocessing techniques that were used include gray-scaling (converting a color image into a gray monochrome image), smoothing (Gaussian blur is used to remove noise and details present in an image), thresholding (Binary thresholding and Otsu’s thresholding in smoothen image to get binary image), erosion (to remove small objects and separate two connected objects) and resizing (to a fixed standard size (in this case, 32x32) that can be fed into the neural network).
-	**Segmentation**: The handwritten word image is then segmented into individual fragments of characters to create seperate character image. (done using a variety of techniques, such as thresholding, edge detection, or clustering)
-	**Character Detection**: The individual characters are then detected by identifying the bounding boxes of the characters in the image.
-	**Prediction**: The detected characters are then passed to the saved final model, obtained from the previous training phase, and used for prediction. The CNN model predicts the class for each detected character image.
-	**Localization and Character label**: Following prediction, the recognized characters are localized within the original word image to associate them with their corresponding positions. Each localized character is labeled with its recognized value, indicating the specific Nepali character it represents.

The predicted character label along with the accuracy of the recognition process is displayed in the interface.

### Performance
The system achieved an impressive accuracy of 98% on both the training and test datasets, indicating its successful development. Furthermore, a user interface was created for easy accessibility, ensuring that individuals seeking to recognize handwritten Devanagari characters can conveniently utilize the system.

### Applications
The developed system holds significant potential for diverse applications, including the digitization of handwritten documents, creation of searchable databases for handwritten materials, and providing handwriting feedback.

**Benefits**:

- Improved accuracy of handwritten Devanagari OCR
- Potential for a variety of applications

**Future Work**:

- Expand the training dataset to include vowel characters for classifying a wider range of handwritten Devanagari characters.
- Improve the accuracy of the system by collecting more data and training the CNN on a larger dataset
- Explore other applications for the system, such as digitizing handwritten documents, creating searchable databases of handwritten documents, and providing feedback on handwriting

------

Happy Learning :)

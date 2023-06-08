
**Project Title**: Handwritten Devnagari Optical Character Recognition

**Project Description**:

The Handwritten Devanagari Optical Character Recognition (OCR) project aimed to develop a system capable of recognizing handwritten Devanagari characters from a dataset of 46 classes, including 36 consonants and 10 numerals. The system utilized a Convolutional Neural Network (CNN) for feature extraction from the handwritten characters and employed OpenCV for image pre-processing.

The training process involved training the CNN on a dataset comprising _78,000_ samples of handwritten Devanagari characters. By learning the patterns present in the handwritten characters, the CNN was able to accurately classify them.

To enhance the accuracy of the CNN, OpenCV was utilized for image pre-processing tasks. These tasks encompassed resizing the images, cropping them, noise removal, background clutter elimination, segmentation, localization, and other essential steps. By standardizing the input images' size and quality, the pre-processing stage significantly contributed to improving the CNN's performance.

The system achieved an impressive accuracy of 98% on both the training and test datasets, indicating its successful development. Furthermore, a user interface was created for easy accessibility, ensuring that individuals seeking to recognize handwritten Devanagari characters can conveniently utilize the system.

The developed system holds significant potential for diverse applications, including the digitization of handwritten documents, creation of searchable databases for handwritten materials, and providing handwriting feedback.

**Technologies Used**:

- Convolutional Neural Network (CNN)
- OpenCV
- Python 

**Benefits**:

- Improved accuracy of handwritten Devanagari OCR
- Potential for a variety of applications

**Future Work**:

- Expand the training dataset to include vowel characters for classifying a wider range of handwritten Devanagari characters.
- Improve the accuracy of the system by collecting more data and training the CNN on a larger dataset
- Explore other applications for the system, such as digitizing handwritten documents, creating searchable databases of handwritten documents, and providing feedback on handwriting

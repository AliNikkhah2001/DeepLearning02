# Theoretical Insights

Upon careful observation, two notable aspects emerge from our analysis:

### 1. Precision in Identifying Changes:
Our model showcases remarkable precision in recognizing distinct alterations:
   - **Displacement:** Achieves an accuracy of approximately 88.54%, evident in spatial exchange due to convolutional model robustness.
   - **Rotation:** Demonstrates about 79.06% accuracy, partly managed by channel exchange.
   - **Scaling:** Presents the lowest accuracy (~71.15%) due to the holistic impact on the output, lacking specific feature extraction methods.

### 2. Dealing with Multiple Changes:
The model inclines towards False Positives, yielding commendable accuracies. However, it often predicts more than one change when there's a single alteration, reducing accuracy. Interestingly, it excels in detecting two changes and achieves its highest accuracy with three changes.

# Exercise Objectives

### Creating Dataset & Building Dataset Class
- Collect or create requisite data, organizing it into a PyTorch Dataset class with methods like `__init__`, `__len__`, and `__getitem__`.
- Manage loading, preprocessing, and retrieval of individual samples and labels.

### Augmentation Techniques
- Expand dataset diversity and robustness by applying various transformations such as flipping, rotating, or adjusting brightness.

### Implementing a 3-Class Classification Model
- Build a classification model in Python capable of categorizing input data into three pre-defined classes.
- Utilize appropriate algorithms and layers, handle multi-class classification tasks, and implement training and evaluation procedures.

This exercise encompasses dataset creation, augmentation, and building a classification model using Python classes for a multi-class classification problem.

# Health-Care-web_app
This is a Machine Learning and Deep Learning project that can predict the chances of getting diseases like Heart_Failure, Diabetes, Malaria and Tuberculosis.<br>
- Heart_Failure and Diabetes prediction used **Machine Learning Model** <br>
- Malaria and Tuberculosis prediction used **Deep Learning Model** where it utilised malaria parasitised cells and chest x-rays images.<br>
![Screenshot 2021-04-26 at 1 20 08 PM](https://user-images.githubusercontent.com/57981133/116048149-7e570400-a692-11eb-808c-d0185cff2599.jpg)

## About Data and Model:
### 1. Heart_Failure :

[Link to Heart_Failure dataset](https://www.kaggle.com/andrewmvd/heart-failure-clinical-data) <br>
<img width="990" alt="Screenshot 2021-05-05 at 1 36 45 PM" src="https://user-images.githubusercontent.com/57981133/117112742-2497bd80-ada7-11eb-9185-096180bb93f0.png">
Target variable is DEATH_EVENT which is boolean-type <br> 
[Heart_Failure.ipynb](https://github.com/rashmiranu/Health_App/blob/main/data/heart_failure.ipynb) <br>
**Model Used :** *Random Forest with GridSearch and hyper parameter tuning*

### 2. Diabetes :
[Link to Diabetes dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database) <br>
[Diabetes.ipynb](https://github.com/rashmiranu/Health_App/blob/main/data/diabetes.ipynb) <br>
**Model Used :** *Xgboost*

### 3. Malaria :
[Malaria image_data](https://www.kaggle.com/miracle9to9/files1) <br>
[Malaria.ipynb](https://www.kaggle.com/rashmiranu/malaria-cell-detection-cnn-inceptionv3?scriptVersionId=61231291) <br>
**Model Used :** *InceptionV3 model of Transfer Learning*

### 4. Tuberculosis :
[Tuberculosis iamge_data](https://www.kaggle.com/tawsifurrahman/tuberculosis-tb-chest-xray-dataset) <br>
[Tuberculosis.ipynb](https://www.kaggle.com/rashmiranu/tuberculosis-chest-x-ray-inceptionv3?scriptVersionId=61225867) <br>
**Model Used :** *InceptionV3 model of Transfer Learning*

## Tools Used :
- Python (3.8 version)
- Flask
- Sci-kit learn
- Tensorflow
- Pandas
- Numpy
- Pickle
- HTML and CSS

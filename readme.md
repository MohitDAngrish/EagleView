## Assumptions

 - Consistency while annotation is must.
 - There should not be wrong annotations i.e. car marked as person or person marked as car. As it would lead to classification loss in object detection.
 - A clearly visible object should not be left un-annotated, as it will correspond to a huge loss if model is able to predict correctly and there is no ground truth, which might end up confusing the model.
 - Bounding box should capture most of the feature of the particular object. If most of the features of an object is not visible no annotation must be done for it.


## Approach

### Data preparation

Since we have data in coco format, first convert it into pascal voc xml format. So we can use Tensorflow Object Detection API code to convert it into tf-records.

Create a virtual environment and use the **requirement.txt** to install the packages required.

```
pip3 install -r requirements.txt
```

Run jsonToXml.py to convert json format to xml format with two arguments:

 - path to json file (j)
 - path to image folder (i)    

The xml will be created in the image folder itself.
```
python jsonToXml.py -i "path_to_image_folder" -j "path_to_json_file"
```

Run dataPrepare.py to split data into **train and test** and create **label_map.pbtxt** file after doing few sanity checks : 

 - Whether the images and xml count matches
 - Check for missing xml or missing images
 - Whether bounding box exceeds image coordinates, if so trimming the bounding box so that bounding box coordinates are within the image itself.

The script take three mandatory arguments as input:

 - path to image folder (i)
 - image extension (ie)
 - path to store the train and test data (t)

```
python dataPrepare.py -i "path_to_image_folder" -ie "image_extension" -t "path_to_store_train_test_data"
```

### Install Tensorflow Object Detection API 

https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html

following the above mentioned blog, we can setup the object detection api and create train.tfrecord and test.tfrecord which will be used for training of the model itself.

## Metric

-   **Mean Average Precision(mAP)** is the current benchmark metric used by the computer vision research community to evaluate the robustness of object detection models.

-  **Precision** measures the prediction accuracy, whereas **recall** measures total numbers of predictions w.r.t ground truth.  

-   **mAP** encapsulates the tradeoff between precision and recall and maximizes the effect of both metrics.  

-   The object detection task's true and false positives are classified using the **IoU threshold**.  

-   Calculating mAP over an **IoU threshold range** avoids the ambiguity of picking the optimal IoU threshold for evaluating the model's accuracy.

**Best Test mAP :**
- **mAP@50** : 63.85
- **mAP@75** : 30.45
- **mAP** : 33.1 (IOU 0.5:0.95)

TrainEagle.ipynb contains the training code as well as the inference code(Object Oriented code). 

> **Note:** Choose ipynb, so that it will be easier to see the output of detections in the notebook.

## Assumptions Vs Reality

* The annotations are not consistent as there are images where human's who are very far or just there hand is visible are tagged as human and in some images the human who are clearly visible aren't tagged. This might confuse the model and untagged labels will be loss even though model was right.








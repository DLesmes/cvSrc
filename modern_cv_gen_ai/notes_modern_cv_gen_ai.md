## Focal loss & Smooth
...

## Data augmentation

It is a technique used in CV and on Object Detection tasks to increase the size of the trainign dataset using diferent image tranformations to the original data set to generate new sample images.

![](https://private-user-images.githubusercontent.com/61529697/379387688-ab35ab6b-0ed8-4639-b49d-5113583de97d.png?jwt=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Mjk3MDI1NTUsIm5iZiI6MTcyOTcwMjI1NSwicGF0aCI6Ii82MTUyOTY5Ny8zNzkzODc2ODgtYWIzNWFiNmItMGVkOC00NjM5LWI0OWQtNTExMzU4M2RlOTdkLnBuZz9YLUFtei1BbGdvcml0aG09QVdTNC1ITUFDLVNIQTI1NiZYLUFtei1DcmVkZW50aWFsPUFLSUFWQ09EWUxTQTUzUFFLNFpBJTJGMjAyNDEwMjMlMkZ1cy1lYXN0LTElMkZzMyUyRmF3czRfcmVxdWVzdCZYLUFtei1EYXRlPTIwMjQxMDIzVDE2NTA1NVomWC1BbXotRXhwaXJlcz0zMDAmWC1BbXotU2lnbmF0dXJlPTZjMWQ3ODRlMjMzOWNiYjczOGEzM2UyZDhjMGMyZTE1OWE5ZDEwYWM3ODI0NmMzZGI0Nzk1NzExYWQ2Yzk0ZTAmWC1BbXotU2lnbmVkSGVhZGVycz1ob3N0In0.R41M5O8jpw271zEr_GCc7HeKwQTVICLHqJ3P5ehf5Kk)

* Rotation
* Blur
* Ilumination
* Projective
* Scaling
* Translation
* Cropping
* Flipping (horizontal and vertical)
* Color adjustments
  * brightness
  * contrans
  * saturation
  * hue   

**This is usefull for:**
* Improve Generalization
* Mitigation of Overfitting
* Increase trainig data
* Handling Real-World Variability
* Enhanced Object Detection
* Reducion Annotation Effort

### Code sample

Keras Sequential model for data augmentation

```python
augmenter = keras.Sequential(
    layers=[
        keras_cv.layers.RandomFlip(
            mode="horizontal",
            bounding_box_format="xywh",
        ),
        keras_cv.layers.JitteredResize(
            target_size=(640, 640),
            scale_factor=(0.75, 1.3),
            bounding_box_format="xywh",
        ),
    ]
)
```
```
train_ds = train_ds.map(
   augmenter,
   num_parallel_calls=tf.data.AUTOTUNE
)
```

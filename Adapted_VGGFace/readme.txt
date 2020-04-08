Made Following:
https://www.kaggle.com/hsinwenchang/vggface-baseline-197x197

Dataset for COVID-19:
https://www.kaggle.com/khoongweihao/covid19-xray-dataset-train-test-sets/data

Siamese Network for decease classification with x-ray images.
Based on exsisting pretrained Siamese Newtork for face recognition.

Structure of Dataset:
                      dataset
             ____________|____________
            |                         |
         train                       test
        ___|____                   ___|___ 
       |        |                 |       |
  decease1  decease2         decease1  decease2
   __|____ ...
  |     |      ...
 img1  img2 ...

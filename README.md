### **Rethinking Video Rain Streak Removal: A New Synthesis Model and A Deraining Network with Video Rain Prior**

### Abstract

Existing video synthetic models and deraining methods are mostly built on a simplified video rain model assuming that rain streak layers of different video frames are uncorrelated, thereby producing degraded performance on real-world rainy videos. To address this problem, we devise a new video rain synthesis model with the concept of rain streak motions to enforce a consistency of rain layers between video frames, thereby generating more realistic rainy video data for network training, and then develop a recurrent disentangled deraining network (RDD-Net) based on our video rain model for boosting video deraining. More specifically, taking adjacent frames of a key frame as the input, our RDD-Net recurrently aggregates each adjacent frame and the key frame by a fusion module, and then devise a disentangle model to decouple the fused features by predicting not only a clean background layer and a rain layer, but also a rain streak motion layer. After that, we develop three attentive recovery modules to combine the decoupled features from different adjacent frames for predicting the final derained result of the key frame. Experiments on three widely-used benchmark datasets and a collected dataset, as well as real-world rainy videos show that our RDD-Net quantitatively and qualitatively outperforms state-of-the-art deraining methods.

### Requirements

* PyTorch >= 1.0.0

### Dataset

RainMotion [[dataset](https://drive.google.com/file/d/1905B_e2RgQGnyfHd5xpjB4lTLYoq0Jm4/view?usp=sharing)]

### Model

[model_motion.pth](https://drive.google.com/file/d/1CRV8wNEAX1qfVea3NmrxkVbMAB8rim6N/view?usp=sharing)

### Setup

- Training

  ```
  python main.py --batchSize 1 --data_dir data/RainMotion/Test --save_folder weights
  ```
- Testing
- ```
  python eval_derain.py --data_dir data/RainMotion/Test --model weights/model_motion.pth --output Results
  ```

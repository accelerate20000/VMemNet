# VMemNet
Video memorability measures the degree to which a video is remembered by different viewers and has shown great potential in various contexts, including advertising, education, and healthcare. Although image memorability has been well studied, research on video memorability is still in its early stages. Current methods have mostly focused on simple feature representation or decision fusion strategies, which neglect the interactions between the spatial and temporal domains. To tackle this issue, we introduce an end-to-end collaborative spatial-temporal network (VMemNet) for video memorability prediction. Specifically, VMemNet incorporates spatial and semantic-guided attention modules to model static local cues and dynamic global cues in videos, respectively. VMemNet integrates the spatial and semantic-guided attention modules by a dual-stream network architecture to simultaneously capture static local cues and dynamic global cues in videos. Specifically, the spatial attention module is to aggregate more memorable elements from spatial locations whereas the semantic-guided attention module is to achieve semantic alignment and intermediate fusion of local cues and global cues. Moreover, two types of loss functions with complementary decision rules are associated with corresponding attention modules to guide the training of the proposed network. Experimental results conducted on a publicly available dataset verify the effectiveness of the proposed VMemNet for video memorability prediction.


## Requiremenets
### Installation
* Linux with Python 3.7
* numpy 1.21.6
* scipy 1.7.3
* PyTorch 1.8.1+cu111 
* torchvision 0.9.1+cu111 
### Dataset
* The VMemNet conduct 8-fold cross-validation on [VideoMem_dataset](https://www.interdigital.com/data_sets/video-memorability-dataset), and report the average prediction performance as the final result. 

## Getting Started 
To ensure that the code runs successfully, please arrange the folder according to the following structure:
```
./VMemNet
    ../list/
    ../checkpoint/
    ../test.py
    ...

./VideoMem
    ../source/
    ../frames/
    ../dynamic-vectors/
    ../ground-truth_dev-set.csv
```
### Prepare keyframe
The VMemNet needs keyframes to extract static local features. You may need modify the dataset path in the `video_frame_prepare.py` before running:
```
python video_frame_prepare.py
```

### Download dynamic features
Download the [dynamic-vectors](https://drive.google.com/file/d/1XoSB7Wg1JHDyT3iwPfs5BRIWKwoNDKFc/view?usp=sharing) to the `Dataset` folder.

### Download trained model
Download the trained model to the `checkpoint` folder:
| Test set | Model | RC | 
| :---: | :---: | :---: |
| Fold 0 | [GoogleDrive](https://drive.google.com/file/d/1YQSv49gHFX8vHbceCN3V1ivLPDekAUH2/view?usp=sharing)| 0.524 |
| Fold 1 | [GoogleDrive](https://drive.google.com/file/d/16WQpbMKWL2YtByU37tFqeY0mzZN0QeM-/view?usp=sharing)| 0.518 |
| Fold 2 | [GoogleDrive](https://drive.google.com/file/d/1QwuJWySKf_YYstI49KglDLcVGXARrOO5/view?usp=sharing)| 0.515 |
| Fold 3 | [GoogleDrive](https://drive.google.com/file/d/1tQuu0GJB2JweG-EDKuL5uVL4AvgARcll/view?usp=sharing)| 0.521 |
| Fold 4 | [GoogleDrive](https://drive.google.com/file/d/1lY9zVVRTwh3DIT46GVufLvoANM9R5Xd8/view?usp=sharing)| 0.517 |
| Fold 5 | [GoogleDrive](https://drive.google.com/file/d/1Er7EcyXtB7I7WCCW4Dy10X06t3R1nL_C/view?usp=sharing)| 0.540 |
| Fold 6 | [GoogleDrive](https://drive.google.com/file/d/1Vu8EVwjazcPTpcgMTo0eodSZqu8orLJM/view?usp=sharing)| 0.535 |
| Fold 7 | [GoogleDrive](https://drive.google.com/file/d/1XBnSzh-aJF8nxewouJBYlgnONCq-JnYF/view?usp=sharing)| 0.561 |

### Testing
Before testing, please modify the file path of the dataset in test.py:
```
data_root_dir = '/media/Datasets/VideoMem/'
```
Then run:
```
python test.py
```

## Citing VMemNet
If you use this code or reference our paper in your work please cite this publication.


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
Download the [dynamic-vectors](https://drive.google.com/file/d/1XoSB7Wg1JHDyT3iwPfs5BRIWKwoNDKFc/view?usp=sharing) to the `VideoMem` folder.

### Download trained model
Download the trained model to the `checkpoint` folder:
| Test set | Model | RC | 
| :---: | :---: | :---: |
| Fold 0 | [GoogleDrive](https://drive.google.com/file/d/1XbIVw4DFNxyJXxGi7HD94X2Jl0x55wSo/view?usp=sharing)| 0.524 |
| Fold 1 | [GoogleDrive](https://drive.google.com/file/d/1Jmv_rvFvYkokuXMxrGZMLfxj2CuDO8m_/view?usp=sharing)| 0.518 |
| Fold 2 | [GoogleDrive](https://drive.google.com/file/d/16AUUxy2y1dYND57wBQH9_XdJJ5MKmxot/view?usp=sharing)| 0.515 |
| Fold 3 | [GoogleDrive](https://drive.google.com/file/d/1z1_GIdWI7NPKl1SXRCM1BP_3JSN0Cy-p/view?usp=sharing)| 0.521 |
| Fold 4 | [GoogleDrive](https://drive.google.com/file/d/1z4GOvELD-nfU6i0MdmVtt7qDU2gXTuqX/view?usp=sharing)| 0.517 |
| Fold 5 | [GoogleDrive](https://drive.google.com/file/d/13FcLzgaD0FX9exAtVf9XlMExC4dyM5nV/view?usp=sharing)| 0.540 |
| Fold 6 | [GoogleDrive](https://drive.google.com/file/d/1Qd1il6ek3CqpVtuN6FgAvTOHLmqn0OWc/view?usp=sharing)| 0.535 |
| Fold 7 | [GoogleDrive](https://drive.google.com/file/d/1f2XoKOLwxdLLQEp9J1f-fJPInuOcHoAr/view?usp=sharing)| 0.561 |

### Testing
Before testing, you may need modify the file path of the dataset and testing fold in `test.py`:
```
data_root_dir = '/media/Datasets/VideoMem/'
```
```
for i in range(7, 8): # test the 7 fold
    print('Starting fold-{0}\n\n'.format(i))
    choose_fold(i)
    print('Ending fold-{0}\n\n'.format(i))
```
Then run:
```
python test.py
```

## Citing VMemNet
If you use this code or reference our paper in your work please cite our paper:"VMemNet: A Deep Collaborative Spatial-Temporal Network With Attention Representation for Video Memorability Prediction," online version [VMemNet](https://ieeexplore.ieee.org/document/10298788).


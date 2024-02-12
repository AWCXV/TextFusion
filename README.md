# TextFusion
This is the offical implementation for the paper titled "TextFusion: Unveiling the Power of Textual Semantics for Controllable Image Fusion". [Paper Link](https://arxiv.org/abs/2312.14209)

<div align="center">
  <img src="Figs/motivation.png" width="600px" />
  <p>To generate appropriate fusion results for a specific scenario (different tasks or concerned objects), existing methods cannot realise it or require expensive retraining. The same goal can be achieved by simply adjusting the focused objectives of textual descriptionin our paradigm.</p>
</div>

## Highlight
- **For the first time**, the text modality is introduced to the image fusion field.
- A benchmark dataset.
- A textual attention assessment.

## To test
For the RGB and infrared image fusion:
```
python main_test_rgb_ir.py
```
**Tips**: you can directly adjust the "description" variable in .py file to control the image fusion results!

For the grayscale and infrared image fusion:

```
python main_test_gray_ir.py
```

## Environment
- Python 3.8.3
- Torch 2.1.1
- torchvision 0.16.1
- opencv-python 4.8.1.78


## Update
- 2024-2-8: The training set of our IVT dataset is available at [this link](https://drive.google.com/file/d/1poc5sWwAY63zNnxlTAPSJZLNml75k6aK/view?usp=sharing) now.
- 2024-2-12: The pre-trained model and test files are available now!

## Citation
If this work is helpful to you, please cite it as:
```
@article{cheng2023textfusion,
  title={TextFusion: Unveiling the Power of Textual Semantics for Controllable Image Fusion},
  author={Cheng, Chunyang and Xu, Tianyang and Wu, Xiao-Jun and Li, Hui and Li, Xi and Tang, Zhangyong and Kittler, Josef},
  journal={arXiv preprint arXiv:2312.14209},
  year={2023}
}
```

Our dataset is annotated based on the LLVIP dataset:
```
@inproceedings{jia2021llvip,
  title={LLVIP: A visible-infrared paired dataset for low-light vision},
  author={Jia, Xinyu and Zhu, Chuang and Li, Minzhen and Tang, Wenqi and Zhou, Wenli},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={3496--3504},
  year={2021}
}
```

# MedSAM2
Segment Anything in 3D Medical Images and Videos

Welcome to join our [mailing list](https://forms.gle/bLxGb5SEpdLCUChQ7) to get updates. We’re also actively looking to collaborate on annotating new large-scale 3D datasets. If you have unlabeled medical images or videos and want to share them with the community, let’s connect!

[[`Paper`](https://arxiv.org/abs/2504.03600)] [[`Project`](https://medsam2.github.io/)] [[Huggingface](https://huggingface.co/wanglab/MedSAM2)] [[`Dataset List`](https://medsam-datasetlist.github.io/)] [[`3D Slicer`](https://github.com/bowang-lab/MedSAMSlicer/tree/MedSAM2)] [[`Gradio App`](app.py)] [[`Colab`](https://colab.research.google.com/drive/1MKna9Sg9c78LNcrVyG58cQQmaePZq2k2?usp=sharing)] [[`BibTeX`](#Bibtex)]



## Installation 

- Create a virtual environment: `conda create -n medsam2 python=3.12 -y` and `conda activate medsam2` 
- Install [PyTorch](https://pytorch.org/get-started/locally/): `pip3 install torch torchvision` (Linux CUDA 12.4)
- Download code `git clone https://github.com/bowang-lab/MedSAM2.git && cd MedSAM2` and run `pip install -e ".[dev]"`
- Download checkpoints: `sh download.sh`
- Optional: Please install the following dependencies for gradio

```bash
sudo apt-get update
sudo apt-get install ffmpeg
pip install gradio==3.38.0
pip install numpy==1.26.3 
pip install ffmpeg-python 
pip install moviepy
```

## Download annotated datasets

- [CT_DeepLesion-MedSAM2](https://huggingface.co/datasets/wanglab/CT_DeepLesion-MedSAM2)



- [LLD-MMRI-MedSAM2](https://huggingface.co/datasets/wanglab/LLD-MMRI-MedSAM2) 

Note: Please also cite the raw [DeepLesion](https://doi.org/10.1117/1.JMI.5.3.036501) and [LLD-MMRI](https://www.sciencedirect.com/science/article/pii/S0893608025001078) dataset paper when using these datasets. 

- [RVENET](https://rvenet.github.io/dataset/): Waiting for authors' approval to release the mask.  


## Inference

### 3D medical image segmentation

- [CoLab](https://colab.research.google.com/drive/1MKna9Sg9c78LNcrVyG58cQQmaePZq2k2?usp=sharing): [MedSAM2_inference_CT_Lesion_Demo.ipynb](notebooks/MedSAM2_inference_CT_Lesion.ipynb)

- CMD

```bash
python medsam2_infer_3D_CT.py -i CT_DeepLesion/images -o CT_DeepLesion/segmentation
```

### Medical video segmentation

- [CoLab](https://colab.research.google.com/drive/1hyVyGh7qjTbFMuv568YcRoccdkNuDTaW?usp=sharing): [MedSAM2_Inference_Video_Demo.ipynb](notebooks/MedSAM2_Inference_Video.ipynb)


- CMD

```bash
python medsam2_infer_video.py -i input_video_path -m input_mask_path -o output_video_path 
```




### Gradio demo

```bash
python app.py
```

## Training

Specify dataset path in `sam2/configs/sam2.1_hiera_tiny_finetune512.yaml`

```bash
sbatch multi_node_train.sh
```

## Acknowledgements

- We highly appreciate all the challenge organizers and dataset owners for providing the public datasets to the community.
- We thank Meta AI for making the source code of [SAM2](https://github.com/facebookresearch/sam2) publicly available. Please also cite this paper when using MedSAM2. 


## Bibtex

```bash
@article{MedSAM2,
    title={MedSAM2: Segment Anything in 3D Medical Images and Videos},
    author={Ma, Jun and Yang, Zongxin and Kim, Sumin and Chen, Bihui and Baharoon, Mohammed and Fallahpour, Adibvafa and Asakereh, Reza and Lyu, Hongwei and Wang, Bo},
    journal={arXiv preprint arXiv:2504.03600},
    year={2025}
}
```



CUDA_VISIBLE_DEVICES=1 nohup python L2D_train.py --base_video_dir ./media/polypgen_seq/Train_new/Images_jpg/ --input_mask_dir ./media/polypgen_seq/Train_new/Masks/ --post_hoc_model_save_dir ./l2d_models/  -o ./media/Inference -e 'full_dataset_Polypgen' > output.log 2>&1 &
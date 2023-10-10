<span style="font-size: 25px;">**LOVECon: Text-driven Training-free <span style="text-decoration: underline;">Lo</span>ng <span style="text-decoration: underline;">V</span>ideo <span style="text-decoration: underline;">E</span>diting with <span style="text-decoration: underline;">Con</span>trolNet**

---
<span style="font-size: 20px;">Zhenyi Liao, Zhijie Deng</span>


# 👇Demo for **48** frames

<figure>
    <video width="768" height="256" controls>
    <source src="https://github.com/zhijie-group/LOVECon/assets/82353245/205e7eba-db9a-42f1-a3b1-81a04d3f3607" type="video/mp4">
    </video>
  <figcaption>| A woman with a white hat -> A woman with a pink hat |</figcaption>
</figure>

<figure>
    <video width="768" height="256" controls>
    <source src="Examples/Background/Flamingos-Flamingos in the blue water.mp4" type="video/mp4">
    </video>
  <figcaption>| Flamingos -> Flamingos in the blue water | </figcaption>
</figure>

<figure>
    <video width="768" height="256" controls>
    <source src="Examples/Style/A man sitting near the river-A man sitting near the river, ink wash painting style.mp4" type="video/mp4">
    </video>
  <figcaption>| A man sitting near the river -> A man sitting near the river, ink wash painting style | </figcaption>
</figure>

# 👇 Abstract

> TL; DR: LOVECon can perform text-driven long video editing with ControlNet in training-free and auto-regressive way with the help of cross-window attention and a video interpolation model, and precise control with mask obtained from DDIM inversion.

<details>
  <summary>Click to check the full version</summary>
  <p>Leveraging pre-trained conditional diffusion models for video editing without further tuning has gained increasing attention due to its promise in film production,
advertising, etc. Yet, seminal works in this line fall short in generation length,
temporal coherence, or fidelity to the source video. This paper aims to bridge the
gap, establishing a simple and effective baseline for training-free diffusion model-based long video editing. As suggested by prior arts, we build the pipeline upon
ControlNet, which excels at various image editing tasks based on text prompts. To
break down the length constraints caused by limited computational memory, we
split the long video into consecutive windows and develop a novel cross-window
attention mechanism to ensure the consistency of global style and maximize the
smoothness among windows. To achieve more accurate control, we extract the information from the source video via DDIM inversion and integrate the outcomes
into the latent states of the generations. We also incorporate a video frame interpolation model to mitigate the frame-level flickering issue. Extensive empirical
studies verify the superior efficacy of our method over competing baselines across
scenarios, including the replacement of the attributes of foreground objects, style
transfer, and background replacement. In particular, our method manages to edit
videos with up to 128 frames according to user requirements.。</p>
</details>

# 👇 Environment

You can create a virtual environment with running the following commands.

```
conda create -n lovecon python=3.8
conda activate lovecon
pip install -r requirements.txt
```

For the installation of Xformer, you can try the following commands.
```
wget https://github.com/ShivamShrirao/xformers-wheels/releases/download/4c06c79/xformers-0.0.15.dev0+4c06c79.d20221201-cp38-cp38-linux_x86_64.whl
pip install xformers-0.0.15.dev0+4c06c79.d20221201-cp38-cp38-linux_x86_64.whl
```


# 👇 Video Editing
## 😊 Model loading 
Download the pretrained model [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/tree/main) and [ControlNet](https://huggingface.co/lllyasviel/sd-controlnet-canny/tree/main) and arrange them like the following directorory structure.
- ckpt
  - stable-diffusion-v1-5
    - feature_extractor
    - safety_checker
    - scheduler
    - text_encoder
    - tokenizer
    - unet
    - vae
    - model_index.json
    - v1-5-pruned-emaonly.ckpt (or any other ckpts)
  - sd-controlnet-canny
    - config.json
    - diffusion_pytorch_model.bin (or any other ckpts)
  - sd-controlnet-hed
  - sd-controlnet-depth

You may use other diffusion models you interest or place them in your will, but do not forget to modify the beginning of the editing configuration, where it indicates the location of models.

## 😊 Reproduction
We categorize editing tasks into three types, attribute, background editing and style transfer. We give some examples in the folder **./Examples**. If interested, you could reproduce them by the following command.

```
CUDA_VISIBLE_DEVICES=0 python test_lovecon.py --config test_config/hat.yaml
```
You can specify the parameters of the configuration to get different editing results for the video.

## 😊 Edit Your Own Video 
You are welcome to edit your own video at your will. You only need to write a configuration file to specify the parameters in the editing process.

>The first three lines are for indicating the base diffusion model (default as Stable Diffusion v1.5) and ControlNet.\ 
Now **control_type** only support **canny**, **hed** and **depth**.

> Dataset_config is for prepaing the data and DDIM inversion. \
**Video_path** indicates the path of the source video in mp4 or gif form.\
**Prompt** is the source prompt.\
**N_sample_frame** specify the frame number we desire to edit from the source video, default as the first **n_sample_frame** frames. \
**Sampling_rate** represents the sampling schedule for the source video.\
The rest is for resize the video.

> Editing_config is for controlling the editing process.\
The first two lines indicate whether to use DDIM inversion and the attention, both default True.\
**Editing_type** can choose from **attribute**, **background** and **style**.
When masks are used, the boundary is not that accurate and may not cover the object, so **dilation_kernel** is used to boarden the boundary.
When we perform attribute editing, **editing_phrase** indicates the editing object, while in background editing, **editing_phrase** represents the unchanged object, and we set it empty string for style transfer.

You can make modifications to the source code according to your needs.


# 👇 Demo for **128** Frames
<figure>
    <video width="768" height="256" controls>
    <source src="Examples/long/A car-A red car.mp4" type="video/mp4">
    </video>
  <figcaption>| A car -> A red car | </figcaption>
</figure>

<figure>
    <video width="768" height="256" controls>
    <source src="Examples/long/A girl- A girl, disney style.mp4" type="video/mp4">
    </video>
  <figcaption>| A girl -> A girl, disney style | </figcaption>
</figure>

# 👇 Todo
■ Add shape editing to our pipeline.

# 👇 Acknowledgment
This repository borrows heavily from [FateZero](https://github.com/ChenyangQiQi/FateZero). Thanks to the authors for sharing their codes.


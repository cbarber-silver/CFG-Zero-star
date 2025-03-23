# CFG-Zero*: Improved Classifier-Free Guidance for Flow Matching Models

<div class="is-size-5 publication-authors", align="center">
              <!-- Paper authors -->
                <span class="author-block">
                  <a href="https://weichenfan.github.io/Weichen//" target="_blank">Weichen Fan</a><sup>1</sup>,</span>
                  <span class="author-block">
                    <a href="https://www.amberyzheng.com/" target="_blank">Amber Yijia Zheng</a><sup>2</sup>,</span>
                  <span class="author-block">
                  <a href="https://raymond-yeh.com/" target="_blank">Raymond A. Yeh</a><sup>2</sup>,</span>
                  <span class="author-block">
                    <a href="https://liuziwei7.github.io/" target="_blank">Ziwei Liu</a><sup>1✉</sup>
                  </span>
                  </div>
<div class="is-size-5 publication-authors", align="center">
                    <span class="author-block">S-Lab, Nanyang Technological University<sup>1</sup> &nbsp;&nbsp;&nbsp;&nbsp; Department of Computer Science, Purdue University <sup>2</sup> </span>
                    <span class="eql-cntrb"><small><br><sup>✉</sup>Corresponding Author.</small></span>
                  </div>

</p>

<div align="center">
                      <a href="">Paper</a> | 
                      <a href="https://weichenfan.github.io/webpage-cfg-zero-star/">Project Page</a> |
                      <a href="https://huggingface.co/spaces/weepiess2383/CFG-Zero-Star">Demo</a>
</div>

---

<!-- ![](https://img.shields.io/badge/Vchitect2.0-v0.1-darkcyan)
![](https://img.shields.io/github/stars/Vchitect/Vchitect-2.0)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2FVchitect%2FVchitect-2.0&count_bg=%23BDC4B7&title_bg=%2342C4A8&icon=octopusdeploy.svg&icon_color=%23E7E7E7&title=visitors&edge_flat=true)](https://hits.seeyoufarm.com)
[![Generic badge](https://img.shields.io/badge/DEMO-Vchitect2.0_Demo-<COLOR>.svg)](https://huggingface.co/spaces/Vchitect/Vchitect-2.0)
[![Generic badge](https://img.shields.io/badge/Checkpoint-red.svg)](https://huggingface.co/Vchitect/Vchitect-XL-2B) -->





## 🔥 Update and News
- [2025.03.17] 🔥 TBD


## :astonished: Gallery

<table class="center">
<tr>

  <td><img src="assets/repo_teaser.jpg"> </td> 
</tr>

<tr>
  <td><img src="assets/1_comparison.gif"> </td>
  <td><img src="assets/3_comparison.gif"> </td>
  <td><img src="assets/4_comparison.gif"> </td> 
</tr>

<tr>
  <td><img src="assets/7_comparison.gif"> </td>
  <td><img src="assets/8_comparison.gif"> </td>
  <td><img src="assets/16_comparison.gif"> </td> 
</tr>

</table>


## Installation

### 1. Create a conda environment and install PyTorch

Note: You may want to adjust the CUDA version [according to your driver version](https://docs.nvidia.com/deploy/cuda-compatibility/#default-to-minor-version).

  ```bash
  conda create -n CFG_Zero_Star python=3.10
  conda activate CFG_Zero_Star

  #Install pytorch according to your cuda version
  conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia

  #Install diffusers
  pip install transformers
  pip install git+https://github.com/huggingface/diffusers.git@main


  pip install gradio
  pip install imageio
  pip install ftfy
  ```

### 2. Install dependencies

  ```bash
  pip install -r requirements.txt
  ```

## Inference
Host a demo on your local machine.
~~~bash
python demo.py
~~~



## BibTex
```

```

## 🔑 License

This code is licensed under Apache-2.0. The framework is fully open for academic research and also allows free commercial usage.


## Disclaimer

We disclaim responsibility for user-generated content. The model was not trained to realistically represent people or events, so using it to generate such content is beyond the model's capabilities. It is prohibited for pornographic, violent and bloody content generation, and to generate content that is demeaning or harmful to people or their environment, culture, religion, etc. Users are solely liable for their actions. The project contributors are not legally affiliated with, nor accountable for users' behaviors. Use the generative model responsibly, adhering to ethical and legal standards.

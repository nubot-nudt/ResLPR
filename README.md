<p align="center">
  <img src="./docs/figs/ResLPR_logo.png" alt="Project Logo" width="400"/>
</p>

<p align="center">
  <a href="http://arxiv.org/abs/2503.12350" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-slategray">
  </a>
  
  <a href="https://www.bilibili.com/video/BV15VQVYAEbD/?spm_id_from=333.1387.homepage.video_card.click&vd_source=434433d4564b5c96e910bfcdd04d34a0" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-%F0%9F%8E%AC-pink">
  </a>

  <a href="https://pan.baidu.com/s/1EN9v7zhDOU2mxKGqlhQvzQ?pwd=9d3f" target='_blank'>
    <img src="https://img.shields.io/badge/Dataset-%F0%9F%94%97-lightblue">
  </a>  
</p>

<h3 align="center">ResLPR: A LiDAR Data Restoration Network and Benchmark for Robust Place Recognition Against Weather Corruptions</h3>

<p align="center">
  <a href="https://github.com/KuangWenqing">Wenqing Kuang</a><sup>1</sup>,
  <a href="https://github.com/Grandzxw">Xiongwei Zhao</a><sup>2</sup>,
  <a href="https://github.com/shenyehui">Yehui Shen</a><sup>1</sup>,
  <a href="https://scholar.google.com.sg/citations?user=OTBgvCYAAAAJ&hl=zh-CN&oi=ao">Congcong Wen</a><sup>3</sup>,
  <a href="https://scholar.google.com.hk/citations?hl=en&user=cp-6u7wAAAAJ">Huimin Lu</a><sup>1</sup>,
  Zongtan Zhou<sup>1</sup>,
  <a href="https://github.com/Chen-Xieyuanli">Xieyuanli Chen</a><sup>1</sup>
</p>

<p align="center"><sup>1</sup>National University of Defense Technology&nbsp;&nbsp;<sup>2</sup>Harbin Institute of Technology&nbsp;&nbsp;<sup>3</sup>New York University Abu Dhabi</p>

### About
***ResLPR*** is a benchmark for LiDAR-based place recognition under adverse weather. ***ResLPRNet***, a lightweight network, restores corrupted LiDAR scans using wavelet transform. It helps pretrained models enhance the robustness of place recognition in bad weather in a ***plug-and-play*** manner.
1. The visualizations of ***WeatherKITTI*** and ***WeatherNCLT*** proposed by this benchmark are as follows:
  <p align="center">
  <img src="./docs/figs/corrupted_level_vis.png" alt="Vis benchmark" width="800"/>
  </p>
Visualization of corrupted point clouds of varying severity in WeatherKITTI and WeatherNCLT. The 🔴 red points  in the point cloud signify the noise points, and the 🔵 blue points denote the lost points.

2. The visualizations of the results of three different preprocessing algorithms (including ResLPRNet) are as follows:
  <p align="center">
  <img src="./docs/figs/vis_preprocessing.png" alt="Vis preprocessing" width="800"/>
  </p>

3. The demo of the results of WeatherKITTI are as follows:
  <p align="center">
  <img src="./docs/figs/WeatherKITTI_demo.gif" alt="demo_WeatherKITTI" width="600"/>
  </p>
  
4. The demo of the results of WeatherNCLT are as follows:
  <p align="center">
  <img src="./docs/figs/WeatherNCLT_demo.gif" alt="demo_WeatherKITTI" width="600"/>
  </p>

### Data Preparation
Our datasets are hosted by Baidu Netdisk. Download the dataset via this [ResLPR datasets](https://pan.baidu.com/s/1EN9v7zhDOU2mxKGqlhQvzQ?pwd=9d3f). The dataset is of an excessive size, and it is currently undergoing continuous updates. Kindly refer to [DATA_PREPARE.md](docs/DATA_PREPARE.md) for the details to prepare the <sup>1</sup>`WeatherKITTI`, <sup>2</sup>`WeatherNCLT`.

### Getting Started
The code is currently being organized and will be open-sourced within 48 hours.


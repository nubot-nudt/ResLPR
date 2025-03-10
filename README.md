<p align="center">
  <img src="./docs/figs/ResLPR_logo.png" alt="Project Logo" width="400"/>
</p>

<h3 align="center">ResLPR: A LiDAR Data Restoration Network and Benchmark for Robust Place Recognition Against Weather Corruptions</h3>

<p align="center">
  <a href="https://github.com/KuangWenqing">Wenqing Kuang</a><sup>1</sup>,
  <a href="https://github.com/Grandzxw">Xiongwei Zhao</a><sup>2</sup>,
  <a href="https://github.com/shenyehui">Yehui Shen</a><sup>1</sup>,
  Congcong Wen<sup>3</sup>,
  Huimin Lu<sup>1</sup>,
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

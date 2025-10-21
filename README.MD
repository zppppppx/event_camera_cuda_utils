# Event Camera CUDA Utils

This repository contains a bunch of utils for event cameras implemented in CUDA. Will include some utils like normal flow extraction, images of warped events, etc.

To use this repo, you could directly extract a directory which is named by its utility, to your workding project using CMake with only a slight change. The corresponding library name is also the name of the directory.

## Utils

### Normal Flow Extraction

Currently implemented a simple ransac-based normal flow extraction, which indicates the direction of one event. This util accept a time surface as the input and the pixel value should be in the unit of seconds or you could modify corresponding thresholds to adapt to your scales. The normal flow is computed using its neighboring events and itself. Control parameters can be viewed in .cuh file.

The statistics tested on a single RTX 3060 is shown below in the table.

| Resolution      | Memory Consumption | Time Consumption |
|-----------------|-------------------|------------------|
| 1280 x 720      | ~350M               | ~7ms            |

A simple visualization of the normal flow is shown below, the normal flow only reflects the direction without any strength.

<p align="center">
  <img src="https://github.com/zppppppx/event_camera_cuda_utils/blob/main/resources/imgs/normal_flow_extraction/accumulated_event_image.jpg" width="45%" alt="Accumulated Event Image"/>
  <img src="https://github.com/zppppppx/event_camera_cuda_utils/blob/main/resources/imgs/normal_flow_extraction/normal_flow_image.jpg" width="45%" alt="Normal Flow Image"/>
</p>

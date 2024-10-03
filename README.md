# InpaintFusion

This project creates realistic product images by placing them in scenes based on a given text prompt. The goal is to produce visually coherent, natural-looking scenes for product photography, with a focus on e-commerce applications, using generative AI methods.

## Problem Statement
Recent progress in generative AI has made it possible to develop workflows that replicate product photography for e-commerce. This project tackles the challenge of placing an object’s image (with a white background) into a scene guided by a text prompt, aiming to produce a coherent and realistic result that aligns with the given description.

## Features
- Takes an object image and a text prompt
- Generates a background based on the text prompt with the object naturally placed in the scene.
- Enables video creation by generating multiple frames with dynamic object movement.

## Requirements
- torch
- numpy
- Pillow
- torchvision
- opencv-python
- google-colab
- segment-anything
- diffusers
- IPython
- requests
- Real-ESRGAN
- urllib

### Installation

To install the required libraries, run the following command:

```bash
pip install torch numpy Pillow torchvision opencv-python google-colab segment-anything diffusers IPython requests Real-ESRGAN urllib
```

### Getting Started
1. Access the Colab notebook using the provided link.
2. Ensure you're connected to a T4 GPU for optimal performance.
3. Execute the cells in order, following the sequential workflow.
4. Input any required information at the designated prompts.
5. For comprehensive guidance, refer to the detailed video tutorial included.

<br>

To run the project in Colab, click the link : [Open in Colab](https://colab.research.google.com/drive/1SByFkXVQgs7CEY6BPY33x9Xfo-6MZDFE?usp=sharing) 




## Approach

***For a complete walkthrough of the project and to better understand the approach, check out the video :*** [Watch the video](./video/approach.mp4)

<br>


The first approach is to use SAM (Segment Anything Model) to generate mask of the object's image.

This model automatically detects the objects in the image and creates a mask for it.

The next step is to use ‘dreamshaper-8-inpainting' model from the Hugging Face library. This model is specifically designed for inpainting tasks. 

Finally Stable Diffusion's Image to Video Model is used to generates video frames that transition smoothly from the initial image, creating a video sequence. 




### Reason for Using the DreamShaper Model for Inpainting
It is specifically fine tuned for inpainting tasks which can give better results compared to other general purpose models. 

Also it has a simplified pipeline setup and offers a streamlined approach for deploying inpainting tasks without needing extensive configuration, which is not always available in other models.

### Reason for Using Stable Diffusion For Image To Video Conversion
It generates a sequence of images that are similar enough to ensure a seamless transition for video creation.

Similar to inpainting, the diffusion process enhances each image, ensuring high-quality output.

It maintains alignment between the object, background, and text prompt across all frames, resulting in a cohesive and consistent video.

## Results

Input Image:

[Given input image](./results/input_image.png) 




Resulting Image with Generated Background:


[Output Image](./results/output_image.png)


Video Generated : [Generated Video](./results/output_video.mp4) 

## Future Improvements
- Improve scene realism by fine-tuning lighting and shadow matching.
- Experiment with different generative models to enhance scene variety.

<br>

### References
- https://huggingface.co/runwayml/stable-diffusion-inpainting
- https://huggingface.co/stabilityai/stable-diffusion-2-inpainting

<br>


## Author
Created and maintained by [Akshiti Agarwal](https://github.com/Akshiti24) 



For inquiries, feel free to reach out at [akshitiagarwal@gmail.com](mailto:akshitiagarwal@gmail.com)



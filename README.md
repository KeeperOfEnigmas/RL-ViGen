# Bachelor Thesis

## Papers
- [SVEA](https://arxiv.org/pdf/2107.00644)
- [RL-ViGen: A Reinforcement Learning Benchmark for Visual Generalization](https://arxiv.org/pdf/2307.10224)
- [A Comprehensive Survey of Data Augmentation in Visual Reinforcement Learning](https://arxiv.org/pdf/2210.04561v4)
- [A Survey of Zero-shot Generalisation in Deep Reinforcement Learning](https://dl.acm.org/doi/pdf/10.1613/jair.1.14174)
- [A Recipe for Unbounded Data Augmentation in Visual Reinforcement Learning](https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_26.pdf)
- [Learning Better with Less: Effective Augmentation for Sample-Efficient Visual Reinforcement Learning](https://arxiv.org/pdf/2305.16379)

## Links
- [RL-ViGen](https://github.com/gemcollector/RL-ViGen)
- [svea-vit](https://github.com/nicklashansen/svea-vit)
- [dmcontrol-generalization-benchmark](https://github.com/nicklashansen/dmcontrol-generalization-benchmark)

## ToDo
- [X] Different augmentations.
  - [x] Cutmix
  - [x] Cutout
  - [x] No aug
  - [x] Overlay
  - [x] Cropping
  - [x] Window
  - [x] Rotation
  - [x] Flipping horizontally
  - [x] Flipping vertically
  - [x] Convolution
  - [x] Mix augmentations
- [x] Different evaluation environments/augmentations.
  - [x] Color easy/hard
  - [x] Video easy/hard
  - [x] Vignette 
  - [x] Distortion 
  - [x] Cutmix
  - [x] Cutout 
  - [x] Overlay
  - [x] Cropping
  - [x] Window
  - [x] Rotation
  - [x] Flip_h
  - [x] Flip_v
  - [x] Convolution
- [ ] Experiment with different tests. (Pendulum, Cheetah, Humanoid)
- [ ] Different algorithms. (PIE-G)
- [ ] Multiple runs with different seeds. 
- [ ] Context aware augmentation with VLM ([A Comprehensive Survey of Data Augmentation in Visual Reinforcement Learning](https://arxiv.org/pdf/2210.04561v4): Section 3.5.1).

## Questions to answer
- How various augmentation strategies impact the performance and generalization of agents in visual environments, and why? 
- Do strong augmentations actually lead to better performance and weak augmentations to worse performance? 
- What is the breaking point when the effect of strong augmentation leads to better performance starts to lose its effectiveness? 
- How do different augmentations during training affect the results of evaluation? 
- Is context-aware augmentation a plausible technique to improve performance and generalization? How does it improve? 

## Things to note
- Mention different kinds of background changes. (Color, video...)
- Show baseline result without any augmentation for comparison.
- Mention frame stacks of images.
- Citation should be directly on the page.

This project contains a simple script showing how to merge the Batch Normalisation transform into each preceding convolutional layer.

[Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)

Background:

The batch normalisation transform has now become one of the base ingredients when training CNNs.  The transform can be described as:
<include transform>
The gamma and beta parameters are learned during training.

This transform uses next to no computation, compared to convolutional layers.  In practice though, the extra steps in the computational graph can incur quite a bit of overhead, especially on GPUs.  Additionally this is another layer type that you need to include in your deployed model.

Fortunately the Batch Normalisation transform is just a linear transform and therefore can be combined with the base convolution weights & bias.

[](https://github.com/pieterluitjens/Merge_Batch_Norm/blob/master/Remove%20BN%20Transform.pdf)

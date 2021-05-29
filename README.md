# wide-minima-density-hypothesis
Details about the wide minima density hypothesis and metrics to compute width of a minima

This repo presents the wide minima density hypothesis as proposed in the following paper:
*   [*Wide-minima Density Hypothesis and the Explore-Exploit Learning Rate Schedule*](https://arxiv.org/abs/2003.03977)

### Key contributions:
*   Hypothesis about minima density
*   A SOTA LR schedule that exploits the hypothesis and beats general baseline schedules
*   SOTA BLEU score on IWSLT'14 ( DE-EN )

### Measuring width of a minima
Keskar et.al 2016 (https://arxiv.org/abs/1609.04836) argue that wider minima generalize much better than sharper minima. The computation method in their work uses the compute expensive LBFGS-B second order method, which is hard to scale. We use a projected gradient ascent based method, which is first order in nature and very easy to implement/use. Here is the simplest way you can compute the width of the minima your model finds during training:

```python
>>> from minima_width_compute import ComputeKeskarSharpness
>>> cks = ComputeKeskarSharpness(model_final_ckpt, optimizer, criterion, trainloader, epsilon, lr, max_steps)
>>> width = cks.compute_sharpness()
```
Details about args required:
- `model_final_ckpt`: Checkpoint saved of the model after final training step
- `optimizer` : optimizer to use for projected gradient ascent ( SGD, Adam )
- `criterion` : criterion for computing loss (e.g. torch.nn.CrossEntropyLoss)
- `trainloader` : iterator over the training dataset (torch.utils.data.DataLoader)
- `epsilon` : epsilon value determines the local boundary around which minima witdh is computed (Default value : 1e-4)
- `lr` : lr for the optimizer to perform projected gradient ascent ( Default: 0.001)
- `max_steps` : max steps to compute the width (Default: 1000). Setting it too low could lead to the gradient ascent method not converging to an optimal point. 
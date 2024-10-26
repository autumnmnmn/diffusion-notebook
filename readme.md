# Diffusion Notebook

A jupyter notebook I use for tinkering with diffusion model inference. Focused on readability and configurability, and algebraic simplification of algorithms found in the literature and in existing implementations.

Implements euler, heun, runge-kutta solvers, as well as CFG, CFG++, and a novel sampling method that improves single-prediction inference as well as CFG.

Currently uses HuggingFace implementations for loading the Unet and CLIP models, but the HF AutoencoderKL has been replaced with a simplified SDXL decoder implementation.

## True Noise Removal

"True Noise Removal" (TNR) is an adjustment to CFG that incorporates the actual initial noise sample into the CFG calculation.

Typically, classifier-free guidance (CFG) is calculated as 

```
prediction = uncond + (cond - uncond) * scale

prediction = cond + (cond - uncond) * scale
```
where `cond` and `uncond` are conditioned and unconditioned model predictions, respectively, and `scale` is a manually selected parameter.

The above definitions are interchangeable, though scale must be incremented or decremented by 1 when switching between them.

Geometrically, the CFG calculation can be understood to be sampling two points from the noise-level isosurface predicted by the model, and selecting a point along their secant line, approximating another point on the isosurface in the direction of the conditioned prediction.

Experimentally, the magnitude of the `(cond - uncond)` difference term, prior to scaling, is on the order of a hundredth of the magnitude of either choice of base term. The base term dominates the combined prediction, but the scaled difference term is more significant in generating a coherent image.

These facts give rise to the intuition that the base term and difference term of CFG primarily fill the roles of removing noise and of adding signal, respectively. This is the impetus for the idea of true noise removal: in inference, we start with a purely noisy latent, and so we know exactly what the actual noise is. We can combine this with the usual CFG difference term to improve CFG, and to create a CFG-like formula for improving single-prediction inference:

```
prediction = noise + (cond - uncond) * scale

prediction = noise + (cond - noise) * scale
```

The actual True Noise Removal calculation in this codebase is slightly more complex, arrived at through iterative experimentation based on this intuition. The relative contribution of the true noise to the prediction needs to be adjusted according to the trained noise schedule. Further mathematical analysis is needed to establish a more rigorous explanation for its efficacy.

In the single-prediction case, prompt-coherence is mildly improved and the "washed-out" look of single-prediction inference is fully resolved by this method. Generated images have a full range of color and value. In the 2-prediction (and general N-prediction) case, the improvement over CFG is comparable to the improvement yielded by CFG++ (https://arxiv.org/pdf/2406.08070), another recent method for improving CFG inspired by a geometric interpretation. By my reckoning, TNR also seems to improve the stability of the RK4 solver.

`TODO`: add example results to repo & readme

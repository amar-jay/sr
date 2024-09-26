### Current Status:
  Ive implemented a VAE model for image upscaling. However not trained yet. The main reason is to understand how VAE works in general.
  Oh and btw, pytorch_lightning is used for training. if you don't want this overhead you can create yours 😂

#### Current Issues:
  - Vanishing and exploding gradients. [checkout this](./vae-testing).  The gradient distribution across activations are insane. Perhaps, next time I will try SwiGLU and see how it does. 

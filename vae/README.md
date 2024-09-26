# boilerplate code to train a mode using lightning

# features

- logging using tensorboard
  [[Paper]](https://arxiv.org/ftp/arxiv/papers/2312/2312.00752.pdf) A Simple Mamba implementation in Pytorch

### Description

- solves the quadratic scaling of input of the Transformer architecture
- focus on key data through the use of selective state spaces Mechanisms (SSMs)
- Somehow this uses a scan rather than a convolution, which much better hardware-wise
- Though RetNet and others have tried to solve this issue, this is said to be the most promising.
- Somehow this looks really similar to an RNN

### Accomplishments

- Outperforms Transformers in modelling DNA sequences and audio waveforms
- higher throughput than transformer of comparative size

### What are state spaces?

Don't know yet! seems to be a concept borrowed from maths or physics. However, this was what is said

$$
h'(t+1) = Ah(t) + Bx(t)
$$

$$
y(t) = Ch(t) + Dx(t)
$$

$Bx_T$ and $Dx_T$ seem to be skip connections. seems like
$x,y \in \mathbb{R}^{1}$ ,
$h(t) \in \mathbb{R}^{N}$ ,
$A,C \in \mathbb{R}^{N \times N}$ and
$B,D \in \mathbb{R}^{1 \times N}$ where $N$ is the hidden dimension.

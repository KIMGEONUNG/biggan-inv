## Experiments 
 
### using samplling real image
- I thought that the small number of real image is the cause of the poor 
performance of the discriminator. So, sampling the images from BIGGAN about 
50K and use this as real image when training discriminator

#### Results
- on-going

---

### Seperate Optimizer Generator Loss from Total loss 
- Adversarial training was extremely unstable. What if I descrease the 
learning rate for GAN loss as training or just very small learning rate for
adversarial training. I expect this scheme would make the training more stable.
- Primarily, just seperate the optimizer and use small learning rate for 
adversarial loss.

#### Results
- Not yet 

---

### multiple discriminator steps 
- In the original BIGGAN training setting, the number of step for discriminator 
is two when generator steps one. So, It would worth using original setting

#### Results
- Not yet 

---

### Very Small Learning Rate
- From now on, the training curve had been really flunctuated. 
I thought that if I use smaller learning rate, it would be more stable. 

#### Results
- Not yet 
---

## Trivial Info.

- Seemingly good quality MSE at 5e-4 

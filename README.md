## Experiments 
 
### using samplling real image (V3)
- I thought that the small number of real image is the cause of the poor 
performance of the discriminator. So, sampling the images from BIGGAN about 
50K and use this as real image when training discriminator

#### Results
- No improvement
---

### Small Learning Rate for Adversarial Training (V4)
- Adversarial training was extremely unstable. What if I descrease the 
learning rate for GAN loss as training or just very small learning rate for
adversarial training. I expect this scheme would make the training more stable.
- Primarily, just seperate the optimizer and use small learning rate for 
adversarial loss.

#### Results
- Ongoing

---
### Use Self supervised encoder training 
- 

---
### Use Pretrained Discriminator (V5)
- 

---

### multiple discriminator steps 
- In the original BIGGAN training setting, the number of step for discriminator 
is two when generator steps one. So, It would worth using original setting

#### Results
- Not yet 

---

## Trivial Info.

- Seemingly good quality MSE at 5e-4 


## Case Study

### Discriminator Pretrained 

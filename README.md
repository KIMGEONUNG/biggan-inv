## Experiments 
 
### Using samplling real image (V3)
- I thought that the small number of real image is the cause of the poor 
performance of the discriminator. So, sampling the images from BIGGAN about 
50K and use this as real image when training discriminator

#### Results
- No improvement
---

### Small Learning Rate for Adversarial Training (V4, V5)
- Adversarial training was extremely unstable. What if I descrease the 
learning rate for GAN loss as training or just very small learning rate for
adversarial training. I expect this scheme would make the training more stable.

- learning rate squence : 0.1 --> 0.01 --> 0.05 --> 0.03 --> ..?

#### Results
- I found a little bit stable learning rate setting. GAN loss fluctuated(this 
is good sign) and structure preserved not bad and a little bit poor vivid color
generated ... but I thouth that what the problem is not the loss balance but
the gan loss it self. So it may worth using learning rate decaying for GAN loss
- 

---
### Learning Rate Decaying for GAN Loss
- 

---
### Use Self Supervised Encoder Training, Naive Version (High Priority)
- 

#### Results
- In the most naive version, training data set

---
### Use Pretrained Discriminator (High Priority)
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

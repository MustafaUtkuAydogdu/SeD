import torch.nn.functional as F
from .vgg_model import PerceptualModel
import torch

class MSE:
    def __init__(self, weight, **others):
        self.weight = weight

    def __call__(self, image, fake_image_final, **others):
        fakes = fake_image_final  
        targets = image  

        mse_loss = F.mse_loss(fakes, targets, reduction='mean')
        return mse_loss
    
    
class VGG: 
    def __init__(self, weight, model_config, device, **others):
        self.weight = weight
        self.device = device
        self.vgg = PerceptualModel(**model_config).to(device)

    def __call__(self, image, fake_image_final, **others):
        fakes = fake_image_final 
        targets = image 
        vgg_loss = F.mse_loss(self.vgg(fakes), self.vgg(targets), reduction='mean')
        return vgg_loss
        
        
class Adversarial_G: 
    def __init__(self, weight, discriminator, **others):
        self.weight = weight
        self.discriminator = discriminator

    def __call__(self, fake_image_final, **others):
        fakes = fake_image_final 
        fake_scores = self.discriminator(fakes)
        g_loss = F.softplus(-fake_scores).mean()
        return g_loss
    
class Adversarial_D: 
    def __init__(self, discriminator, r1_gamma, r2_gamma, **others):
        self.discriminator = discriminator
        self.r1_gamma = r1_gamma
        self.r2_gamma = r2_gamma

    @staticmethod
    def compute_grad_penalty(images, scores):
        """Computes gradient penalty."""
        image_grad = torch.autograd.grad(
            outputs=scores.sum(),
            inputs=images,
            create_graph=True,
            retain_graph=True)[0].view(images.shape[0], -1)
        penalty = image_grad.pow(2).sum(dim=1).mean()
        return penalty

    def __call__(self, image, fake_image_final, **others):
        reals = image.clone()
        fake_image_final_clone = fake_image_final.detach().clone()
        fakes =  fake_image_final_clone 

        reals.requires_grad = True # To calculate gradient penalty
        real_scores = self.discriminator(reals)
        fake_scores = self.discriminator(fakes)
        loss_fake = F.softplus(fake_scores).mean()
        loss_real = F.softplus(-real_scores).mean()
        d_loss = loss_fake + loss_real
        real_grad_penalty = torch.zeros_like(d_loss)
        fake_grad_penalty = torch.zeros_like(d_loss)
        if self.r1_gamma != 0:
            real_grad_penalty = self.compute_grad_penalty(reals, real_scores)
        if self.r2_gamma != 0:
            fake_grad_penalty = self.compute_grad_penalty(fakes, fake_scores)
        reals.requires_grad = False
        return d_loss + real_grad_penalty * (self.r1_gamma * 0.5) + fake_grad_penalty * (self.r2_gamma * 0.5)
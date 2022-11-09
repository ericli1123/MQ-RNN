import torch

def loss(quantiles, local_decoder_output, cur_real_vals_tensor, device):
    total_loss = torch.tensor([0.0],device=device)
    for i in range(len(quantiles)):
      p = quantiles[i]
      errors = cur_real_vals_tensor - local_decoder_output[:,:,:,i]
      cur_loss = torch.max( (p-1)*errors, p*errors ) # CAUTION
      total_loss += torch.sum(cur_loss)
    return total_loss
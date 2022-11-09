import torch
import torch.nn as nn
from Encoder import Encoder
from Decoder import GlobalDecoder, LocalDecoder

class MQRNN(nn.Module):
    def __init__(self, 
                horizon_size:int, 
                hidden_size:int, 
                quantiles:list,
                dropout:float,
                layer_size:int,
                by_direction:bool,
                context_size:int, 
                covariate_size:int,
                device):
        super(MQRNN,self).__init__()
        print(f"device is: {device}")
        self.device = device
        self.horizon_size = horizon_size
        self.quantile_size = len(quantiles)
        self.quantiles = quantiles
        self.covariate_size = covariate_size
        quantile_size = self.quantile_size
        self.encoder = Encoder(horizon_size=horizon_size,
                               covariate_size=covariate_size,
                               hidden_size=hidden_size, 
                               dropout=dropout,
                               layer_size=layer_size,
                               by_direction=by_direction,
                               device=device)
        
        self.gdecoder = GlobalDecoder(hidden_size=hidden_size,
                                    covariate_size=covariate_size,
                                    horizon_size=horizon_size,
                                    context_size=context_size)
        self.ldecoder = LocalDecoder(covariate_size=covariate_size,
                                    quantile_size=quantile_size,
                                    context_size=context_size,
                                    quantiles=quantiles,
                                    horizon_size=horizon_size)
        self.encoder.double()
        self.gdecoder.double()
        self.ldecoder.double()
    def forward(self, cur_series_covariate_tensor, next_covariate_tensor,trainer: int = 1):
        enc_hs = self.encoder(cur_series_covariate_tensor) #[seq_len, batch_size, hidden_size]
        if trainer == 0:
            enc_hs = torch.unsqueeze(enc_hs[-1], dim=0)
        hidden_and_covariate = torch.cat([enc_hs, next_covariate_tensor], dim=2) #[seq_len, batch_size, hidden_size+covariate_size * horizon_size]
        gdecoder_output = self.gdecoder(hidden_and_covariate) #[seq_len, batch_size, (horizon_size+1)*context_size]
        
        quantile_size = self.ldecoder.quantile_size
        horizon_size = self.encoder.horizon_size

        local_decoder_input = torch.cat([gdecoder_output, next_covariate_tensor], dim=2) #[seq_len, batch_size,(horizon_size+1)*context_size + covariate_size * horizon_size]
        local_decoder_output = self.ldecoder( local_decoder_input) #[seq_len, batch_size, horizon_size* quantile_size]
        seq_len = local_decoder_output.shape[0]
        batch_size = local_decoder_output.shape[1]
        
        local_decoder_output = local_decoder_output.view(seq_len, batch_size, horizon_size, quantile_size) #[[seq_len, batch_size, horizon_size, quantile_size]]
        return local_decoder_output
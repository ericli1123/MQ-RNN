import torch
import torch.nn as nn

class GlobalDecoder(nn.Module):
    """
    全局 MLP
    input_size = hidden_size + covariate_size * horizon_size
    output_size: (horizon_size+1) * context_size
    """
    def __init__(self,
                 hidden_size:int, 
                 covariate_size:int,
                 horizon_size:int,
                 context_size:int):
        super(GlobalDecoder,self).__init__()
        self.hidden_size = hidden_size
        self.covariate_size = covariate_size
        self.horizon_size = horizon_size
        self.context_size = context_size

        self.linear1 = nn.Linear(in_features= hidden_size + covariate_size*horizon_size, 
                                out_features= horizon_size*hidden_size*3)
        
        self.linear2 = nn.Linear(in_features= horizon_size*hidden_size*3, 
                                out_features= horizon_size*hidden_size*2)
        
        self.linear3 = nn.Linear(in_features= horizon_size*hidden_size*2, 
                                out_features= (horizon_size+1)*context_size)

        self.activation = nn.ELU()
    def forward(self, input):
        layer1_output = self.linear1(input)
        layer1_output = self.activation(layer1_output)

        layer2_output = self.linear2(layer1_output)
        layer2_output = self.activation(layer2_output)

        layer3_output = self.linear3(layer2_output)
        layer3_output = self.activation(layer3_output)
        return layer3_output


class LocalDecoder(nn.Module):
    """
    局部 MLP

    input_size: (horizon_size+1)*context_size + horizon_size*covariate_size
    output_size: horizon_size * quantile_size
    """
    def __init__(self,
                covariate_size, 
                quantile_size,
                context_size,
                quantiles,
                horizon_size):
        super(LocalDecoder,self).__init__()
        self.covariate_size = covariate_size
        self.quantiles = quantiles
        self.quantile_size = quantile_size
        self.horizon_size = horizon_size
        self.context_size = context_size

        self.linear1 = nn.Linear(in_features= horizon_size*context_size + horizon_size* covariate_size + context_size,
                                 out_features= horizon_size* context_size)
        self.linear2 = nn.Linear(in_features= horizon_size* context_size,
                                 out_features= horizon_size* quantile_size)
        self.activation = nn.ELU()
    
    def forward(self,input):
        layer1_output = self.linear1(input)
        layer1_output = self.activation(layer1_output)

        layer2_output = self.linear2(layer1_output)
        layer2_output = self.activation(layer2_output)
        return layer2_output
import torch
from torch.utils.data import Dataset
import numpy as np

class MQRNN_dataset(Dataset):
    
    def __init__(self,
                series_df:np.array,
                covariate_df:np.array, 
                horizon_size:int,
                quantile_size:int):
        
        self.series_df = series_df
        self.covariate_df = covariate_df
        self.horizon_size = horizon_size
        self.quantile_size = quantile_size
        full_covariate = []
        final_covariate = []
        covariate_size = self.covariate_df.shape[2]
        print(f"self.covariate_df.shape : {self.covariate_df.shape}")
        for j in range(self.covariate_df.shape[0]):
            for i in range(1, self.covariate_df.shape[1] - horizon_size+1):
                cur_covariate = []
                cur_covariate.append(self.covariate_df[j,i:i+horizon_size,:])
                full_covariate.append(cur_covariate)
            print(np.array(full_covariate).shape)
            final_covariate.append(full_covariate)
            full_covariate = []
        final_covariate = np.array(final_covariate)
        print(f"full_covariate shape: {final_covariate.shape}")
        full_covariate = final_covariate.reshape(-1, self.covariate_df.shape[0], horizon_size * covariate_size)
        print(full_covariate.shape)
        self.next_covariate = full_covariate
    
    def __len__(self):
        return self.series_df.shape[1]
    
    def __getitem__(self,idx):
        cur_series = self.series_df[: -self.horizon_size, idx]
        cur_covariate = self.covariate_df[idx, :-self.horizon_size, :] 
    
        real_vals_list = []
        for i in range(1, self.horizon_size+1):
            real_vals_list.append(self.series_df[i: self.series_df.shape[0]-self.horizon_size+i, idx])
        real_vals_array = np.array(real_vals_list) #[horizon_size, seq_len]
        real_vals_array = real_vals_array.T #[seq_len, horizon_size]
        cur_series_tensor = torch.tensor(cur_series)
        
        cur_series_tensor = torch.unsqueeze(cur_series_tensor,dim=1) # [seq_len, 1]
        cur_covariate_tensor = torch.tensor(cur_covariate) #[seq_len, covariate_size]
        cur_series_covariate_tensor = torch.cat([cur_series_tensor, cur_covariate_tensor],dim=1)
        next_covariate_tensor = torch.tensor(self.next_covariate[:,idx,:]) #[seq_len, horizon_size * covariate_size]
        next_covariate_tensor = torch.squeeze(next_covariate_tensor,dim=0)

        cur_real_vals_tensor = torch.tensor(real_vals_array)
        return cur_series_covariate_tensor, next_covariate_tensor, cur_real_vals_tensor
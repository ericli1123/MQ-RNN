import torch
from torch.utils.data import DataLoader
from MQRNN import MQRNN
from MQRNN_dataset import MQRNN_dataset
from loss import loss
import numpy as np
import pandas as pd

def train(x, y, config):
    model = MQRNN(
        config["horizon_size"],
        config["hidden_size"],
        config["quantiles"],
        config['dropout'],
        config['layer_size'],
        config['by_direction'],
        config['context_size'],
        config['covariate_size'],
        config['device']
        )
    optimizer = torch.optim.AdamW(model.parameters(),lr=config['lr']) 
    dataset = MQRNN_dataset(y, x, config['horizon_size'], len(config['quantiles']))
    data_iter = DataLoader(dataset=dataset, batch_size=config['batch_size'], shuffle=True,num_workers=0)
    mins_loss = float('inf')
    for i in range(config['num_epochs']):
        epoch_loss_sum = 0.0
        total_sample = 0
        for (cur_series_tensor, cur_covariate_tensor, cur_real_vals_tensor) in data_iter:
            batch_size = cur_series_tensor.shape[0]
            seq_len = cur_series_tensor.shape[1]
            horizon_size = cur_covariate_tensor.shape[-1]
            total_sample += batch_size * seq_len * horizon_size
            cur_series_tensor = cur_series_tensor.double() #[batch_size, seq_len, 1+covariate_size]
            cur_covariate_tensor = cur_covariate_tensor.double() # [batch_size, seq_len, covariate_size * horizon_size]
            cur_real_vals_tensor = cur_real_vals_tensor.double() # [batch_size, seq_len, horizon_size]

            cur_series_tensor = cur_series_tensor.to(config['device'])
            cur_covariate_tensor = cur_covariate_tensor.to(config['device'])
            cur_real_vals_tensor = cur_real_vals_tensor.to(config['device'])
            cur_series_tensor = cur_series_tensor.permute(1,0,2) #[seq_len, batch_size, 1+covariate_size]
            cur_covariate_tensor = cur_covariate_tensor.permute(1,0,2) #[seq_len, batch_size, covariate_size * horizon_size]
            cur_real_vals_tensor = cur_real_vals_tensor.permute(1,0,2)
            optimizer.zero_grad()

            local_decoder_output = model(cur_series_tensor, cur_covariate_tensor)
            losses = loss(config['quantiles'], local_decoder_output, cur_real_vals_tensor, config['device'])
            losses.backward()
            optimizer.step()
            epoch_loss_sum += losses.item()
        epoch_loss_mean = epoch_loss_sum/ total_sample
        if epoch_loss_mean <= mins_loss:
            mins_loss = epoch_loss_mean
            torch_script_model = torch.jit.script(model)
            torch_script_model.save("./mqrnn.pt")
        if (i+1) % 10 == 0:
            print(f"epoch_num {i+1}, current loss is: {epoch_loss_mean}")
    
    
def predict(train_target_df, train_covariate_df, test_covariate_df, col_name, config):
    model = torch.jit.load('./mqrnn.pt')
    input_target_tensor = torch.tensor(train_target_df[[col_name]].to_numpy())
    full_covariate = train_covariate_df.to_numpy()
    full_covariate_tensor = torch.tensor(full_covariate)

    next_covariate = test_covariate_df.to_numpy()
    next_covariate = next_covariate.reshape(-1, config['horizon_size'] * config['covariate_size'])
    next_covariate_tensor = torch.tensor(next_covariate) #[1,horizon_size * covariate_size]
    next_covariate_tensor = torch.unsqueeze(next_covariate_tensor, dim=0)#[1, 1, horizon_size * covariate_size]

    input_target_tensor = input_target_tensor.to(config['device'])
    full_covariate_tensor = full_covariate_tensor.to(config['device'])
    next_covariate_tensor = next_covariate_tensor.to(config['device'])
    with torch.no_grad():
        input_target_covariate_tensor = torch.cat([input_target_tensor, full_covariate_tensor], dim=1)
        input_target_covariate_tensor = torch.unsqueeze(input_target_covariate_tensor, dim= 0) #[1, seq_len, 1+covariate_size]
        input_target_covariate_tensor = input_target_covariate_tensor.permute(1,0,2)
        output = model(input_target_covariate_tensor, next_covariate_tensor, 0)
        output = output.view(config['horizon_size'],len(config['quantiles']))
        output = output.cpu().numpy()
        result = {}
        for i in range(len(config['quantiles'])):
                result[config['quantiles'][i]] = output[:,i]
        return result

if __name__ == "__main__":
    config = {
    'horizon_size':7,
    'hidden_size':50,
    'quantiles': [0.3,0.5,0.8], 
    'dropout': 0.3,
    'layer_size':2,
    'by_direction':False,
    'lr': 1e-3,
    'batch_size': 2,
    'num_epochs':500,
    'context_size': 10,
    }
    time_range = pd.date_range('2010-01-01','2020-12-01',freq='12h')
    time_len = len(time_range)
    series_dict = {}
    for i in range(1,3):
        cur_vals = [np.sin(i*t) for t in range(time_len)]
        series_dict[i] = cur_vals
    y = pd.DataFrame(index=time_range, 
                            data=series_dict)
    horizon_size = config['horizon_size']
    x1 = pd.DataFrame(index=y.index,
                                data={'hour':y.index.hour,
                                    'dayofweek':y.index.dayofweek,
                                    'month': y.index.month
                                })
    x1_train = np.array(x1.iloc[:-horizon_size,:])
    x2_train = np.array(x1.iloc[:-horizon_size,:])
    y_train = np.array(y.iloc[:-horizon_size,:])
    x_train = np.array([x1_train,x2_train])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    config['covariate_size'] = x_train.shape[2]
    config['device'] = device
    #train(x_train, y_train, config)
    test_y = y.iloc[:-horizon_size,:]
    test_x = x1.iloc[:-horizon_size,:]
    x2_test = x1.iloc[-horizon_size:,:]
    y2_test = y.iloc[-horizon_size:,:]
    predict_result = predict(test_y,test_x,x2_test,2,config)
    import matplotlib.pyplot as plt
    plt.plot(predict_result[0.3], color = 'red', label='0.3-prediction')
    plt.plot(predict_result[0.5], color = 'green', label='0.5-prediction')
    plt.plot(predict_result[0.8], color = 'blue', label='0.8-prediction')
    plt.plot(y2_test[2].to_list(), color= 'black', label='real')
    plt.legend()
    plt.show()




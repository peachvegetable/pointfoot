from legged_gym.models.sequential import LSTMModel
from legged_gym.models.LSTM import LSTMModel
import torch.nn as nn
from pathlib import Path
import torch


def inference():
    input_dim = 27
    output_dim = 10
    hidden_dim = 512
    criterion = nn.MSELoss()

    bilstm = LSTMModel(input_dim, hidden_dim, output_dim)

    trajs_path = f'/home/peachvegetable/GAN/output/sim_trajs/sim_traj0'
    param_path = f'/home/peachvegetable/GAN/output/real_params/params0'
    model_path = f'/home/peachvegetable/GAN/output/models'

    path = Path(model_path)
    num_models = len(list(path.glob('*')))

    total_loss = 0
    cnt = 99

    for i in range(num_models):
        bilstm.load_state_dict(torch.load(f'/home/peachvegetable/GAN/output/models/model{cnt}'))

        traj = torch.load(trajs_path)
        real_param = torch.load(param_path)

        # lstm.eval()
        bilstm.eval()

        # lstm_output = lstm(traj)
        bilstm_output = bilstm(traj)

        bilstm_output = torch.mean(bilstm_output, dim=0)

        # loss_lstm = criterion(lstm_output, real_param)
        loss_bilstm = criterion(bilstm_output, real_param)

        # print(f'Loss LSTM: {loss_lstm.item()}')
        print(f'Loss BiLSTM: {loss_bilstm.item()}, Model: {bilstm_output}, real: {real_param}')

        total_loss += loss_bilstm
        cnt += 100

    avg_loss = total_loss / num_models
    print(f'Average loss: {avg_loss}')


if __name__ == '__main__':
    inference()



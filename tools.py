import numpy as np
import prettytable as pt
from torch.utils.data.dataloader import DataLoader
from sklearn.model_selection import train_test_split
import random
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)


def data_analyze(X, y, A):
    X_shape = len(X.shape)
    y_shape = len(y.shape)
    A_shape = len(A.shape)
    assert X_shape == 2 and y_shape == 1 and A_shape == 2
    analyze_result = {
        'rate for label=1': [],
        'rate for sensitive=1': [],
        'statistic parity': []
    }
    for i in range(A.shape[1]):
        a = A[:, i]
        a_0_index = np.where(a == 0)
        a_1_index = np.where(a == 1)
        y_a_0 = y[a_0_index]
        y_a_1 = y[a_1_index]

        analyze_result['rate for sensitive=1'].append(np.mean(a))
        analyze_result['statistic parity'].append(np.abs(np.mean(y_a_0) - np.mean(y_a_1)))
        analyze_result['rate for label=1'].append(np.mean(y_a_1))

    ls = [f'group-{i + 1}' for i in range(A.shape[1])]
    p = pt.PrettyTable(['', *ls])
    for k in analyze_result:
        p.add_row([k, *analyze_result[k]])
    p.align = 'l'
    print(p)
    return analyze_result


class LossRecoder:
    def __init__(self):
        self.loss_list = None
        self.lozz = []
        # self.loss_name = []

    def pack(self, losses: list):
        self.packs(*losses)

    def packs(self, *losses):
        self.lozz.append(losses)
        self.add(*losses)

    def sum(self, coefficients=None):
        if len(self.lozz) == 0:
            raise Exception("没有Loss需要叠加！")
        if coefficients is None:
            coefficients = [1 for _ in self.lozz[-1]]
        loss = self.lozz[-1][0] - self.lozz[-1][0]
        for i, s in enumerate(self.lozz[-1]):
            loss += coefficients[i] * s
        return loss

    def add(self, *losses):

        """
        :param losses: type Tensor list
        :return:
        """
        with torch.no_grad():
            if self.loss_list:
                for i in range(len(losses)):
                    if type(losses[i]) == torch.Tensor:
                        self.loss_list[i] += losses[i].cpu().item()
            else:
                self.loss_list = [i.cpu().item() for i in losses]

    def dict(self):
        ass = {f'loss_{i}': self.loss_list[i] for i in range(len(self.loss_list))
               }
        ass['loss'] = sum(self.loss_list)
        return ass


class DataStream:

    def __init__(self, data_name, batch_size=64, x_key='x',
                 y_key='y',
                 s_key='sensitive_sex'):
        data_raw = np.load(f'data_bin/{data_name}_train.npz', allow_pickle=True)
        # # 调试
        # import pdb; pdb.set_trace()
        if data_name == 'adult':
            s_key = 'sensitive_sex'
        if data_name in ['compas', 'compas_m','bank','default']:
            s_key = 'sensitives'
        if data_name in ['heritage', 'heritage(sex)', 'german_sex', 'german_age', 'german_ss', 'Hospital_readmission', 'lsac']:
            s_key = 's'
        # if data_name == 'heritage':
        #     x_key = 'x_train'
        #     y_key = 'y_train'
        #     s_key = 'a_train'
        # import pdb; pdb.set_trace()
        X = data_raw[x_key]
        Y = data_raw[y_key]
        A = data_raw[s_key]
        data_stream = X, Y, A
        tp = X.shape[0]
        X, Y, A = list(map(lambda x: torch.from_numpy(x).float(), data_stream))
        self.a_dim = A.shape[1] if len(A.shape) > 1 else 1
        self.x_dim = X.shape[1] if len(X.shape) > 1 else 1
        self.y_dim = Y.shape[1] if len(Y.shape) > 1 else 1
        #  # 调试
        # import pdb; pdb.set_trace()

        if data_name in ['compas_m', 'heritage', 'compas', 'heritage(sex)', 'german_sex', 'german_age', 'german_ss', 'Hospital_readmission', 'lsac', 'bank', 'default']:
            tpp = np.load(f'data_bin/{data_name}_verify.npz', allow_pickle=True)
            X2, y2, A2 = tpp[x_key], tpp[y_key], tpp[s_key]
            data_stream = X2, y2, A2
            X2, y2, A2 = list(map(lambda x: torch.from_numpy(x).float(), data_stream))
            X1, y1, A1 = X, Y, A
        elif data_name == 'adult':
            # import pdb;pdb.set_trace()
            X1, X2, y1, y2, A1, A2 = train_test_split(X, Y, A, test_size=0.2, random_state=1, shuffle=True)
        else:
            raise NotImplementedError
        self.split_desc = f'train:{tp}\tvalidate{X2.shape[0]}\ttext{X1.shape[0]}.'
        data_raw = np.load(f'data_bin/{data_name}_test.npz')
        X = data_raw[x_key]
        Y = data_raw[y_key]
        A = data_raw[s_key]
        data_stream = X, Y, A
        X, Y, A = list(map(lambda x: torch.from_numpy(x).float(), data_stream))
        # import pdb;pdb.set_trace()
        train_dataset = torch.utils.data.TensorDataset(X1, y1, A1)
        # import pdb;pdb.set_trace()
        val_dataset = torch.utils.data.TensorDataset(X2, y2, A2)
        test_dataset = torch.utils.data.TensorDataset(X, Y, A)
        train_load = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                                 drop_last=True)
        val_load = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_load = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True,
                                                drop_last=True)

        print('train/val/test:', X1.shape[0], X2.shape[0], X.shape[0])
        self.train_loader = train_load
        self.validate_loader = val_load
        self.test_loader = test_load
        self.data_dict = {
            'train': self.train_loader,
            'val': self.validate_loader,
            'test': self.test_loader
        }

from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import calculate_metrics
import torch, os, time, warnings
import numpy as np

from models import Transformer, DLinear
from data.dataloader import AgeData
from exp.config import DataConfig, Split
import datetime, time

warnings.filterwarnings('ignore')

class Exp_Forecast(object):
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting
        self.output_folder = os.path.join(args.result_path, setting)
        if not os.path.exists(self.output_folder):
            print(f'Output folder {self.output_folder} does not exist. Creating ..')
            os.makedirs(self.output_folder, exist_ok=True)
        
        self.model_dict = {
            'Transformer': Transformer,
            'DLinear': DLinear
        }
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        self.age_data = AgeData(
            data_path=os.path.join(args.root_path, args.data_path),
            date_index=DataConfig.date_index, 
            seq_len=DataConfig.seq_len, pred_len=DataConfig.pred_len,
            group_ids=DataConfig.group_ids, 
            static_reals=DataConfig.static_reals,
            observed_reals=DataConfig.observed_reals,
            known_reals=DataConfig.known_reals,
            targets=DataConfig.targets,
            scale=DataConfig.scale
        )
        
        self.total_data = self.age_data.read_df()
        self.train_data, self.val_data, self.test_data = self.age_data.split_data(
            self.total_data, Split.primary()
        )
        
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag:str='train'):
        assert flag in ['train', 'test', 'val'], \
            f'Flag {flag} not supported. Supported flags: [train, test, val]'
        
        if flag =='train': data = self.train_data
        elif flag == 'val': data = self.val_data
        else: data = self.test_data
        
        return self.age_data.create_tslib_timeseries(
            data, train = (flag == 'train')
        )

    def _select_optimizer(self):
        model_optim = torch.optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate
        )
        return model_optim

    def _select_criterion(self):
        criterion = torch.nn.MSELoss()
        return criterion

    def vali(self, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        
                f_dim = - self.args.n_targets
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item())
        
        if len(total_loss) == 0:
            print('Warning: no loss values found.')
            total_loss = np.inf
        else:
            total_loss = np.average(total_loss)
        
        self.model.train()
        return total_loss

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='val')
        _, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.result_path, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            self.output_folder, patience=self.args.patience, verbose=True
        )

        time_now = time.time()
        model_optim = self._select_optimizer()
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model_optim, patience=1, factor=0.5, verbose=True
        )
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)

                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        outputs = outputs[:, -self.args.pred_len:]
                        batch_y = batch_y[:, -self.args.pred_len:].to(self.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = - self.args.n_targets
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 500 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)
            test_loss = self.vali(test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            lr_scheduler.step(vali_loss)
            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        print(f'Loading the best model from {early_stopping.best_model_path}')
        self.model.load_state_dict(torch.load(early_stopping.best_model_path))

        return self.model

    def test(self, setting:str, load_model:bool=False, flag:str='test'):
        _, test_loader = self._get_data(flag)
        if load_model:
            best_model_path = os.path.join(self.output_folder, 'checkpoint.pth')
            print(f'loading best model from {best_model_path}')
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = - self.args.n_targets
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                preds.append(outputs)
                trues.append(batch_y)
                
                # if i % 20 == 0:
                #     input = batch_x.detach().cpu().numpy()
                #     gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                #     pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                #     visual(gt, pd, os.path.join(self.output_folder, str(i) + '.pdf'))

        # this line handles different size of batch. E.g. last batch can be < batch_size.
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        # possible since this problem has single target
        # otherwise use the next block
        preds = preds.reshape(*preds.shape[:2])
        trues = trues.reshape(*preds.shape[:2])
        # preds = preds.reshape((-1, *preds.shape[-2:]))
        # trues = trues.reshape((-1, *trues.shape[-2:]))
        
        print('Preds and Trues shape:', preds.shape, trues.shape)

        mae, mse, msle, r2 = calculate_metrics(preds, trues)
        result_string = f'mse:{mse:0.5g}, mae:{mae:0.5g}, msle: {msle:0.5g}, r2: {r2:0.5g}.'
        print(result_string)
        
        with open("result.txt", 'a') as output_file:
            output_file.write(setting + "  \n" + result_string + '\n\n')
            # The file is automatically closed when the 'with' block ends
            # contents will be auto flushed before closing

        np.save(os.path.join(self.output_folder, f'{flag}_metrics.npy'), np.array([mae, mse, msle, r2]))
        np.save(os.path.join(self.output_folder, f'{flag}_pred.npy'), preds)
        np.save(os.path.join(self.output_folder, f'{flag}_true.npy'), trues)

        return

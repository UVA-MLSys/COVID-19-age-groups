from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import calculate_metrics
import torch, os, time, warnings
import numpy as np
import pandas as pd
from models import DLinear, Autoformer, FEDformer, TimesNet, PatchTST
from data.data_factory import AgeData
from exp.config import Split, DataConfig
from datetime import datetime
from utils.plotter import PlotResults
from utils.tools import align_predictions

warnings.filterwarnings('ignore')

class Exp_Forecast(object):
    model_dict = {
        'DLinear': DLinear,
        'Autoformer': Autoformer,
        'FEDformer': FEDformer,
        'PatchTST': PatchTST,
        'TimesNet': TimesNet
    }
    
    def __init__(self, args, setting):
        self.args = args
        self.setting = setting
        self.output_folder = os.path.join(args.result_path, setting)
        
        if not os.path.exists(self.output_folder):
            print(f'Output folder {self.output_folder} does not exist. Creating ..')
            os.makedirs(self.output_folder, exist_ok=True)    
        print(f'Starting experiment. Result folder {self.output_folder}.')
        
        self.device = self._acquire_device()
        self.model = self._build_model().to(self.device)
        
        self.age_data = AgeData.build(args)
        self.total_data = self.age_data.read_df()
        train, val, test = self.age_data.split_data(
            self.total_data, Split.primary()
        )
        self.data_map = {
            'train': train, 'val':val, 'test': test
        }
        
        self.dataset_map = {
            # 'train': None, 'val':None, 'test': None
        }
        self.dataset_root = os.path.join(DataConfig.root_folder, args.data_path.split('.')[0])
        for flag in ['train', 'val', 'test']:
            path = os.path.join(self.dataset_root, f'{flag}.pt')
            ts_dataset = None
            if os.path.exists(path):
                print(f'Loading dataset from {path}')
                ts_dataset = torch.load(path, map_location=self.device)
            
            self.dataset_map[flag] = ts_dataset
        
    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(
                self.args.gpu) if not self.args.use_multi_gpu else self.args.devices
            device = torch.device(f'cuda:{self.args.gpu}')
            print(f'Use GPU: cuda:{self.args.gpu}')
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = torch.nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag:str='train', train=False):
        dataset, dataloader = self.age_data.create_tslib_timeseries(
            self.data_map[flag], train, 
            self.dataset_map[flag] # using a cached dataset help speeding up
        )
        if self.dataset_map[flag] is None:
            self.dataset_map[flag] = dataset
            output_path = os.path.join(self.dataset_root, f'{flag}.pt')
            if not os.path.exists(self.dataset_root):
                os.makedirs(self.dataset_root, exist_ok=True)
                
            print(f'Saving dataset at {output_path}')
            torch.save(dataset, output_path, pickle_protocol=4)
            
        return dataset, dataloader

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
        f_dim = - self.args.n_targets
        
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
        _, train_loader = self._get_data(flag='train', train=True)
        _, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.result_path, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            self.output_folder, patience=self.args.patience, verbose=True
        )

        time_now = time.time()
        start_time = datetime.now()
        model_optim = self._select_optimizer()
        # https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            model_optim, patience=0, factor=0.1, verbose=True, min_lr=1e-5
        )
        criterion = self._select_criterion()
        f_dim = - self.args.n_targets

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

                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                if (i + 1) % 500 == 0:
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7g}")
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

            print(f"Epoch: {epoch + 1} cost time: {(time.time() - epoch_time):0.5g}")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_loader, criterion)

            print(f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.5g} Vali Loss: {vali_loss:.5g}")
            early_stopping(vali_loss, self.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            elif early_stopping.counter > 0:
                best_model_path = os.path.join(self.output_folder, 'checkpoint.pth')
                self.model.load_state_dict(torch.load(best_model_path))

            lr_scheduler.step(vali_loss)
            # adjust_learning_rate(model_optim, epoch + 1, self.args)

        total_time = datetime.now() - start_time
        print(f'Train ended. Total time {total_time}, per epoch {total_time/(epoch+1)}\n')
        print(f'Loading the best model from {early_stopping.best_model_path}\n')
        self.model.load_state_dict(torch.load(early_stopping.best_model_path))

        return self.model
    
    def pred(self, load_model:bool=False, flag:str='test', return_index=False):
        test_dataset, test_loader = self._get_data(flag, train=False)
        if load_model: self.load_model()

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

        # this line handles different size of batch. E.g. last batch can be < batch_size.
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        # preds = preds.reshape((-1, *preds.shape[-2:]))
        # trues = trues.reshape((-1, *trues.shape[-2:]))
        
        print('Preds and Trues shape:', preds.shape, trues.shape)
        
        if return_index:
            predictions_index = pd.DataFrame(
                test_dataset.ranges, 
                columns=self.age_data.group_ids + [self.age_data.time_index]
            )
            return preds, trues, predictions_index
        else: 
            return preds, trues
        
    def load_model(self):
        best_model_path = os.path.join(self.output_folder, 'checkpoint.pth')
        try:
            print(f'loading best model from {best_model_path}')
            self.model.load_state_dict(torch.load(best_model_path))
        except:
            raise

    def test(
        self, setting:str, load_model:bool=False, flag:str='test',
        output_results=True, plot_results=True
    ):
        preds, trues, predictions_index = self.pred(
            load_model, flag, return_index=True
        )
        
        trues = self.age_data.upscale_target(trues)
        preds = self.age_data.upscale_target(preds)
        # since dataset was standard scaled, prediction can 
        # be negative after upscaling. but covid cases are non-negative
        preds = np.where(preds<0, 0, preds)
        
        if output_results:
            self.output_results(trues, preds, setting, flag)
            
        if plot_results:        
            plotter = PlotResults(
                self.output_folder, self.age_data.targets
            )
            df = self.data_map[flag]
            time_index = self.age_data.time_index
            predictions_index[time_index] += self.args.pred_len + df[time_index].min()
            
            pred_list = [
                preds[:, :, target] for target in range(preds.shape[-1])
            ]
            merged = align_predictions(
                df, predictions_index, pred_list, self.age_data, 
                remove_negative=True, upscale=False
            )
            plotter.summed_plot(merged, type=flag)

        return
    
    def output_results(self, trues, preds, setting, flag, filename='result.txt'):
        with open(filename, 'a') as output_file:
            # The file is automatically closed when the 'with' block ends
            # contents will be auto flushed before closing
            n_targets = preds.shape[-1]
            evaluation_metrics = np.zeros(shape=(n_targets, 4))
            
            for target_index in range(n_targets):
                mae, rmse, rmsle, r2 = calculate_metrics(preds[:, :, target_index], trues[:, :, target_index])
                result_string = f'{flag}: rmse:{rmse:0.5g}, mae:{mae:0.5g}, msle: {rmsle:0.5g}, r2: {r2:0.5g}'
                
                print(result_string)
                output_file.write(setting + ', ' + result_string + '\n')
                evaluation_metrics[target_index] = [mae, rmse, rmsle, r2]
        
            np.savetxt(os.path.join(self.output_folder, f'{flag}_metrics.txt'), np.array(evaluation_metrics))
        
        np.save(os.path.join(self.output_folder, f'{flag}_pred.npy'), preds)
        np.save(os.path.join(self.output_folder, f'{flag}_true.npy'), trues)
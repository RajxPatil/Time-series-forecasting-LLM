import argparse
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm
import time
import random
import numpy as np
import os
import csv
import datetime

from models import Autoformer, DLinear, TimeLLM
from data_provider.data_factory import data_provider
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

# =======================
# 🔧 Environment & Device Setup
# =======================
os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"
torch.set_default_dtype(torch.float32)

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

device = "cpu"
print(f"✅ Running on device: {device} (forced CPU to avoid MPS FFT bugs)")

# =======================
# 🔧 Argument Parser
# =======================
parser = argparse.ArgumentParser(description='Time-LLM')

parser.add_argument('--task_name', type=str, required=True, help='task name')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, default='Autoformer', help='model name')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, default='ETTm1', help='dataset type')
parser.add_argument('--root_path', type=str, default='./dataset', help='root path of data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
parser.add_argument('--features', type=str, default='M', help='forecasting task type')
parser.add_argument('--target', type=str, default='OT', help='target feature')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='checkpoint folder')

# forecasting task
parser.add_argument('--seq_len', type=int, default=96)
parser.add_argument('--label_len', type=int, default=48)
parser.add_argument('--pred_len', type=int, default=96)
parser.add_argument('--seasonal_patterns', type=str, default='Monthly')

# model define
parser.add_argument('--enc_in', type=int, default=7)
parser.add_argument('--dec_in', type=int, default=7)
parser.add_argument('--c_out', type=int, default=7)
parser.add_argument('--d_model', type=int, default=16)
parser.add_argument('--n_heads', type=int, default=8)
parser.add_argument('--e_layers', type=int, default=2)
parser.add_argument('--d_layers', type=int, default=1)
parser.add_argument('--d_ff', type=int, default=32)
parser.add_argument('--moving_avg', type=int, default=25)
parser.add_argument('--factor', type=int, default=1)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--embed', type=str, default='timeF')
parser.add_argument('--activation', type=str, default='gelu')
parser.add_argument('--output_attention', action='store_true')
parser.add_argument('--patch_len', type=int, default=16)
parser.add_argument('--stride', type=int, default=8)
parser.add_argument('--prompt_domain', type=int, default=0)
parser.add_argument('--llm_model', type=str, default='LLAMA')
parser.add_argument('--llm_dim', type=int, default=4096)
parser.add_argument('--num_workers', type=int, default=4)

# optimization
parser.add_argument('--itr', type=int, default=1)
parser.add_argument('--train_epochs', type=int, default=10)
parser.add_argument('--align_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--patience', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--des', type=str, default='test')
parser.add_argument('--loss', type=str, default='MSE')
parser.add_argument('--lradj', type=str, default='type1')
parser.add_argument('--pct_start', type=float, default=0.2)
parser.add_argument('--use_amp', action='store_true', default=False)
parser.add_argument('--llm_layers', type=int, default=6)
parser.add_argument('--percent', type=int, default=100)

# =======================
# 🔧 Utility: Convert to float32
# =======================
def convert_to_float32(batch):
    if isinstance(batch, torch.Tensor):
        return batch.float()
    elif isinstance(batch, (list, tuple)):
        return [convert_to_float32(x) for x in batch]
    elif isinstance(batch, dict):
        return {k: convert_to_float32(v) for k, v in batch.items()}
    else:
        return batch

# =======================
# 🚀 Main Function
# =======================
def main():
    args = parser.parse_args()

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    # Force CPU to avoid MPS/FFT fallback issues on macOS
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], cpu=True)

    for ii in range(args.itr):
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name, args.model_id, args.model, args.data, args.features,
            args.seq_len, args.label_len, args.pred_len, args.d_model, args.n_heads,
            args.e_layers, args.d_layers, args.d_ff, args.factor, args.embed, args.des, ii
        )

        train_data, train_loader = data_provider(args, 'train')
        vali_data, vali_loader = data_provider(args, 'val')
        test_data, test_loader = data_provider(args, 'test')

        if args.model == 'Autoformer':
            model = Autoformer.Model(args).float()
        elif args.model == 'DLinear':
            model = DLinear.Model(args).float()
        else:
            model = TimeLLM.Model(args).float()

        path = os.path.join(args.checkpoints, setting + '-' + args.model_comment)
        args.content = load_content(args)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        trained_parameters = [p for p in model.parameters() if p.requires_grad]
        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(
                optimizer=model_optim, steps_per_epoch=train_steps,
                pct_start=args.pct_start, epochs=args.train_epochs, max_lr=args.learning_rate
            )

        criterion = nn.MSELoss()
        mae_metric = nn.L1Loss()

        train_loader, vali_loader, test_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, vali_loader, test_loader, model, model_optim, scheduler
        )

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # =======================
        # 🔁 Training Loop
        # =======================
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            model.train()
            epoch_time = time.time()
            time_now = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in tqdm(enumerate(train_loader), total=len(train_loader)):
                if i == 0:  # only print once
                    print(f"🔍 batch_x shape: {batch_x.shape}")
                    print(f"🔍 batch_y shape: {batch_y.shape}")
                    print(f"🔍 batch_x_mark shape: {batch_x_mark.shape}")
                    print(f"🔍 batch_y_mark shape: {batch_y_mark.shape}")

                batch_x, batch_y, batch_x_mark, batch_y_mark = convert_to_float32(
                    (batch_x, batch_y, batch_x_mark, batch_y_mark)
                )

                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.to(accelerator.device)
                batch_y = batch_y.to(accelerator.device)
                batch_x_mark = batch_x_mark.to(accelerator.device)
                batch_y_mark = batch_y_mark.to(accelerator.device)

                dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(accelerator.device)
                dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(accelerator.device)

                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if args.output_attention \
                            else model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:]
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0] if args.output_attention \
                        else model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    f_dim = -1 if args.features == 'MS' else 0
                    outputs = outputs[:, -args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    loss = criterion(outputs, batch_y)
                    train_loss.append(loss.item())

                accelerator.backward(loss)
                model_optim.step()

            # =======================
            # 📊 Validation
            # =======================
            accelerator.print(f"Epoch {epoch+1}/{args.train_epochs} | Avg Train Loss: {np.average(train_loss):.6f}")
            vali_loss, vali_mae_loss = vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric)
            test_loss, test_mae_loss = vali(args, accelerator, model, test_data, test_loader, criterion, mae_metric)
            accelerator.print(f"Val Loss: {vali_loss:.6f}, Test Loss: {test_loss:.6f}, MAE: {test_mae_loss:.6f}")
            
            
            # ====== TRAIN LOGGING BLOCK (append right after the accelerator.print(...) above) ======
            
            # compute epoch elapsed if not already available
            # if you already track epoch_time above, you can use that; otherwise measure roughly here
            # (we try to use epoch_time if defined; else fallback to time.time())
            try:
                epoch_elapsed = time.time() - epoch_time
            except Exception:
                epoch_elapsed = None
            
            log_path = os.path.join("train_log.csv")
            file_exists = os.path.isfile(log_path)
            
            # device string
            device_str = "cpu"
            try:
                if torch.cuda.is_available():
                    device_str = "cuda"
                elif torch.backends.mps.is_available():
                    device_str = "mps"
                else:
                    device_str = "cpu"
            except Exception:
                device_str = "unknown"
            
            with open(log_path, mode="a", newline="") as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["timestamp","epoch","train_loss","val_loss","test_loss","mae_loss","epoch_time_s","device"])
                writer.writerow([datetime.datetime.utcnow().isoformat(), epoch + 1, train_loss, vali_loss, test_loss, test_mae_loss, epoch_elapsed if epoch_elapsed is not None else "", device_str])
            
            accelerator.print(f"✅ Logged epoch {epoch+1} to {log_path} (epoch_time_s={epoch_elapsed})")
            # ====== END TRAIN LOGGING BLOCK ======


            early_stopping(vali_loss, model, path)
            if early_stopping.early_stop:
                accelerator.print("Early stopping triggered.")
                break

            scheduler.step()

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        del_files('./checkpoints')
        accelerator.print('✅ Successfully deleted checkpoints.')

# =======================
# 🧠 Multiprocessing Safe Entry
# =======================
if __name__ == "__main__":
    import multiprocessing
    import torch

    # Force 'spawn' start method for multiprocessing (needed on macOS)
    multiprocessing.set_start_method("spawn", force=True)

    # Ensure all tensors default to float32 to avoid MPS float64 error
    torch.set_default_dtype(torch.float32)

    # Optional: patch dataset to convert any float64 tensors
    def convert_to_float32(batch):
        if isinstance(batch, torch.Tensor):
            return batch.float()
        elif isinstance(batch, (list, tuple)):
            return type(batch)(convert_to_float32(x) for x in batch)
        elif isinstance(batch, dict):
            return {k: convert_to_float32(v) for k, v in batch.items()}
        else:
            return batch

    # Monkey-patch DataLoader collate_fn to force float32 conversion
    from torch.utils.data import DataLoader
    old_iter = DataLoader.__iter__

    def new_iter(self):
        for batch in old_iter(self):
            yield convert_to_float32(batch)

    DataLoader.__iter__ = new_iter

    # Run main
    main()


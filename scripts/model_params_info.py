# scripts/model_params_info.py
import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from argparse import Namespace
from models import Autoformer

# Minimal full config similar to run_main.py
args = Namespace(
    task_name='long_term_forecast',
    is_training=1,
    model_id='test',
    model_comment='none',
    model='Autoformer',
    seed=2021,

    data='ETTm1',
    root_path='./dataset',
    data_path='ETTm1.csv',
    features='S',
    target='OT',
    loader='modal',
    freq='h',
    checkpoints='./checkpoints/',

    seq_len=96,
    label_len=48,
    pred_len=96,
    seasonal_patterns='Monthly',

    enc_in=1, dec_in=1, c_out=1,
    d_model=16, n_heads=8,
    e_layers=2, d_layers=1, d_ff=32,
    moving_avg=25, factor=1,
    dropout=0.1, embed='timeF',
    activation='relu', output_attention=False,
    patch_len=16, stride=8,
    prompt_domain=0, llm_model='LLAMA',
    llm_dim=4096,

    num_workers=0,
    itr=1, train_epochs=1, align_epochs=1,
    batch_size=8, eval_batch_size=8,
    patience=3, learning_rate=1e-4,
    des='test', loss='MSE', lradj='type1',
    pct_start=0.2, use_amp=False,
    llm_layers=6, percent=100
)

# Instantiate model
model = Autoformer.Model(args)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
pct = 100 * trainable_params / total_params

print(f"✅ Model: {args.model}")
print(f"Total Parameters: {total_params:,}")
print(f"Trainable Parameters: {trainable_params:,}")
print(f"Trainable Ratio: {pct:.4f}%")

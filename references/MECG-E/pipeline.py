#============================================================
#
#  Deep Learning BLW Filtering
#  Deep Learning pipelines
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import argparse
import torch
import datetime
import json
import os
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset

from sklearn.model_selection import train_test_split
from torch.optim import Adam
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
import numpy as np
from models.MECGE import MECGE


def train_dl(Dataset, experiment, n_type, config, nv, tb_writer, valid_epoch_interval=1, signal_size=512):

    print('Deep Learning pipeline: Training the model for exp ' + str(experiment))
    model_filepath = 'model_weight/' + experiment + f'_{n_type}_nv{nv}_weights.pth'
    [X_train, y_train, X_test, y_test] = Dataset

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3, shuffle=True, random_state=1)

    X_train = torch.FloatTensor(X_train)
    X_train = X_train.permute(0,2,1)
    
    y_train = torch.FloatTensor(y_train)
    y_train = y_train.permute(0,2,1)
    
    X_val = torch.FloatTensor(X_val)
    X_val = X_val.permute(0,2,1)
    
    y_val = torch.FloatTensor(y_val)
    y_val = y_val.permute(0,2,1)

    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0,2,1)
    
    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0,2,1)
    

    train_set = TensorDataset(y_train, X_train)
    val_set = TensorDataset(y_val, X_val)
    test_set = TensorDataset(y_test, X_test)
    
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=0)
    valid_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True, num_workers=0)
    
    # ==================
    # LOAD THE DL MODEL
    # ==================

    device = 'cuda:0'
    model = MECGE(config).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']["lr"], betas=[0.8, 0.99])
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99, last_epoch=-1)
    
    best_valid_loss = 1e10
    
    for epoch_no in range(config['train']["epochs"]):
        avg_loss = 0
        model.train()
        
        with tqdm(train_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                optimizer.zero_grad()
                
                loss = model(clean_batch, noisy_batch)
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.model.parameters(), 1.0)
                optimizer.step()
                avg_loss += loss.item()
                
                #ema.update(model)
                
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=True,
                )
            
            lr_scheduler.step()
            
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                        clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                        loss = model(clean_batch, noisy_batch)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=True,
                        )
            if tb_writer is not None:
                tb_writer.add_scalar('val_loss', avg_loss_valid / batch_no, epoch_no)
            
            if best_valid_loss > avg_loss_valid/batch_no:
                best_valid_loss = avg_loss_valid/batch_no
                print("\n best loss is updated to ",avg_loss_valid / batch_no,"at", epoch_no,)
                torch.save(model.state_dict(), model_filepath)



def test_dl(Dataset, experiment, n_type, config, nv, device, signal_size=512):
    
    model = MECGE(config).to(device)

    model_filepath = 'model_weight/' + experiment + f'_{n_type}_nv{nv}_weights.pth'
    model.load_state_dict(torch.load(model_filepath,map_location='cpu'))
    model.eval()
    print('Deep Learning pipeline: Testing the model')

    [train_set, train_set_GT, X_test, y_test] = Dataset

    X_test = torch.FloatTensor(X_test)
    X_test = X_test.permute(0,2,1)
    
    y_test = torch.FloatTensor(y_test)
    y_test = y_test.permute(0,2,1)

    test_set = TensorDataset(y_test, X_test)
    test_loader = DataLoader(test_set, batch_size=50, num_workers=0)
    
    
    # ==================
    # LOAD THE DL MODEL
    # ==================

    restored_sig = []
    with tqdm(test_loader) as it:
        for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
            clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)

            output = model.denoising(noisy_batch) #B,1,L
            clean_batch = clean_batch.permute(0, 2, 1)
            noisy_batch = noisy_batch.permute(0, 2, 1)
            output = output.permute(0, 2, 1) #B,L,1
            out_numpy = output.cpu().detach().numpy()
            
            restored_sig.append(out_numpy)
    
    y_pred = np.concatenate(restored_sig)
    X_test = X_test.permute(0, 2, 1).cpu().detach().numpy()
    y_test = y_test.permute(0, 2, 1).cpu().detach().numpy()
    #np.save(foldername + '/denoised.npy', restored_sig)

    return [X_test, y_test, y_pred]

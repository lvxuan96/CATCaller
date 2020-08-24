# -*- coding: utf-8 -*-

import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from litetr_encoder import LiteTransformerEncoder
import generate_dataset.constants as constants
from ctc.ctc_decoder import BeamCTCDecoder, GreedyDecoder
from ctc.ScheduledOptim import ScheduledOptim
from generate_dataset.train_dataloader import TrainBatchBasecallDataset, TrainBatchProvider
import time

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def train(model, optimizer, device, opt):
    print(model,flush=True)
    train_dataset = TrainBatchBasecallDataset(
        signal_dir=opt.train_signal_path, label_dir=opt.train_label_path)
    valid_dataset = TrainBatchBasecallDataset(
        signal_dir=opt.test_signal_path, label_dir=opt.test_label_path)

    list_charcter_error = []
    list_valid_loss = []
    start = time.time()
    show_shape = True
    for id in range(opt.epoch):
        train_provider = TrainBatchProvider(
            train_dataset, opt.batch_size, shuffle=True)
        valid_provider = TrainBatchProvider(
            valid_dataset, opt.batch_size, shuffle=False)
        # train
        model.train()
        total_loss = []
        batch_step = 0
        target_decoder = GreedyDecoder(
            '-ATCG ', blank_index=0)
        decoder = BeamCTCDecoder(
            '-ATCG ', cutoff_top_n=6, beam_width=3, blank_index=0)
        while True:
            batch = train_provider.next()
            signal, label = batch
            if signal is not None and label is not None:
                batch_step += 1
                if show_shape:
                    print('signal shape:', signal.size(), flush=True)
                    print('label shape:', label.size(), flush=True)
                    show_shape = False
                signal = signal.type(torch.FloatTensor).to(
                    device)  
                label = label.type(torch.LongTensor).to(
                    device)  
               
                # forward
                optimizer.zero_grad()
                signal_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
                enc_output, enc_output_lengths = model(
                    signal, signal_lengths)  
                

                log_probs = enc_output.transpose(1, 0).log_softmax(
                    dim=-1)  # (L,N,C), [256,32,6]
               
                assert signal.size(2) == 1
                # input_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
                target_lengths = label.ne(constants.PAD).sum(1)

                concat_label = torch.flatten(label)
                concat_label = concat_label[concat_label.lt(constants.PAD)]

                loss = F.ctc_loss(log_probs, concat_label, enc_output_lengths,
                                  target_lengths, blank=0, reduction='sum')
                loss.backward()

                optimizer.step_and_update_lr()
                total_loss.append(loss.item() / signal.size(0))
                if batch_step % opt.show_steps == 0:
                    print('training: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}'.format(
                        epoch=id,
                        step=batch_step,
                        loss=np.mean(total_loss),
                        t=(time.time() - start) / 60), flush=True)
                    start = time.time()
            else:
                print('training: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}'.format(
                    epoch=id,
                    step=batch_step,
                    loss=np.mean(total_loss),
                    t=(time.time() - start) / 60), flush=True)
                break
        # valid
        start = time.time()
        model.eval()
        total_loss = []
        batch_step = 0
        with torch.no_grad():
            total_wer, total_cer, num_tokens, num_chars = 0, 0, 0, 0
            while True:
                batch = valid_provider.next()
                signal, label = batch
                if signal is not None and label is not None:
                    batch_step += 1
                    signal = signal.type(torch.FloatTensor).to(device)
                    label = label.type(torch.LongTensor).to(device)

                    signal_lengths = signal.squeeze(
                        2).ne(constants.SIG_PAD).sum(1)
                    enc_output, enc_output_lengths = model(
                        signal, signal_lengths)

                    log_probs = enc_output.transpose(
                        1, 0).log_softmax(2)  # (L,N,C)


                    assert signal.size(2) == 1
                    # input_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
                    target_lengths = label.ne(constants.PAD).sum(1)
                    concat_label = torch.flatten(label)
                    concat_label = concat_label[concat_label.lt(constants.PAD)]

                    loss = F.ctc_loss(log_probs, concat_label, enc_output_lengths, target_lengths, blank=0,
                                      reduction='sum')
                    total_loss.append(loss.item() / signal.size(0))
                    if batch_step % opt.show_steps == 0:
                        print('validate: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}'.format(
                            epoch=id,
                            step=batch_step,
                            loss=np.mean(total_loss),
                            t=(time.time() - start) / 60), flush=True)
                        start = time.time()

                    log_probs = log_probs.transpose(1, 0)  # (N,L,C)
                    target_strings = target_decoder.convert_to_strings(
                        label, target_lengths)
                    decoded_output, _ = decoder.decode(
                        log_probs, enc_output_lengths)
                    # decoded_output, _ = target_decoder.decode(
                    #     log_probs, enc_output_lengths)
                    for x in range(len(label)):
                        transcript, reference = decoded_output[x][0], target_strings[x][0]
                        cer_inst = decoder.cer(transcript, reference)
                        total_cer += cer_inst
                        num_chars += len(reference)
                else:
                    break
            cer = float(total_cer) / num_chars
            list_charcter_error.append(cer)
            list_valid_loss.append(np.mean(total_loss))
            print(
                'validate: epoch {epoch:d}, loss {loss:.6f}, charcter error {cer:.3f} time: {time:.3f}'.format(
                    epoch=id,
                    loss=np.mean(total_loss),
                    cer=cer * 100,
                    time=(time.time() - start) / 60))
            start = time.time()
            if cer <= min(list_charcter_error) and np.mean(total_loss) <= min(list_valid_loss) and opt.store_model:
                model_state_dic = model.state_dict()
                model_name = ('%s_e%d_loss%.2f_cer%.2f.chkpt') %(opt.save_model, id, np.mean(total_loss),cer * 100)
                checkpoint = {'model': model_state_dic,
                              'settings': opt,
                              'epoch': id}
                torch.save(checkpoint, model_name)
                print('    - [Info] The checkpoint file has been updated.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_model', help="Save model path")
    parser.add_argument('--store_model',default=False)
    parser.add_argument('--from_model', help="load from exist model", default=None)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--train_signal_path', '-as', required=True)
    parser.add_argument('--train_label_path', '-al', required=True)
    parser.add_argument('--test_signal_path', '-es', required=True)
    parser.add_argument('--test_label_path', '-el', required=True)
    # parser.add_argument('-encoder_branch_type', nargs='+', default=None, type=str,
    #                     help='type of branches type:kernel:dim:head')
    parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', '-wd', default=0.01, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int)
    parser.add_argument('--epoch', type=int, default=200)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    # parser.add_argument('-raw_win_size', type=int, default=2048, help='signal window size')
    parser.add_argument('--label_vocab_size', type=int,
                        default=6)  # {0,1,2,3,4,5}
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--show_steps', type=int, default=500)
    parser.add_argument('--cuda', default=True)
    opt = parser.parse_args()
    device = torch.device('cuda' if opt.cuda else 'cpu')
    print(device, flush=True)


    if opt.from_model is None:
        model = LiteTransformerEncoder(d_model=opt.d_model,
                                       d_ff=opt.d_ff,
                                       n_head=opt.n_head,
                                       num_encoder_layers=opt.n_layers,
                                       label_vocab_size=opt.label_vocab_size,
                                       dropout=opt.dropout).to(device)

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        optim = torch.optim.Adam(
            model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
        optim_schedule = ScheduledOptim(
            optimizer=optim, d_model=opt.d_model, n_warmup_steps=opt.warmup_steps)
    else:
        checkpoint = torch.load(opt.from_model)
        model_opt = checkpoint['settings']
        # use trained model setting cover current setting
        # opt = model_opt
        model = LiteTransformerEncoder(d_model=model_opt.d_model,
                                       d_ff=model_opt.d_ff,
                                       n_head=model_opt.n_head,
                                       num_encoder_layers=model_opt.n_layers,
                                       label_vocab_size=model_opt.label_vocab_size,
                                       dropout=model_opt.dropout).to(device)
        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        optim = torch.optim.Adam(
            model.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999), weight_decay=opt.weight_decay)
        optim_schedule = ScheduledOptim(
            optimizer=optim, d_model=opt.d_model, n_warmup_steps=opt.warmup_steps)

    train(model=model,
          optimizer=optim_schedule,
          device=device, opt=opt)


if __name__ == "__main__":
    main()

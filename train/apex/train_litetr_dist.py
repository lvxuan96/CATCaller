# -*- coding: utf-8 -*-
import warnings
import random
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
from generate_dataset.train_dataloader_dist import TrainBatchBasecallDataset, TrainBatchProvider
import time
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

from apex import amp
from apex.parallel import convert_syncbn_model
from apex.parallel import DistributedDataParallel
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'


def main():
    parser = argparse.ArgumentParser(description='Basecalling Model Training')
    #data options
    parser.add_argument('--save_model', help="Save model path")
    parser.add_argument('--store_model',default=False)
    parser.add_argument('--from_model', default=None, help="load from exist model")
    parser.add_argument('--batch_size', default=128, type=int, help='batch size of all GPUs')
    parser.add_argument('--train_signal_path', '-as', required=True)
    parser.add_argument('--train_label_path', '-al', required=True)
    parser.add_argument('--test_signal_path', '-es', required=True)
    parser.add_argument('--test_label_path', '-el', required=True)
    #model options
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--n_head', type=int, default=8)
    parser.add_argument('--label_vocab_size', type=int,default=6)  # {0,1,2,3,4,5}
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--dropout', type=float, default=0.1)
    #training options
    parser.add_argument('--learning_rate', '-lr', default=1e-4, type=float)
    parser.add_argument('--weight_decay', '-wd', default=0.01, type=float)
    parser.add_argument('--warmup_steps', default=10000, type=int)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
    parser.add_argument('--show_steps', type=int, default=500)
    parser.add_argument('--local_rank', default=0, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--ngpus_per_node', type=int,default=4,
                        help='number of data loading workers')
    parser.add_argument('--opt_level', type=str, default='O0', help='O0/O1/O2/O3')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training. ')

    args = parser.parse_args()

    cudnn.benchmark = True
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.benchmark = False
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
        
    main_worker(args.local_rank,args.ngpus_per_node, args)

def main_worker(gpu, ngpus_per_node,args):
    torch.cuda.set_device(gpu)
    # device = torch.device(f'cuda:{args.local_rank}')
    assert torch.distributed.is_nccl_available()
    dist.init_process_group(backend='nccl', init_method='env://')
    assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."

    if args.from_model is None:
        print("=> creating model")
        model = LiteTransformerEncoder(d_model=args.d_model,
                                       d_ff=args.d_ff,
                                       n_head=args.n_head,
                                       num_encoder_layers=args.n_layers,
                                       label_vocab_size=args.label_vocab_size,
                                       dropout=args.dropout).cuda()
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        model = convert_syncbn_model(model) #Synchronize BN
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level) #need change
        model = DistributedDataParallel(model, delay_allreduce=True)
        
    else: #need change
        print("=> loading from existing model")
        checkpoint = torch.load(args.from_model)
        model_opt = checkpoint['settings']
        model = LiteTransformerEncoder(d_model=model_opt.d_model,
                                          d_ff=model_opt.d_ff,
                                          n_head=model_opt.n_head,
                                          num_encoder_layers=model_opt.n_layers,
                                          label_vocab_size=model_opt.label_vocab_size,
                                          dropout=model_opt.dropout).cuda()
        args.start_epoch =  checkpoint['epoch']
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), weight_decay=args.weight_decay)
        model = convert_syncbn_model(model) #Synchronize BN
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level) #need change
        model = DistributedDataParallel(model, delay_allreduce=True)
        model.load_state_dict(checkpoint['model'])
        amp.load_state_dict(checkpoint['amp'])


    model.cuda()


    optimizer = ScheduledOptim(
        optimizer=optimizer, d_model=args.d_model, n_warmup_steps=args.warmup_steps)

    args.batch_size = int(args.batch_size / ngpus_per_node)
    args.world_size = ngpus_per_node
    train_dataset = TrainBatchBasecallDataset(
        signal_dir=args.train_signal_path, label_dir=args.train_label_path)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,num_replicas=args.world_size,rank=gpu)
    valid_dataset = TrainBatchBasecallDataset(
        signal_dir=args.test_signal_path, label_dir=args.test_label_path)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(valid_dataset,num_replicas=args.world_size,rank=gpu)

    list_charcter_error = []
    list_valid_loss = []
    start = time.time()
    show_shape = True
    for epoch in range(args.start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)  # used for pytorch >= 1.2
        train_provider = TrainBatchProvider(
            train_dataset, args.batch_size, num_workers=0, train_sampler=train_sampler, pin_memory=True,  shuffle=(train_sampler is None))
        valid_provider = TrainBatchProvider(
            valid_dataset, args.batch_size, num_workers=0, train_sampler=valid_sampler, pin_memory=True,  shuffle=False)
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
                    print('gpu {} signal shape:{}'.format(gpu, signal.size()) , flush=True)
                    print('gpu {} label shape:{}'.format(gpu, label.size()) , flush=True)
                    show_shape = False
                signal = signal.type(torch.FloatTensor).cuda(non_blocking=True)
                label = label.type(torch.LongTensor).cuda(non_blocking=True)
                # forward
                optimizer.zero_grad()
                signal_lengths = signal.squeeze(2).ne(constants.SIG_PAD).sum(1)
                enc_output, enc_output_lengths = model(
                    signal, signal_lengths)  # (N,L,C), [32, 256, 6]

                log_probs = enc_output.transpose(1, 0).log_softmax(
                    dim=-1)  # (L,N,C), [256,32,6]
                assert signal.size(2) == 1
                target_lengths = label.ne(constants.PAD).sum(1)

                concat_label = torch.flatten(label)
                concat_label = concat_label[concat_label.lt(constants.PAD)]

                loss = F.ctc_loss(log_probs, concat_label, enc_output_lengths,
                                  target_lengths, blank=0, reduction='sum')

                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()

                optimizer.step_and_update_lr()
                total_loss.append(loss.item() / signal.size(0))
                if batch_step % args.show_steps == 0:
                    print('{gpu:d} training: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}'.format(
                        gpu=gpu,
                        epoch=epoch+1,
                        step=batch_step,
                        loss=np.mean(total_loss),
                        t=(time.time() - start) / 60), flush=True)
                    start = time.time()
            else:
                print('{gpu:d} training: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}'.format(
                    gpu=gpu,
                    epoch=epoch+1,
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
                    signal = signal.type(torch.FloatTensor).cuda(non_blocking=True)
                    label = label.type(torch.LongTensor).cuda(non_blocking=True)

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
                    if batch_step % args.show_steps == 0:
                        print('{gpu:d} validate: epoch {epoch:d}, step {step:d}, loss {loss:.6f}, time: {t:.3f}'.format(
                            gpu=gpu,
                            epoch=epoch+1,
                            step=batch_step,
                            loss=np.mean(total_loss),
                            t=(time.time() - start) / 60), flush=True)
                        start = time.time()

                    log_probs = log_probs.transpose(1, 0)  # (N,L,C)
                    target_strings = target_decoder.convert_to_strings(
                        label, target_lengths)
                    decoded_output, _ = decoder.decode(
                        log_probs, enc_output_lengths)

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
                '{gpu:d} validate: epoch {epoch:d}, loss {loss:.6f}, charcter error {cer:.3f} time: {time:.3f}'.format(
                    gpu=gpu,
                    epoch=epoch+1,
                    loss=np.mean(total_loss),
                    cer=cer * 100,
                    time=(time.time() - start) / 60))
            start = time.time()
            # remember best_cer and save checkpoint
            if cer <= min(list_charcter_error) and np.mean(total_loss) <= min(list_valid_loss) and args.store_model:
                if gpu==0:
                    model_name = ('%s_e%d_loss%.2f_cer%.2f.chkpt') %(args.save_model, epoch+1, np.mean(total_loss),cer * 100)
                    checkpoint = {'model': model.state_dict(),
                                  # 'optimizer': optimizer.state_dict(),
                                  'settings': args,
                                  'amp': amp.state_dict(),
                                  'epoch': epoch+1}
                    torch.save(checkpoint, model_name)
                    print('    - [Info] The checkpoint file has been updated.', flush=True)
            # dist.barrier()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

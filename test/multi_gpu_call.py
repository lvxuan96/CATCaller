

import argparse
import os
import copy
import time
import datetime
import numpy as np
from litetr_encoder import LiteTransformerEncoder
import torch
import torch.nn as nn
import torch.utils.data as Data
import constants as Constants
from ctc_decoder import BeamCTCDecoder, GreedyDecoder
from fast_ctc_decode import beam_search
# from test_dataloader import TestBatchProvider
from tqdm import tqdm
from torch import multiprocessing as mp
import subprocess
torch.multiprocessing.set_sharing_strategy('file_system')

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,4' #need change

class CallDataset(Data.Dataset):
    def __init__(self, argv):
        self.records_dir = argv.records_dir
        self.filenames = os.listdir(argv.records_dir)
        self.count = len(self.filenames)
        self.half = argv.half

    def __len__(self):
        return self.count

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        signal = np.load(self.records_dir + '/' + fname)
        if self.half:
            signal = signal.astype(np.float16)
        read_id = os.path.splitext(fname)[0]
        return read_id, signal

class Call(nn.Module):
    def __init__(self, opt):
        super(Call, self).__init__()
        checkpoint = torch.load(opt.model)
        model_opt = checkpoint['settings']
        self.model = LiteTransformerEncoder(d_model=model_opt.d_model,
                                            d_ff=model_opt.d_ff,
                                            n_head=model_opt.n_head,
                                            num_encoder_layers=model_opt.n_layers,
                                            label_vocab_size=model_opt.label_vocab_size,
                                            dropout=model_opt.dropout)
        # self.model = nn.DataParallel(self.model, device_ids=[0,1,2,3]) #solution1
        # self.model.load_state_dict(checkpoint['model'])
        state_dict = checkpoint['model']
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)  #solution2
        # print('[Info] Trained model state loaded.')
    def forward(self, signal, signal_lengths):
        return self.model(signal, signal_lengths)




def encode_process(argv,call_dataset, gpu_rank, qdecoder):
    # qdecoder = mp.Queue()
    # load model and prepare dataloader
    torch.set_grad_enabled(False)
    argv.device = torch.device("cuda:{}".format(gpu_rank))
    # print(argv.device, gpu_rank)
    model = Call(argv).to(argv.device)
    model.eval()
    model.share_memory()
    if argv.half:
        model.half()


    test_sampler = Data.distributed.DistributedSampler(call_dataset, num_replicas=argv.num_gpu, rank=gpu_rank)
    data_iter = Data.DataLoader(dataset=call_dataset, batch_size=1, num_workers=0, sampler=test_sampler, pin_memory=True, shuffle=False)
   
    read_id_list = []
    row_num_list = []
    log_probs_list = []
    output_lengths_list = []
    encoded_num = 0
    for batch in tqdm(data_iter):
        read_id, signal = batch
        read_id = read_id[0]
        signal = signal[0]
        # print(read_id,signal.shape[0], flush=True)
        row_num = 0
        signal_segs = signal.shape[0]
        for i in range(signal_segs // 64 + 1):
            if i != signal_segs // 64:
                signal_batch = signal[i * 64:(i + 1) * 64]
            elif signal_segs % 64 != 0:
                signal_batch = signal[i * 64:]
            else:
                continue
            if argv.half:
                signal_batch = torch.HalfTensor(signal_batch).to(argv.device)
            else:
                signal_batch = torch.FloatTensor(signal_batch).to(argv.device)

            signal_lengths = signal_batch.squeeze(2).ne(Constants.SIG_PAD).sum(1)
            output, output_lengths = model(signal_batch, signal_lengths)
            log_probs = output.log_softmax(2)
            row_num += log_probs.shape[0]
            log_probs_list.append(log_probs.cpu().detach())
            output_lengths_list.append(output_lengths.cpu().detach())
        read_id_list.append(read_id)
        row_num_list.append(row_num)
        # print(gpu_rank, read_id, flush=True)
        # encoded_num += 1
        # print('\r encoded_num={}'.format(encoded_num),end="")
        if len(read_id_list) == 50:
            qdecoder.put((read_id_list, row_num_list, log_probs_list, output_lengths_list))
            read_id_list = []
            row_num_list = []
            log_probs_list = []
            output_lengths_list = []
    if len(read_id_list) > 0:
        qdecoder.put((read_id_list, row_num_list, log_probs_list, output_lengths_list))
    qdecoder.put(0)

def writer(argv,item,write_mutex):
    read_id_list, row_num_list, log_probs_list, output_lengths_list = item
    assert len(read_id_list) == len(row_num_list)
    probs = torch.cat(log_probs_list)
    lengths = torch.cat(output_lengths_list)

    ##C++ beam search
    # decoder = BeamCTCDecoder('-ATCG ', blank_index=0, alpha=0.0, lm_path=None, beta=0.0, cutoff_top_n=6,
    #                              cutoff_prob=1.0, beam_width=3, num_processes=8)
    # decoded_output, offsets = decoder.decode(probs, lengths)
    ##rust beam search
    probs = probs.numpy().astype(np.float32)
    decoded_output = []
    for row in range(probs.shape[0]):
        probs_row = probs[row]
        probs_row = probs_row[0:int(lengths[row]), :]
        probs_row = np.exp(probs_row)
        # print(type(probs_row), type(base_list))
        seq, _ = beam_search(probs_row, list('-ATCG '))
        decoded_output.append(seq)

    while write_mutex.value != 1:
        time.sleep(0.2)

    fw=open(argv.outpath,'a')
    write_mutex.value = 0
    idx=0
    for x in range(len(row_num_list)):
        row_num = row_num_list[x]
        read_id = read_id_list[x]
        ## C++ beam search
        # transcript = [v[0] for v in decoded_output[idx:idx + row_num]]
        ##rust beam search
        transcript = [v for v in decoded_output[idx:idx + row_num]]
        idx = idx + row_num
        transcript = ''.join(transcript)
        transcript = transcript.replace(' ', '')
        if len(transcript) > 0:
            fw.write('>' + str(read_id) + '\n')
            fw.write(transcript + '\n')
    fw.close()
    # print('\n end decode, time = ', datetime.datetime.now()-start_decode)
    write_mutex.value = 1



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', required=True)
    parser.add_argument('-records_dir', required=True)
    parser.add_argument('-output', required=True)
    parser.add_argument('-threads', type=int, default=8)
    parser.add_argument('-half', action='store_true', default=False)
    argv = parser.parse_args()


    if not os.path.exists(argv.output):
        os.makedirs(argv.output)
    if os.path.exists(os.path.join(argv.output, 'out.fasta')):
        os.remove(os.path.join(argv.output, 'out.fasta'))
    argv.outpath = os.path.join(argv.output, 'out.fasta')

    call_dataset = CallDataset(argv)
    len_read = len(call_dataset)
   
    qdecoder = mp.Manager().Queue()
   

    nDevice = int(subprocess.getoutput("nvidia-smi -L | grep GPU |wc -l"))
    argv.num_gpu = nDevice
    encode_pool = mp.Pool(nDevice)
    for device_idx in range(nDevice):
        encode_pool.apply_async(func=encode_process, args=(argv, call_dataset, device_idx, qdecoder))
        # time.sleep(3)

    finish_encoder = 0
    decode_num = 0
    write_pool = mp.Pool(argv.threads)
    write_mutex = mp.Manager().Value('i', 1)
    while True:
        item = qdecoder.get(timeout=200)
        try:
            qdecoder_size = qdecoder.qsize()
            print('current qdecoder size: ', qdecoder_size,flush=True)
        except NotImplementedError:
            pass
        if item == 0:
            finish_encoder += 1
        if finish_encoder == nDevice:
            write_pool.close()
            write_pool.join()
            break
        if item is not None and item != 0:
            read_id_list, row_num_list, log_probs_list, output_lengths_list = item
            decode_num += len(read_id_list)
            # print('decode_num=',decode_num, flush=True)
            write_pool.apply_async(func=writer, args=(argv, item, write_mutex))


    encode_pool.close()
    encode_pool.join()






if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()






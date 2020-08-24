#!/bin/bash
basecall_compare_data_root=path/to/raw/training/fast5/files
merge_kp_path=path/to/merge/fast5/files/for/validate
signal_out_path=path/to/signal_out_path/valid
label_out_path=path/to/label_out_path/valid

mkdir ${merge_kp_path}
mkdir ${signal_out_path}
mkdir ${label_out_path}

for p in `ls ${basecall_compare_data_root}`
do
    
    datapath="${basecall_compare_data_root}/${p}"
    refpath="${datapath}/read_references.fasta"
    fast5path="${datapath}/validation_fast5s"
    basecalledfast5path="${datapath}/validation_fast5s_guppy"



    # guppy ---gpu
    printf "\n"
    printf "guppy caller running..."
    guppy_basecaller --kit SQK-LSK108 --flowcell FLO-MIN106 --fast5_out --input_path ${fast5path} --save_path ${basecalledfast5path} --num_callers 40  --device "cuda:0" #change to other cuda


    # tombo
    printf "\n"
    printf "tombo running ..."
    tombo resquiggle ${basecalledfast5path}/workspace/ ${refpath} --overwrite --processes 40

    # copy
    printf "\n"
    echo "copy fast5s to ${merge_kp_path}"
    cp ${basecalledfast5path}/workspace/* ${merge_kp_path}
done

printf "\n"
echo "generate signal and label into ${signal_out_path} and ${label_out_path}..."
python path/to/generate_dataset/train_data.py -i ${merge_kp_path} -so ${signal_out_path} -lo ${label_out_path} -raw_len 2048 -seq_len 300 -t 40
echo "done"



#!/bin/bash
model_path=$1   #path/to/model.2048.chkpt
fast5_dir_root=$2  #path/to/test/fast5s
signal_window_length=$3  # 2048
basecalled_dir=$4  #path/to/basecalled/fasta/files
tmp_records_root="tmp_data_dir"
for p in `ls ${fast5_dir_root}`
do
    mkdir -p ${basecalled_dir}/${p}
    fast5_path="${fast5_dir_root}/${p}"
    time_out_path="${basecalled_dir}/${p}/time.txt"

    starttime=`date +'%Y-%m-%d %H:%M:%S'`
    # data preprocessing
    python path/to/generate_dataset/inference_data.py -fast5 ${fast5_path} -records_dir ${tmp_records_dir} -raw_len ${signal_window_length}
    printf "${p} preprocessing done"

    # caller
    python path/to/caller.py -model ${model_path} -records_dir ${tmp_records_dir} -output ${basecalled_dir}/${p} -half
    endtime=`date +'%Y-%m-%d %H:%M:%S'`
    echo "${p} finished!"
    start_seconds=$(date --date="$starttime" +%s);
    end_seconds=$(date --date="$endtime" +%s);
    echo "${p} running time： "$((end_seconds-start_seconds))"s" > ${time_out_path}
    delete tmp records
    rm -rf ${tmp_records_dir}
done

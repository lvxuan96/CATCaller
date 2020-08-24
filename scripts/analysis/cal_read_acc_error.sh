#!/bin/bash

basecall_dir=path/to/basecalled/fasta_dir  #need change
ref_dir=path/to/genome/reference
threads=20

#acc, result in basecall_name_reads.tsv
for p in `ls ${basecall_dir}`
do
    result_dir="${basecall_dir}"/"${p}"/"${p}"_result
    mkdir -p ${result_dir}
    read_alignment="${result_dir}"/"${p}"_reads.paf
    read_data="${result_dir}"/"${p}"_reads.tsv
    reference="${ref_dir}/${p}_reference.fasta.gz"
    basecall_name="${basecall_dir}"/"${p}"/"out.fasta"  #need change
    echo "reads alignment: minimap2..."
    printf "\n"
    minimap2 -x map-ont -t ${threads} -c ${reference} ${basecall_name} > ${read_alignment}
    
    echo "calculate read identity..."
    printf "\n"
    python read_length_identity.py ${basecall_name} ${read_alignment} > ${read_data}
    echo "${p} finished!"
done

#error, result in CATCaller_error.txt
for p in `ls ${basecall_dir}`
do
    result_dir="${basecall_dir}"/"${p}"/"${p}"_result
    reference="${ref_dir}/${p}_reference.fasta.gz"
    basecall_name="${basecall_dir}"/"${p}"/"out.fasta"  #need change
    echo "reads alignment: minimap2..."
    printf "\n"
    minimap2 -ax map-ont ${reference} ${basecall_name} --eqx -t 40 > ${result_dir}/CATCaller_aln.sam
    python minimap2_alignment_report.py ${result_dir}/CATCaller_aln.sam > ${result_dir}/CATCaller.sketch
    python read_sketch.py ${result_dir}/CATCaller.sketch ${result_dir}/CATCaller_error.txt #modify

done


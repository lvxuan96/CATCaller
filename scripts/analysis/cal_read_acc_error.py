import statistics
import sys
import os

def cal_read_identity(filename):
    identities = []
    with open(filename, 'rt') as data_file:
        for line in data_file:
            parts = line.strip().split()
            if parts[0] == 'Name':
                continue
            identities.append(float(parts[2]))
    try:
        total_count = int(sys.argv[2])
        while len(identities) < total_count:
            identities.append(0.0)
    except IndexError:
        pass

    return statistics.median(identities)
def show_error_rate(filename):
    f=open(filename,'r')
    lines=f.read().split('\n')
    line=lines[0]
    parts=line.strip().split()
    error=parts[4]
    f.close()
    return error


file_dir="path/to/basecalled/fasta_dir"  #need change
file_list=os.listdir(file_dir)
for file in file_list:
    read_acc_path=file_dir+'/'+file+'/'+file+'_result/'+file+'_reads.tsv'
    read_acc=cal_read_identity(read_acc_path)
    error_txt=file_dir+'/'+file+'/'+file+'_result/sacall_error.txt'
    error=show_error_rate(error_txt)
    print(file, read_acc, error)
    


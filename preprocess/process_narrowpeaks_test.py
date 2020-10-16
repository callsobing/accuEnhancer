import subprocess
import argparse
import random


parser = argparse.ArgumentParser(description="""Preprocess narrowpeaks""")

#parser.add_argument('--celltype', help='celltype name', required=True)
parser.add_argument('--h3k27ac_file', help='h3k27ac bed file path', required=True)
parser.add_argument('--epi_file', help='epigenome peak file path', required=True)
#parser.add_argument('--output_prefix', help='output prefix name', required=True)
parser.add_argument('--hg38_fasta_bins', help='hg38 fasta file in 200bps bins', required=True)
parser.add_argument('--hg38_bed_bins', help='hg38 bed file in 200bps bins', required=True)
parser.add_argument('--output_name', help='outout name', required=True)
args = parser.parse_args()
output_name = args.output_name
hg38_fasta_bins = args.hg38_fasta_bins
hg38_bed_bins = args.hg38_bed_bins

# 讀入hg38 200bp bin的座標跟序列
hg38_200bin_fh = open(hg38_fasta_bins)
seq_id = ""
seq_fasta = {}
seq_ids = []
chrom_dict = {}
start_dict = {}
end_dict = {}
for line in hg38_200bin_fh:
    line = line.rstrip()
    if line.startswith(">"):
        seq_id = line.lstrip(">")
        cols = seq_id.split(":")
        chrom = cols[0]
        cols2 = cols[1].split("-")
        start = int(cols2[0])
        end = int(cols2[1])
        # In case same position, different snp
        if seq_id in chrom_dict.keys():
            seq_id = seq_id+'_'
        chrom_dict[seq_id] = chrom
        start_dict[seq_id] = start
        end_dict[seq_id] = end
    else:
        seq_fasta[seq_id] = line
        seq_ids.append(seq_id)
hg38_200bin_fh.close()

#================================
#put the bedtools intersect here
def extend_region(in_bed):
    '''
    extend Dnase region to 25 bins
    '''
    wh = open(in_bed+'_dnase', 'w') 
    with open(in_bed,'r') as rh:
        for item in rh:
            wh.write(item)
            info = item.split('\t')
            ptr1 = int(info[1])
            ptr2 = int(info[2])
            space = ptr2-ptr1
            for i in range(12):
                start = str(ptr1-space*(i+1))
                end = str(ptr1-space*i)
                wh.write(f'{info[0]}\t{start}\t{end}\n')
                start = str(ptr2+space*i)
                end = str(ptr2+space*(i+1))
                wh.write(f'{info[0]}\t{start}\t{end}\n')
    wh.close()

def get_peaks(in_bed, epi_type, cell_type):
    '''
    generate peaks .bed files
    '''
    if epi_type=='H3K27ac':
        peak_file = f'data/peakfiles/{cell_type}-H3K27ac.narrowPeak.gz'
        overlap_thrd = 0.5
    else:
        peak_file = f'data/peakfiles/{cell_type}-DNase.macs2.narrowPeak.gz'
        overlap_thrd = 0.3
    result = []
    if epi_type=='DNase':
        extend_region(in_bed)
        in_bed = in_bed+'_dnase'
    with open(in_bed,'r') as h:
        for item in h:
            with open('data/test_row.bed','w') as h:
                h.write(item)
            cmd = f'bedtools intersect -a data/test_row.bed -b {peak_file} -f {overlap_thrd} -wb -wa'
            process = subprocess.Popen(args=cmd,stdout=subprocess.PIPE,shell=True)
            res = process.communicate()[0].decode("utf-8")
            if len(res)==0:
                res = item.rstrip()+'\t0.0\n'
            else:
                info = res.split('\t')
                try:
                    res = f'{info[0]}\t{info[1]}\t{info[2]}\t{info[9]}\n'
                except:
                    print('Error: ',item)
            result.append(res)
    out_bed = f'{peak_file}_test200bp.bins.bed'
    with open(out_bed,'w') as h:
        for item in result:
            h.write(item)
    return result

#=======================

# 讀入要處理的h3k27ac資料 -> 和hg38 200bp bin去做overlap
h3k27ac_file = args.h3k27ac_file
print("### Now fetching signal values from h3k27ac bed files...")

#
cell_type = h3k27ac_file.split("/")[-1].split('-')[0]
                               
for epi in ['H3K27ac', 'DNase']:
    get_peaks(hg38_bed_bins, epi, cell_type)

'''
cmd = "bedtools intersect -a %s -b %s -f 0.5 -wb -wa | awk -F\"\t\" '{print $1\"\t\"$2\"\t\"$3\"\t\"$10}' >%s_test200bp.bins.bed" % (hg38_bed_bins, h3k27ac_file, h3k27ac_file)
process = subprocess.Popen(args=cmd, shell=True)
process.wait()
'''
h3k27ac_positive_bins = {}
h3k27ac_label = []
h3k27ac_file_fh = open("%s_test200bp.bins.bed" % h3k27ac_file)
for line in h3k27ac_file_fh:
    line = line.rstrip()
    cols = line.split("\t")
    key = "%s:%s-%s" % (cols[0], cols[1], cols[2])
    # In case same position, different snp
    if key in h3k27ac_positive_bins.keys():
        key = key+'_'
    h3k27ac_positive_bins[key] = seq_fasta[key]
    try:
        label = 1 if float(cols[3])>0 else 0
    except:
        print(line)
    h3k27ac_label.append(label)
    # del seq_fasta[key]
    # seq_ids.remove(key)
h3k27ac_file_fh.close()

# 讀入要處理的dnase peak資料 -> 和hg38 200bp bin去做overlap
epi_file = args.epi_file
epigenetic_list = ["dnase"]
epi_mark_collection = {}
for epi_mark in epigenetic_list:
    if epi_mark not in epi_mark_collection:
        epi_mark_collection[epi_mark] = {}
    #epi_file = "data/%s/%s.bed" % (epi_mark, args.celltype)
    print("### Now fetching signal values from %s bed files..." % epi_mark)
    '''
    cmd = "bedtools intersect -a %s -b %s -f 0.3 -wb -wa | awk -F\"\t\" '{print $1\"\t\"$2\"\t\"$3\"\t\"$10}' >%s_test200bp.bins.bed" % (hg38_bed_bins,epi_file, epi_file)
    process = subprocess.Popen(args=cmd, shell=True)
    process.wait()
    '''
    epi_file_fh = open("%s_test200bp.bins.bed" % epi_file)
    for line in epi_file_fh:
        line = line.rstrip()
        cols = line.split("\t")
        key = "%s:%s-%s" % (cols[0], cols[1], cols[2])
        # In case same position, different snp
        if key in epi_mark_collection[epi_mark].keys():
            key = key+'_'
        epi_mark_collection[epi_mark][key] = cols[3]
    epi_file_fh.close()


selected_epigenetic_list = ["dnase"]

cmd = "mkdir -p data/single_cell_type"
subprocess.Popen(args=cmd, shell=True).wait()

for selected_epi in selected_epigenetic_list:
    output_test_dest_fh = open("data/single_cell_type/%s_dnase.test.dat" % args.output_name, "w")
    # Process and output the final result:
    print("### Now processing positive bins...")

    pos_idx = 0
    #for key in h3k27ac_positive_bins:
    for li,key in enumerate(h3k27ac_positive_bins):
        pos_idx += 1
        if pos_idx % 50000 == 0:
            print("Processed %d positive sequences for selected %s..." % (pos_idx, selected_epi))
        chrom = chrom_dict[key]
        start = start_dict[key]
        end = end_dict[key]
        feature_signal = ""

        #先處理dnase, 2020.03.09 -> broad dnase peaks 先暫時不做
        #向左取12個bin, 向右取12個bin
        if selected_epi == "dnase":
            start_lhs = start - 12 * 200
            for i in range(25):
                st = start_lhs + i * 200
                key_lhs = "%s:%d-%d" % (chrom, st, st + 200)
                if key_lhs in epi_mark_collection["dnase"]:
                    feature_signal += "\t%s" % epi_mark_collection["dnase"][key_lhs]
                else:
                    feature_signal += "\t0"

        output_test_dest_fh.write("%s\t%d\t%d\t%d\t%s" % (chrom, start, end, h3k27ac_label[li], h3k27ac_positive_bins[key]))
        output_test_dest_fh.write("%s\n" % feature_signal)


    output_test_dest_fh.close()
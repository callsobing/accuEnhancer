import subprocess
import argparse
import random


parser = argparse.ArgumentParser(description="""Preprocess""")

parser.add_argument('--celltype', help='celltype name', required=True)
parser.add_argument('--h3k27ac_file', help='h3k27ac bed file path', required=True)
parser.add_argument('--output_prefix', help='output prefix name', required=True)
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
        chrom_dict[seq_id] = chrom
        start_dict[seq_id] = start
        end_dict[seq_id] = end
    else:
        seq_fasta[seq_id] = line
        seq_ids.append(seq_id)
hg38_200bin_fh.close()

# 讀入要處理的h3k27ac資料 -> 和hg38 200bp bin去做overlap
h3k27ac_file = args.h3k27ac_file
print("### Now fetching signal values from h3k27ac bed files...")
cmd = "bedtools intersect -a %s -b %s -f 0.5 -wb -wa | awk -F\"\t\" '{print $1\"\t\"$2\"\t\"$3\"\t\"$10}' >%s.200bp.bins.bed" % (hg38_bed_bins, h3k27ac_file, h3k27ac_file)
process = subprocess.Popen(args=cmd, shell=True)
process.wait()
h3k27ac_positive_bins = {}
h3k27ac_file_fh = open("%s.200bp.bins.bed" % h3k27ac_file)
for line in h3k27ac_file_fh:
    line = line.rstrip()
    cols = line.split("\t")
    key = "%s:%s-%s" % (cols[0], cols[1], cols[2])
    h3k27ac_positive_bins[key] = seq_fasta[key]
    # del seq_fasta[key]
    # seq_ids.remove(key)
h3k27ac_file_fh.close()

# 讀入要處理的dnase peak資料 -> 和hg38 200bp bin去做overlap
epigenetic_list = ["dnase"]
epi_mark_collection = {}
for epi_mark in epigenetic_list:
    if epi_mark not in epi_mark_collection:
        epi_mark_collection[epi_mark] = {}
    epi_file = "data/%s/%s.bed" % (epi_mark, args.celltype)
    print("### Now fetching signal values from %s bed files..." % epi_mark)
    cmd = "bedtools intersect -a %s -b %s -f 0.3 -wb -wa | awk -F\"\t\" '{print $1\"\t\"$2\"\t\"$3\"\t\"$10}' >%s.200bp.bins.bed" % (hg38_bed_bins,epi_file, epi_file)
    process = subprocess.Popen(args=cmd, shell=True)
    process.wait()

    epi_file_fh = open("%s.200bp.bins.bed" % epi_file)
    for line in epi_file_fh:
        line = line.rstrip()
        cols = line.split("\t")
        key = "%s:%s-%s" % (cols[0], cols[1], cols[2])
        epi_mark_collection[epi_mark][key] = cols[3]
    epi_file_fh.close()



# 準備產生negative序列:
negative_bins = {}
positive_count = len(h3k27ac_positive_bins)
print("### Now selecting negative seq from hg38 200bp bins pool...")
for idx in range(positive_count * 10):
# for idx in range(positive_count * 5):
    random_id = random.choice(seq_ids)
    while (random_id in h3k27ac_positive_bins) or (random_id in negative_bins):
        # print("%s is in h3k27ac positive data, skipping.." % random_id)
        random_id = random.choice(seq_ids)
    negative_bins[random_id] = seq_fasta[random_id]
    # del seq_fasta[random_id]
    # seq_ids.remove(random_id)
    if idx % positive_count == 0:
        print("Processed %d negative sequences..." % idx)

pos_validation_set = random.sample(range(1, positive_count), int(positive_count/10))
neg_validation_set = random.sample(range(1, positive_count * 5), int((positive_count * 5) / 10))

selected_epigenetic_list = ["dnase"]

cmd = "mkdir -p data/single_cell_type"
subprocess.Popen(args=cmd, shell=True).wait()

for selected_epi in selected_epigenetic_list:
    output_train_dest_fh = open("data/single_cell_type/%s_dnase.training.dat" % args.output_name, "w")
    output_val_dest_fh = open("data/single_cell_type/%s_dnase.validation.dat" % args.output_name, "w")
    # Process and output the final result:
    print("### Now processing positive bins...")

    pos_idx = 0
    for key in h3k27ac_positive_bins:
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

        if pos_idx in pos_validation_set:
            output_val_dest_fh.write("%s\t%d\t%d\t1\t%s" % (chrom, start, end, h3k27ac_positive_bins[key]))
            output_val_dest_fh.write("%s\n" % feature_signal)
        else:
            output_train_dest_fh.write("%s\t%d\t%d\t1\t%s" % (chrom, start, end, h3k27ac_positive_bins[key]))
            output_train_dest_fh.write("%s\n" % feature_signal)

    neg_idx = 0
    print("### Now processing negative bins...")
    for key in negative_bins:
        neg_idx += 1
        if neg_idx % 50000 == 0:
            print("Processed %d positive sequences for selected %s..." % (neg_idx, selected_epi))
        chrom = chrom_dict[key]
        start = start_dict[key]
        end = end_dict[key]
        feature_signal = ""

        # 先處理dnase, 2020.03.09 -> broad dnase peaks 先暫時不做
        # 向左取12個bin, 向右取12個bin
        if selected_epi == "dnase":
            start_lhs = start - 12 * 200
            for i in range(25):
                st = start_lhs + i * 200
                key_lhs = "%s:%d-%d" % (chrom, st, st + 200)
                if key_lhs in epi_mark_collection["dnase"]:
                    feature_signal += "\t%s" % epi_mark_collection["dnase"][key_lhs]
                else:
                    feature_signal += "\t0"

        if neg_idx in neg_validation_set:
            output_val_dest_fh.write("%s\t%d\t%d\t0\t%s" % (chrom, start, end, negative_bins[key]))
            output_val_dest_fh.write("%s\n" % feature_signal)
        else:
            output_train_dest_fh.write("%s\t%d\t%d\t0\t%s" % (chrom, start, end, negative_bins[key]))
            output_train_dest_fh.write("%s\n" % feature_signal)
    output_train_dest_fh.close()
    output_val_dest_fh.close()

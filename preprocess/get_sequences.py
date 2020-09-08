import subprocess
import argparse

parser = argparse.ArgumentParser(description="""Generate 200-bp sequences from genome""")
parser.add_argument('--genome',help='reference genome path', required=True)
parser.add_argument('--out',help='output path of 200bp sequences from genome', required=True)

args = parser.parse_args()
genome_path = args.genome
output_path = args.out

# Build index
cmd = f'samtools faidx {genome_path}'
process = subprocess.Popen(args=cmd, shell=True)
process.wait()

# Generate .bed of 200bp
with open(f'{genome_path}.fai','r') as h:
    genome_idx = h.readlines()

chr_order = [item.split('\t')[0][3:] for item in genome_idx]
chr_keys = [str(i) for i in range(1,23)] + ['X','Y']
chr_order_idx = [i for key in chr_keys for i,item in enumerate(chr_order) if key==item]
genome_idx_sorted = [genome_idx[i] for i in chr_order_idx]

h = open(f'{output_path}_200bin.bed','w')
for item in genome_idx_sorted:
  bed_info = item.split('\t')
  chromosome = bed_info[0]
  start = 0
  end = int(bed_info[1])
  for i in range(start,end,200):
    h.write(chromosome+'\t'+str(i)+'\t'+str(i+200)+'\n')
h.close()

# Generate 200bp sequences
cmd = f'bedtools getfasta -fi {genome_path} -bed {output_path}_200bin.bed -fo {output_path}_200bin.fa'
process = subprocess.Popen(args=cmd, shell=True)
process.wait()

print(f'200-bp bins fasta file generated in {output_path}_200bin.fa')
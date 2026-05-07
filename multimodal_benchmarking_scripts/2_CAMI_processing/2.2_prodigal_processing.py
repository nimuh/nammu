#!/usr/bin/env python3
"""
Convert prodigal output (proteins .faa + nucleotides .fna) into a per-gene CSV.

Columns produced:
    CSMAG_ID, Sequence_ID, contig_ID, Start, End, Strand, ID, Partial,
    Start_Type, RBS_Motif, RBS_Spacer, GC_Content, Sequence,
    Sequence_nucleotide, Length

Usage:
    python prodigal_to_csv.py \\
        --proteins proteins.faa \\
        --nucleotides genes.fna \\
        --sample-id S0 \\
        --output prodigal_output.csv
"""

import argparse
import csv
from Bio import SeqIO


def parse_prodigal_header(description):
    """
    Prodigal FASTA headers look like:
      >contigID_geneNum # start # end # strand # ID=...;partial=...;start_type=...;rbs_motif=...;rbs_spacer=...;gc_cont=...
    """
    parts = description.split(' # ')
    seq_id = parts[0]
    start = int(parts[1])
    end = int(parts[2])
    strand = int(parts[3])  # 1 or -1
    attrs_raw = parts[4] if len(parts) > 4 else ''

    attrs = {}
    for kv in attrs_raw.split(';'):
        if '=' in kv:
            k, v = kv.split('=', 1)
            attrs[k.strip()] = v.strip()

    return {
        'seq_id': seq_id,
        'start': start,
        'end': end,
        'strand': strand,
        'id': attrs.get('ID', ''),
        'partial': attrs.get('partial', ''),
        'start_type': attrs.get('start_type', ''),
        'rbs_motif': attrs.get('rbs_motif', ''),
        'rbs_spacer': attrs.get('rbs_spacer', ''),
        'gc_cont': attrs.get('gc_cont', ''),
    }


def extract_contig_id(seq_id):
    """
    Prodigal names genes as <contig_id>_<gene_number>.
    Strip the trailing _N to recover the contig ID.
    """
    return seq_id.rsplit('_', 1)[0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--proteins', required=True, help='Prodigal -a output (.faa)')
    p.add_argument('--nucleotides', required=True, help='Prodigal -d output (.fna)')
    p.add_argument('--sample-id', default='', help='Sample/MAG ID for CSMAG_ID column')
    p.add_argument('--output', required=True, help='Output CSV path')
    args = p.parse_args()

    # Load nucleotide sequences keyed by gene seq_id
    nt_lookup = {rec.id: str(rec.seq).upper()
                 for rec in SeqIO.parse(args.nucleotides, 'fasta')}

    columns = [
        'CSMAG_ID', 'Sequence_ID', 'contig_ID', 'Start', 'End', 'Strand',
        'ID', 'Partial', 'Start_Type', 'RBS_Motif', 'RBS_Spacer', 'GC_Content',
        'Sequence', 'Sequence_nucleotide', 'Length',
    ]

    n_written = 0
    with open(args.output, 'w', newline='') as fout:
        writer = csv.DictWriter(fout, fieldnames=columns)
        writer.writeheader()

        for rec in SeqIO.parse(args.proteins, 'fasta'):
            meta = parse_prodigal_header(rec.description)
            protein_seq = str(rec.seq)
            nt_seq = nt_lookup.get(rec.id, '')
            contig_id = extract_contig_id(rec.id)

            row = {
                'CSMAG_ID': args.sample_id,
                'Sequence_ID': rec.id,
                'contig_ID': contig_id,
                'Start': meta['start'],
                'End': meta['end'],
                'Strand': meta['strand'],
                'ID': meta['id'],
                'Partial': meta['partial'],
                'Start_Type': meta['start_type'],
                'RBS_Motif': meta['rbs_motif'],
                'RBS_Spacer': meta['rbs_spacer'],
                'GC_Content': meta['gc_cont'],
                'Sequence': protein_seq,
                'Sequence_nucleotide': nt_seq,
                'Length': len(protein_seq.rstrip('*')),
            }
            writer.writerow(row)
            n_written += 1

    print(f'Wrote {n_written} gene rows to {args.output}')


if __name__ == '__main__':
    main()
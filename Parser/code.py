import json
import pandas as pd


def main():
    names_json = ['meth_normal.json', 'meth_primary_tumor.json', 'miRNA_normal.json', 'miRNA_primary_tumor.json',
                  'mRNA_normal.json', 'mRNA_primary_tumor.json']
    names_manifest = ['meth_normal_manifest.txt', 'meth_primary_tumor_manifest.txt', 'miRNA_normal_manifest.txt',
                      'miRNA_primary_tumor_manifest.txt', 'mRNA_normal_manifest.txt', 'mRNA_primary_tumor_manifest.txt']
    for json_file, txt_file in zip(names_json, names_manifest):
        manifest_name = './final_manifests/' + json_file.split('.')[0] + '_final_manifest.txt'
        f = open(json_file, )
        data = json.load(f)
        manifest = pd.read_csv(txt_file, sep="\t", header=0)
        for element in data:
            manifest['filename'] = manifest['filename'].replace([element['file_name']],
                                                                        element['cases'][0]['case_id'])
        manifest.to_csv(manifest_name, sep='\t', index=False)


if __name__ == '__main__':
    main()
'''
get image and label for image datasets
'''

import os
import csv
import click
import glob

# TODO: 自动生成，参考d2l代码，保存类别文件到数据集文件下
constellationGraph_str2int_dict = {'4--PSK':0, '8--QAM':1, '16-QAM':2, '32-QAM':3, '64-QAM':4}

@click.command()
@click.option('--data_dir', default='data/constellationGraph', help='Directory for storing input data')
@click.option('--output_file', default='data/constellationGraph.csv', help='File for storing input data')
def get_image_label(data_dir, output_file):
    label_category = os.listdir(data_dir)
    click.echo(label_category)
    with open(output_file, 'w', newline='') as f:
        for label in label_category:
            label_path = os.path.join(data_dir, label)
            # 匹配所有的jpg文件的地址
            image_path = glob.glob(os.path.join(label_path, '*.jpg'))
            writer = csv.writer(f)
            for image in image_path:
                writer.writerow([image, constellationGraph_str2int_dict[label]])

@click.command()
@click.option('--data_dir', default='data/constellationGraph', help='Directory for storing input data')
@click.option('--output_file', default='data/constellationGraph_noisy.csv', help='File for storing output data')
def get_image_label_noisy(data_dir, output_file):
    import re

    label_category = os.listdir(data_dir)
    click.echo(label_category)
    noisy_pattern = r'(?<=_)\d+'
    with open(output_file, 'w', newline='') as f:
        for label in label_category:
            label_path = os.path.join(data_dir, label)
            image_path = glob.glob(os.path.join(label_path, '*.jpg'))
            noisy_strings = [re.search(noisy_pattern, image).group() for image in image_path]
            writer = csv.writer(f)
            for image, noisy in zip(image_path, noisy_strings):
                writer.writerow([image, constellationGraph_str2int_dict[label], noisy])


if __name__ == '__main__':
    get_image_label()
    # get_image_label_noisy()




import csv
import os
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm


def save_csv(data, path,
             fieldnames=['image_path', 'gender', 'articleType', 'baseColour']):
    with open(path, 'w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in data:
            writer.writerow(dict(zip(fieldnames, row)))


def main(args):
    input_folder = args.input
    output_folder = args.output
    annotation = os.path.join(input_folder, 'styles.csv')
    all_data = []

    with open(annotation) as csv_file:
        fashion = csv.DictReader(csv_file)
        for row in tqdm(fashion, total=fashion.line_num):
            image_id = row['id']
            gender = row['gender']
            articleType = row['articleType']
            baseColour = row['baseColour']
            image_name = os.path.join(
                input_folder, 'images', str(image_id) + '.jpg')
            if os.path.exists(image_name):
                image = Image.open(image_name)
                if image.size == (60, 80) and image.mode == 'RGB':
                    all_data.append(
                        [image_name, gender, articleType, baseColour])
            else:
                print("Something went wrong: there is no file ", image_name)

    np.random.seed(42)
    all_data = np.asarray(all_data)

    inds = np.random.choice(40000, 40000, replace=False)

    save_csv(all_data[inds][:32000], os.path.join(output_folder, 'train.csv'))
    save_csv(all_data[inds][32000:40000],
             os.path.join(output_folder, 'val.csv'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split data for the dataset')
    parser.add_argument('--input', type=str, required=True,
                        help="Path to the dataset")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to the working folder")
    args = parser.parse_args()
    main(args)

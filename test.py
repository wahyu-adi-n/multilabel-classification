import argparse
from utils.utils import *
from config.config import *
from dataset.dataloader import get_dataloader
from model.FashionMultilableModel import FashionMultilableModel as FSM

def main(args):
    attr = args.attributes_file
    _, _, test_dl = get_dataloader(attr)
    model = FSM(n_color_classes=attr.num_colors,
                n_gender_classes=attr.num_genders,
                n_article_classes=attr.num_articles
                ).to(DEVICE)
    visualize_grid(model, test_dl, attr, DEVICE, show_cn_matrices=False,
                   show_images=True, checkpoint=None, show_gt=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference pipeline')
    parser.add_argument('--checkpoint', type=str,
                        required=True, help="Path to the checkpoint")
    parser.add_argument('--attributes_file', type=str, default='./styles.csv',
                        help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)

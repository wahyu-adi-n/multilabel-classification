from utils.utils import *
from utils import engine
from config.config import *
from dataset.dataloader import get_dataloader
from dataset.dataset import AttributesDataset
from model.FashionMultilableModel import FashionMultilableModel as FSM
from torch import optim
from torch.utils.tensorboard import SummaryWriter, summary
import argparse


def main(args):
    attr = AttributesDataset(args.attributes_file)
    train_dl, val_dl, _ = get_dataloader(attr)
    model = FSM(n_color_classes=attr.num_colors,
                n_gender_classes=attr.num_genders,
                n_article_classes=attr.num_articles
                ).to(DEVICE)
    opt = optim.Adam(model.parameters())

    # LOGGER
    logdir = os.path.join('./logs/', get_current_time())
    savedir = os.path.join('./checkpoints/', get_current_time())
    os.makedirs(logdir, exist_ok=True)
    os.makedirs(savedir, exist_ok=True)
    logger = SummaryWriter(logdir)

    # VISUALIZE
    # visualize_grid(model, val_dl, attr, DEVICE, show_cn_matrices=False, show_images=True, checkpoint=None, show_gt=True)
    # print("\nAll gender labels:\n", attr.gender_labels)
    # print("\nAll color labels:\n", attr.color_labels)
    # print("\nAll article labels:\n", attr.article_labels)

    engine.train(model=model,
                 train_dataloader=train_dl, val_dataloader=val_dl, optimizer=opt,
                 logger=logger,
                 savedir=savedir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Training pipeline')
    parser.add_argument('--attributes_file', type=str,
                        default='./styles.csv', help="Path to the file with attributes")
    parser.add_argument('--device', type=str, default='cpu',
                        help="Device: 'cuda' or 'cpu'")
    args = parser.parse_args()
    main(args)

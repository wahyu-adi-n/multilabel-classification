from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
from config.config import *
from warnings import filterwarnings
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import warnings
filterwarnings('ignore')


def get_current_time():
    return datetime.strftime(datetime.now(), '%Y-%m-%d_%H-%M')


def checkpoint_save(model: nn.Module, name: str, epoch: int):
    f = os.path.join(name, 'checkpoint-{:06d}.pth'.format(epoch))
    torch.save(model.state_dict(), f)
    print(f'Saved Checkpoint:{f}')


def checkpoint_load(model, name):
    print('Restoring checkpoint: {}'.format(name))
    model.load_state_dict(torch.load(name, map_location='cpu'))
    epoch = int(os.path.splitext(os.path.basename(name))[0].split('-')[1])
    return epoch


def calculate_metrics(output, target):
    _, predicted_color = output['color'].cpu().max(1)
    gt_color = target['color_labels'].cpu()

    _, predicted_gender = output['gender'].cpu().max(1)
    gt_gender = target['gender_labels'].cpu()

    _, predicted_article = output['article'].cpu().max(1)
    gt_article = target['article_labels'].cpu()

    with warnings.catch_warnings():  # sklearn may produce a warning when processing zero row in confusion matrix
        warnings.simplefilter("ignore")
        acc_color = balanced_accuracy_score(
            y_true=gt_color.numpy(), y_pred=predicted_color.numpy())
        acc_gender = balanced_accuracy_score(
            y_true=gt_gender.numpy(), y_pred=predicted_gender.numpy())
        acc_article = balanced_accuracy_score(
            y_true=gt_article.numpy(), y_pred=predicted_article.numpy())

    return acc_color, acc_gender, acc_article


def visualize_grid(model, dataloader, attributes, device, show_cn_matrices=True, show_images=True, checkpoint=None,
                   show_gt=False):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)
    model.eval()

    imgs = []
    labels = []
    gt_labels = []
    gt_color_all = []
    gt_gender_all = []
    gt_article_all = []
    predicted_color_all = []
    predicted_gender_all = []
    predicted_article_all = []

    accuracy_color = 0
    accuracy_gender = 0
    accuracy_article = 0

    with torch.no_grad():
        for batch in dataloader:
            img = batch['img']
            gt_colors = batch['labels']['color_labels']
            gt_genders = batch['labels']['gender_labels']
            gt_articles = batch['labels']['article_labels']
            output = model(img.to(device))

            batch_accuracy_color, batch_accuracy_gender, batch_accuracy_article = \
                calculate_metrics(output, batch['labels'])
            accuracy_color += batch_accuracy_color
            accuracy_gender += batch_accuracy_gender
            accuracy_article += batch_accuracy_article

            # get the most confident prediction for each image
            _, predicted_colors = output['color'].cpu().max(1)
            _, predicted_genders = output['gender'].cpu().max(1)
            _, predicted_articles = output['article'].cpu().max(1)

            for i in range(img.shape[0]):
                image = np.clip(img[i].permute(
                    1, 2, 0).numpy() * std + mean, 0, 1)

                predicted_color = attributes.color_id_to_name[predicted_colors[i].item(
                )]
                predicted_gender = attributes.gender_id_to_name[predicted_genders[i].item(
                )]
                predicted_article = attributes.article_id_to_name[predicted_articles[i].item(
                )]

                gt_color = attributes.color_id_to_name[gt_colors[i].item()]
                gt_gender = attributes.gender_id_to_name[gt_genders[i].item()]
                gt_article = attributes.article_id_to_name[gt_articles[i].item(
                )]

                gt_color_all.append(gt_color)
                gt_gender_all.append(gt_gender)
                gt_article_all.append(gt_article)

                predicted_color_all.append(predicted_color)
                predicted_gender_all.append(predicted_gender)
                predicted_article_all.append(predicted_article)

                imgs.append(image)
                labels.append("{}\n{}\n{}".format(
                    predicted_gender, predicted_article, predicted_color))
                gt_labels.append("{}\n{}\n{}".format(
                    gt_gender, gt_article, gt_color))

    if not show_gt:
        n_samples = len(dataloader)
        print("\nAccuracy:\ncolor: {:.4f}, gender: {:.4f}, article: {:.4f}".format(
            accuracy_color / n_samples,
            accuracy_gender / n_samples,
            accuracy_article / n_samples))

    # Draw confusion matrices
    if show_cn_matrices:
        # color
        cn_matrix = confusion_matrix(
            y_true=gt_color_all,
            y_pred=predicted_color_all,
            labels=attributes.color_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.color_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.title("Colors")
        plt.tight_layout()
        plt.show()

        # gender
        cn_matrix = confusion_matrix(
            y_true=gt_gender_all,
            y_pred=predicted_gender_all,
            labels=attributes.gender_labels,
            normalize='true')
        ConfusionMatrixDisplay(cn_matrix, attributes.gender_labels).plot(
            xticks_rotation='horizontal')
        plt.title("Genders")
        plt.tight_layout()
        plt.show()

        # Uncomment code below to see the article confusion matrix (it may be too big to display)
        cn_matrix = confusion_matrix(
            y_true=gt_article_all,
            y_pred=predicted_article_all,
            labels=attributes.article_labels,
            normalize='true')
        plt.rcParams.update({'font.size': 1.8})
        plt.rcParams.update({'figure.dpi': 300})
        ConfusionMatrixDisplay(cn_matrix, attributes.article_labels).plot(
            include_values=False, xticks_rotation='vertical')
        plt.rcParams.update({'figure.dpi': 100})
        plt.rcParams.update({'font.size': 5})
        plt.title("Article types")
        plt.show()

    if show_images:
        labels = gt_labels if show_gt else labels
        title = "Ground truth labels" if show_gt else "Predicted labels"
        n_cols = 5
        n_rows = 3
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(10, 10))
        axs = axs.flatten()
        for img, ax, label in zip(imgs, axs, labels):
            ax.set_xlabel(label, rotation=0)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.imshow(img)
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    model.train()

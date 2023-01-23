import torch
from config.config import *
from utils.utils import *


def train(model, train_dataloader, val_dataloader, optimizer, logger, savedir):
    n_train_samples = len(train_dataloader)
    print("Starting training ...")
    for epoch in range(1, NUM_EPOCHS+1):
        total_loss = 0
        acc_color = 0
        acc_gender = 0
        acc_article = 0

        for batch in train_dataloader:
            optimizer.zero_grad()
            image = batch['image']
            labels = batch['labels']
            labels = {t: labels[t].to(DEVICE) for t in labels}
            output = model(image)

            loss_train, losses_train = model.get_loss(output, labels)
            total_loss += loss_train.item()
            batch_acc_color, batch_acc_gender, batch_acc_article = calculate_metrics(
                output, labels)

            acc_color += batch_acc_color
            acc_gender += batch_acc_gender
            acc_article += batch_acc_article

            loss_train.backward()
            optimizer.step()
        print(f'epoch: {epoch:4d} - loss: {total_loss/n_train_samples:.4f} - acc_color: {acc_color/n_train_samples:.4f} - acc_gender: {acc_gender/n_train_samples:.4f} - acc_article: {acc_article/n_train_samples:.4f}')

        logger.add_scalar('loss', total_loss / n_train_samples, epoch)
        logger.add_scalar('acc_color', acc_color/n_train_samples, epoch)
        logger.add_scalar('acc_gender', acc_gender /
                          n_train_samples, epoch)
        logger.add_scalar('acc_article', acc_article /
                          n_train_samples, epoch)

        if epoch % 5 == 0:
            validate(model, val_dataloader, logger, epoch, DEVICE)
        if epoch % 25 == 0:
            checkpoint_save(model, savedir, epoch)


def validate(model, dataloader, logger, iteration, device, checkpoint=None):
    if checkpoint is not None:
        checkpoint_load(model, checkpoint)

    model.eval()
    with torch.no_grad():
        avg_loss = 0
        acc_color = 0
        acc_gender = 0
        acc_article = 0

        for batch in dataloader:
            img = batch['img']
            target_labels = batch['labels']
            target_labels = {t: target_labels[t].to(
                device) for t in target_labels}
            output = model(img.to(device))

            val_train, val_train_losses = model.get_loss(output, target_labels)
            avg_loss += val_train.item()
            batch_accuracy_color, batch_accuracy_gender, batch_accuracy_article = \
                calculate_metrics(output, target_labels)

            accuracy_color += batch_accuracy_color
            accuracy_gender += batch_accuracy_gender
            accuracy_article += batch_accuracy_article

    n_samples = len(dataloader)
    avg_loss /= n_samples
    acc_color /= n_samples
    acc_gender /= n_samples
    acc_article /= n_samples
    print('-' * 72)
    print("val_loss: {:.4f}, val_acc_color: {:.4f}, val_acc_gender: {:.4f}, val_acc_article: {:.4f}\n".format(
        avg_loss, acc_color, acc_gender, acc_article))

    logger.add_scalar('val_loss', avg_loss, iteration)
    logger.add_scalar('val_acc_color', acc_color, iteration)
    logger.add_scalar('val_acc_gender', acc_gender, iteration)
    logger.add_scalar('val_acc_article', acc_article, iteration)

    model.train()

import logging
import os

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import (
    accuracy,
    save_checkpoint,
    save_config_file,
    save_json,
    save_simclr_history_plot,
)

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter(log_dir=self.args.run_dir)
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)
        history = []

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        for epoch_counter in range(self.args.epochs):
            total_loss = 0.0
            total_top1 = 0.0
            total_top5 = 0.0
            num_batches = 0

            for images, _ in tqdm(train_loader, desc=f"SimCLR epoch {epoch_counter + 1}/{self.args.epochs}"):
                images = torch.cat(images, dim=0)

                images = images.to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    embeddings, projections = self.model(images, return_embedding=True)
                    logits, labels = self.info_nce_loss(projections)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                total_loss += loss.item()
                total_top1 += top1[0].item()
                total_top5 += top5[0].item()
                num_batches += 1

                if n_iter % self.args.log_every_n_steps == 0:
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                    self.writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                    self.writer.add_scalar('learning_rate', self.scheduler.get_last_lr()[0], global_step=n_iter)
                    self.writer.add_scalar('feature_dim', float(self.model.feature_dim), global_step=n_iter)
                    self.writer.add_scalar('projection_dim', float(self.model.projection_dim), global_step=n_iter)

                n_iter += 1

            # warmup for the first 10 epochs
            if epoch_counter >= 10:
                self.scheduler.step()

            epoch_metrics = {
                "epoch": epoch_counter + 1,
                "train_loss": total_loss / num_batches,
                "train_top1": total_top1 / num_batches,
                "train_top5": total_top5 / num_batches,
                "learning_rate": self.scheduler.get_last_lr()[0],
            }
            history.append(epoch_metrics)

            self.writer.add_scalar("epoch/loss", epoch_metrics["train_loss"], global_step=epoch_counter + 1)
            self.writer.add_scalar("epoch/top1", epoch_metrics["train_top1"], global_step=epoch_counter + 1)
            self.writer.add_scalar("epoch/top5", epoch_metrics["train_top5"], global_step=epoch_counter + 1)
            self.writer.add_scalar("epoch/learning_rate", epoch_metrics["learning_rate"], global_step=epoch_counter + 1)
            logging.debug(
                f"Epoch: {epoch_counter + 1}\t"
                f"Loss: {epoch_metrics['train_loss']:.4f}\t"
                f"Top1 accuracy: {epoch_metrics['train_top1']:.2f}\t"
                f"Top5 accuracy: {epoch_metrics['train_top5']:.2f}"
            )
            print(
                f"Epoch {epoch_counter + 1}\t"
                f"Train Loss {epoch_metrics['train_loss']:.4f}\t"
                f"Train Top1 {epoch_metrics['train_top1']:.2f}\t"
                f"Train Top5 {epoch_metrics['train_top5']:.2f}"
            )

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'aug_strength': self.args.aug_strength,
            'use_projection_head': not self.args.disable_projection_head,
            'feature_dim': self.model.feature_dim,
            'projection_dim': self.model.projection_dim,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))

        history_payload = {
            "run_name": self.args.run_name,
            "run_dir": os.path.abspath(self.writer.log_dir),
            "epochs": self.args.epochs,
            "arch": self.args.arch,
            "dataset_name": self.args.dataset_name,
            "history": history,
        }
        history_path = os.path.join(self.writer.log_dir, "training_history.json")
        save_json(history_payload, history_path)

        plot_path = os.path.join(self.writer.log_dir, "training_curves.png")
        plot_saved = save_simclr_history_plot(history, plot_path, f"SimCLR Training: {self.args.run_name}")
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
        logging.info(f"Training history has been saved at {history_path}.")
        if plot_saved:
            logging.info(f"Training curves have been saved at {plot_path}.")
        print(f"Saved checkpoint to {os.path.join(self.writer.log_dir, checkpoint_name)}")
        print(f"Saved training history to {history_path}")
        if plot_saved:
            print(f"Saved training curves to {plot_path}")

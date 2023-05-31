from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.seed import seed_everything
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import sys
from segmentation_models_pytorch.losses import *

torch.multiprocessing.set_sharing_strategy('file_system')

def get_class_labels():
    return ['BG', 'RV', 'MYO', 'LV']


class SupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        if cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
        elif cfg.arch == "custom":
            self.net = CustomFeaturizer(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        self.cluster_probe = ClusterLookup(dim, cfg.clustering_classes)
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", cfg.clustering_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", n_classes, 0, False)

        if cfg.linear_loss == 'dice':
            self.linear_probe_loss_fn = DiceLoss('multiclass', from_logits=True)
        elif cfg.linear_loss == 'jaccard':
            self.linear_probe_loss_fn = JaccardLoss('multiclass', from_logits=True)
        elif cfg.linear_loss == 'focal':
            self.linear_probe_loss_fn = FocalLoss('multiclass')
        elif cfg.linear_loss == 'tversky':
            self.linear_probe_loss_fn = TverskyLoss('multiclass', from_logits=True, alpha=0.3, beta=0.7)
        else:
            self.linear_probe_loss_fn = nn.CrossEntropyLoss()

        self.automatic_optimization = False

        self.use_crf_in_val = cfg.use_crf_in_val

        self.label_cmap = create_pascal_label_colormap()

        self.val_steps = 0
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        if self.cfg.freeze_segmenter:
            self.net.eval()
        else:
            self.net.train()

        optims = self.optimizers()
        net_optim, linear_probe_optim, cluster_probe_optim = optims[0], optims[1], optims[2]

        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        img, label, _ = batch

        feats, code = self.net(img)

        log_args = dict(sync_dist=False, rank_zero_only=True)

        loss = 0

        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)
        linear_logits = self.linear_probe(code)
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)

        if len(flat_label[mask]) > 0:
            linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask])
        else:
            linear_loss = 0

        loss += linear_loss
        self.log('loss/linear', linear_loss, **log_args)

        detached_code = torch.clone(code.detach())
        cluster_loss, cluster_probs = self.cluster_probe(detached_code, None)
        loss += cluster_loss
        self.log('loss/total', loss, **log_args)

        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()
        
        if self.cfg.clr:
            scheduler = self.lr_schedulers()
            scheduler.step()

        return loss

    def on_train_start(self):
        self.logger.log_hyperparams(self.cfg)

    def validation_step(self, batch, batch_idx):
        img, label, _ = batch
        self.net.eval()

        with torch.no_grad():
            feats, code = self.net(img)
            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

            linear_preds = self.linear_probe(code)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            cluster_loss, cluster_preds = self.cluster_probe(code, None)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)

            return {
                'img': img[:self.cfg.n_images].detach().cpu(),
                'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()}

    # Visualizing and logging validation results
    def validation_epoch_end(self, outputs) -> None:
        super().validation_epoch_end(outputs)
        with torch.no_grad():
            tb_metrics = {
                **self.linear_metrics.compute(),
                **self.cluster_metrics.compute(),
            }

            if self.trainer.is_global_zero:
                output_num = random.randint(0, len(outputs) -1)
                output = {k: v.detach().cpu() for k, v in outputs[output_num].items()}
                
                # Visualizing predictions
                fig, ax = plt.subplots(4, self.cfg.n_images, figsize=(self.cfg.n_images * 3, 4 * 3))
                num_of_sample = self.cfg.n_images if self.cfg.n_images < len(output["img"]) else len(output["img"])
                for i in range(num_of_sample):
                    ax[0, i].imshow(prep_for_plot(output["img"][i]))
                    ax[1, i].imshow(np.squeeze(self.label_cmap[output["label"][i]]))
                    ax[2, i].imshow(self.label_cmap[output["linear_preds"][i]])
                    ax[3, i].imshow(self.label_cmap[self.cluster_metrics.map_clusters(output["cluster_preds"][i])])
                ax[0, 0].set_ylabel("Image", fontsize=16)
                ax[1, 0].set_ylabel("Label", fontsize=16)
                ax[2, 0].set_ylabel("Linear Probe", fontsize=16)
                ax[3, 0].set_ylabel("Cluster Probe", fontsize=16)
                remove_axes(ax)
                plt.tight_layout()
                add_plot(self.logger, "plot_labels", self.global_step)
                
                # Visualizing predictions as confusion matrix
                if self.cfg.has_labels:
                    fig = plt.figure(figsize=(13, 10))
                    ax = fig.gca()
                    hist = self.linear_metrics.histogram.detach().cpu().to(torch.float32)
                    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
                    sns.set(font_scale=2.3)
                    g = sns.heatmap(hist.t(), annot=True, fmt='.3f', ax=ax, cmap="Blues")
                    ax.set_xlabel('Prediktált címkék', fontsize=22)
                    ax.set_ylabel('Valódi címkék', fontsize=22)
                    names = get_class_labels()
                    ax.set_xticks(np.arange(0, len(names)) + .5)
                    ax.set_yticks(np.arange(0, len(names)) + .5)
                    ax.xaxis.tick_top()
                    ax.xaxis.set_ticklabels(names, fontsize=22)
                    ax.yaxis.set_ticklabels(names, fontsize=22)
                    colors = [self.label_cmap[i] / 255.0 for i in range(len(names))]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
                    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
                    plt.tight_layout()
                    add_plot(self.logger, "linear_conf_matrix", self.global_step)
                    
                    fig = plt.figure(figsize=(13, 10))
                    ax = fig.gca()
                    hist = self.cluster_metrics.histogram.detach().cpu().to(torch.float32)
                    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
                    sns.set(font_scale=2.3)
                    sns.heatmap(hist.t(), annot=True, fmt='.3f',  ax=ax, cmap="Blues")
                    ax.set_xlabel('Prediktált címkék', fontsize=22)
                    ax.set_ylabel('Valódi címkék', fontsize=22)
                    names = get_class_labels()
                    ax.set_xticks(np.arange(0, len(names)) + .5)
                    ax.set_yticks(np.arange(0, len(names)) + .5)
                    ax.xaxis.tick_top()
                    ax.xaxis.set_ticklabels(names, fontsize=22)
                    ax.yaxis.set_ticklabels(names, fontsize=22)
                    colors = [self.label_cmap[i] / 255.0 for i in range(len(names))]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
                    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
                    plt.tight_layout()
                    add_plot(self.logger, "cluster_conf_matrix", self.global_step)

            if self.global_step > 2:
                self.log_dict(tb_metrics)

            self.linear_metrics.reset()
            self.cluster_metrics.reset()

    # Setting optimizer used for teaching the segmentation layer
    def configure_optimizers(self):
        main_params = list(self.net.parameters())
        if self.cfg.optim == 'adam':
            net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        elif self.cfg.optim == 'sgd':
            net_optim = torch.optim.SGD(main_params, lr=self.cfg.lr, momentum=0.9)
        else:
            net_optim = torch.optim.RMSprop(main_params, lr=self.cfg.lr)
        
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=5e-3)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)
        
        if self.cfg.clr:
            scheduler = torch.optim.lr_scheduler.CyclicLR(net_optim, self.cfg.lr, 10*self.cfg.lr, step_size_up=1000)
            return [net_optim, linear_probe_optim, cluster_probe_optim], [scheduler]
        return [net_optim, linear_probe_optim, cluster_probe_optim]
            
    # function to drop weights and not fail if key is not found in state_dict 
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)

@hydra.main(config_path="configs", config_name="supervised_config.yml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_everything(seed=0)

    print(data_dir)
    print(cfg.output_root)

    sys.stdout.flush()

    train_dataset = DirectoryDataset(
        root=pytorch_data_dir,
        path=cfg.dir_dataset_name,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type)
    )
    
    batch_size = cfg.batch_size
    val_freq = cfg.val_freq

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    # configuring dataloader for experiments trained on certain part of the dataset
    if not cfg.usable_data_ratio is None:
        length = len(train_dataset)
        sampler = SubsetRandomSampler(torch.randperm(len(train_dataset))[:(int)(length*cfg.usable_data_ratio)])
        train_loader = DataLoader(train_dataset, batch_size, shuffle=False, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
    
    # configuring dataloader for experiments trained on certain number of samples
    if not cfg.sample_num is None:
        sample_num = cfg.sample_num if cfg.sample_num < len(train_dataset) else len(train_dataset)
        sampler = SubsetRandomSampler(torch.randperm(len(train_dataset))[:sample_num])
        train_loader = DataLoader(train_dataset, batch_size, shuffle=False, sampler=sampler, num_workers=cfg.num_workers, pin_memory=True)
        print("Num of batches:",len(train_loader))

    val_loader_crop = "center"
    val_dataset = DirectoryDataset(
        root=pytorch_data_dir,
        path=cfg.dir_dataset_name,
        image_set="val",
        transform=get_transform(cfg.res, False, val_loader_crop),
        target_transform=get_transform(cfg.res, True, val_loader_crop),
    )

    val_batch_size = cfg.batch_size

    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

    model = SupervisedSegmenter(cfg.dir_dataset_n_classes, cfg)
    if not cfg.checkpoint_file is None:
        checkpoint = SupervisedSegmenter.load_from_checkpoint(cfg.checkpoint_file, cfg=cfg, strict=False)
        model.net = checkpoint.net

    logger = WandbLogger(project="dipterv", log_model=True, name=cfg.run_name)

    max_epochs = cfg.max_epochs
    if not cfg.sample_num is None:
        max_epochs = 1

    trainer = Trainer(
    log_every_n_steps=cfg.scalar_log_freq,
    logger=logger,
    max_epochs=max_epochs,
    callbacks=[
        ModelCheckpoint(
            dirpath=join(checkpoint_dir, name),
            every_n_train_steps=cfg.checkpoint_freq,
            save_top_k=1,
            monitor="test/linear/mIoU",
            mode="max",
        )
    ],
    accelerator='cuda', val_check_interval=val_freq)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()

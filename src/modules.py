import torch

from utils import *
import torch.nn.functional as F
import dino.vision_transformer as vits
import timm


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        patch_size = self.cfg.dino_patch_size
        self.patch_size = patch_size
        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model_type
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        
        self.dropout = torch.nn.Dropout2d(p=.1)

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")
        
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768

        self.cluster1 = self.make_clusterer(self.n_feats)
        self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

        #making possible the freezing of different parts of the net by config
        if cfg.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval().cuda()
            if not cfg.freeze_embedding:
                for p in self.model.patch_embed.parameters():
                    p.requires_grad = True
                self.model.patch_embed.train().cuda()
        else:
            self.model.cuda()

        if self.cfg.freeze_segmenter:
            for p in self.cluster1.parameters():
                p.requires_grad = False
            self.cluster1.eval().cuda()
            for p in self.cluster2.parameters():
                p.requires_grad = False
            self.cluster2.eval().cuda()
        else:
            self.cluster1.cuda()
            self.cluster2.cuda()

        

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img, n=1, return_class_feat=False):
        if self.cfg.freeze_backbone:
            self.model.eval()
            with torch.no_grad():
                image_feat = self.model_forward(img, n)
        else:
            image_feat = self.model_forward(img, n)

        if self.cfg.freeze_segmenter:
            self.cluster1.eval()
            self.cluster2.eval()
            with torch.no_grad():
                return self.segmenter_forward(image_feat)
        else:
            return self.segmenter_forward(image_feat)

    def model_forward(self, img, n):
        assert (img.shape[2] % self.patch_size == 0)
        assert (img.shape[3] % self.patch_size == 0)

        # get selected layer activations
        feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
        feat, attn, qkv = feat[0], attn[0], qkv[0]

        feat_h = img.shape[2] // self.patch_size
        feat_w = img.shape[3] // self.patch_size

        if self.feat_type == "feat":
            image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
        else:
            raise ValueError("Unknown feat type:{}".format(self.feat_type))

        return image_feat

    def segmenter_forward(self, image_feat):
        code = self.cluster1(self.dropout(image_feat))
        code += self.cluster2(self.dropout(image_feat))

        if self.cfg.dropout:
            return self.dropout(image_feat), code
        else:
            return image_feat, code


class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs

# class for wrapping the convolutional network used for comparison
class CustomFeaturizer(nn.Module):

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        self.model = timm.create_model('convnext_small', pretrained=True, num_classes=0, global_pool='')
        
        self.dropout = torch.nn.Dropout2d(p=.1)

        self.n_feats = 768

        self.cluster1 = self.make_clusterer(self.n_feats)
        self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

        if cfg.freeze_backbone:
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval().cuda()
            
            if not cfg.freeze_embedding:
                for p in self.model.patch_embed.parameters():
                    p.requires_grad = True
                self.model.patch_embed.train().cuda()
        else:
            self.model.cuda()

        if self.cfg.freeze_segmenter:
            for p in self.cluster1.parameters():
                p.requires_grad = False
            self.cluster1.eval().cuda()
            for p in self.cluster2.parameters():
                p.requires_grad = False
            self.cluster2.eval().cuda()
        else:
            self.cluster1.cuda()
            self.cluster2.cuda()

        

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img):
        if self.cfg.freeze_backbone:
            self.model.eval()
            with torch.no_grad():
                image_feat = self.model_forward(img)
        else:
            image_feat = self.model_forward(img)

        if self.cfg.freeze_segmenter:
            self.cluster1.eval()
            self.cluster2.eval()
            with torch.no_grad():
                return self.segmenter_forward(image_feat)
        else:
            return self.segmenter_forward(image_feat)

    def model_forward(self, img):
        return self.model(img)

    def segmenter_forward(self, image_feat):
        code = self.cluster1(self.dropout(image_feat))
        code += self.cluster2(self.dropout(image_feat))

        if self.cfg.dropout:
            return self.dropout(image_feat), code
        else:
            return image_feat, code
    
def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def average_norm(t):
    return t / t.square().sum(1, keepdim=True).sqrt().mean()


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])

# class implementing the contrastive correlation loss function used for training STEGO model in a self-supervised way
class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self, cfg, ):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,
                orig_salience: torch.Tensor, orig_salience_pos: torch.Tensor,
                orig_code: torch.Tensor, orig_code_pos: torch.Tensor,
                ):

        coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]
        coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
        coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.cfg.pos_intra_shift)
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg.pos_inter_shift)

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg.neg_samples):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg.neg_inter_shift)
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)
        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (pos_intra_loss.mean(),
                pos_intra_cd,
                pos_inter_loss.mean(),
                pos_inter_cd,
                neg_inter_loss,
                neg_inter_cd)

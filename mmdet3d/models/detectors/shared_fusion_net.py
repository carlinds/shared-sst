import torch
import numpy as np
from mmcv.runner import force_fp32
from torch.nn import functional as F

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from .voxelnet import VoxelNet
from ..model_utils.transformer import PatchEmbed

def get_patch_coors(unflattened_patches, patch_size):
    batch_size, height, width = unflattened_patches.shape[0], unflattened_patches.shape[1], unflattened_patches.shape[2]
    patch_coors = torch.zeros((height * width * batch_size, 4), device=unflattened_patches.device)

    # Width indices
    patch_coors[:, 3] = torch.arange(width).repeat(height * batch_size)

    # Height and batch indices
    height_indices = np.repeat(np.arange(height), width)
    for batch_index in range(batch_size):
        patch_coors[batch_index * height * width : (batch_index + 1) * height * width, 0] = batch_index
        patch_coors[batch_index * height * width : (batch_index + 1) * height * width, 2] = torch.from_numpy(height_indices)

    # Scale to image size
    patch_coors[:, 2] = patch_coors[:, 2] * patch_size + patch_size // 2
    patch_coors[:, 3] = patch_coors[:, 3] * patch_size + patch_size // 2
    return patch_coors


@DETECTORS.register_module()
class SharedFusionNet(VoxelNet):
    r"""Shared Fusion Net for 3D detection.

    Uses a shared backbone to jointly process lidar and image features.
    The middle encoder is responsible for grouping voxels and patch embeddings together in regional groups.
    The shared backbone performs sparse attention on the regional groups.

    Extension of DynamicVoxelNet, which in turn is a VoxelNet using `dynamic voxelization <https://arxiv.org/abs/1910.06528>`_.
    """

    def __init__(
        self,
        voxel_layer,
        voxel_encoder,
        patch_embedder,
        middle_encoder,
        backbone,
        neck=None,
        bbox_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
        freeze=None,
    ):
        super(SharedFusionNet, self).__init__(
            voxel_layer=voxel_layer,
            voxel_encoder=voxel_encoder,
            middle_encoder=middle_encoder,
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg,
        )
        self.patch_embedder = PatchEmbed(**patch_embedder)
        self.freeze = freeze
        if self.freeze:
            assert (
                type(self.freeze) == list
            ), "The freeze input should be a list of the blocks to freeze"
            # Freeze the patch embedder
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            # turn of voxel_encoder
            for param in self.voxel_encoder.parameters():
                param.requires_grad = False
            if hasattr(self.backbone, "linear0"):
                for param in self.backbone.linear0.parameters():
                    param.requires_grad = False
            for i, block in enumerate(self.backbone.block_list):
                if i in freeze:
                    for param in block.parameters():
                        param.requires_grad = False

    def extract_feat(self, points, img, img_metas):
        """Extract features from points and images."""

        # Voxelize point cloud
        voxels, coors = self.voxelize(points)  # [Batch, Z, Y, X]
        batch_size = coors[-1, 0].item() + 1
        voxel_features, voxel_feature_coors = self.voxel_encoder(voxels, coors)
        voxel_mean, _ = self.voxel_encoder.cluster_scatter(voxels, coors)
        
        # Patchify wide image
        img_wide = torch.cat([img[:, i] for i in self.middle_encoder.camera_order], dim=3)
        patches = self.patch_embedder(img_wide)
        
        # Convert patches to same format as voxels
        unflattened_patches = patches[0].unflatten(1, patches[1])
        patch_features = patches[0].flatten(0, 1)
        patch_coors = get_patch_coors(unflattened_patches, self.patch_embedder.projection.kernel_size[0])

        x = self.middle_encoder(
            voxel_features,
            voxel_feature_coors,
            voxel_mean,
            patch_features,
            patch_coors,
            img_metas,
            batch_size,
        )
        x = self.backbone(x)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_train(
        self, points, img, img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore=None
    ):
        """Training forward function.

        Args:
            points (list[torch.Tensor]): Point cloud of each sample.
            img (list[torch.Tensor]): Image of each sample.
            img_metas (list[dict]): Meta information of each sample
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """
        x = self.extract_feat(points, img, img_metas)
        outs = self.bbox_head(x)
        loss_inputs = outs + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def extract_feats(self, points, img, img_metas):
        """Extract features of multiple samples."""
        return [
            self.extract_feat(pts, im, img_meta)
            for pts, im, img_meta in zip(points, img, img_metas)
        ]

    def simple_test(self, points, img, img_metas, rescale=False):
        """Test function without augmentaiton."""
        x = self.extract_feat(points, img, img_metas)
        outs = self.bbox_head(x)
        bbox_list = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return bbox_results

    def aug_test(self, points, img, img_metas, rescale=False):
        """Test function with augmentaiton."""
        feats = self.extract_feats(points, img, img_metas)

        # only support aug_test for one sample
        aug_bboxes = []
        for x, img_meta in zip(feats, img_metas):
            outs = self.bbox_head(x)
            bbox_list = self.bbox_head.get_bboxes(*outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(
            aug_bboxes, img_metas, self.bbox_head.test_cfg
        )

        return [merged_bboxes]


    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points and coordinates.
        """
        coors = []
        # dynamic voxelization only provide a coors mapping
        for res in points:
            res_coors = self.voxel_layer(res)
            coors.append(res_coors)
        points = torch.cat(points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return points, coors_batch

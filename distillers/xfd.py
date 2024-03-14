import torch
import torch.nn.functional as F
from torch import nn
import math
from timm.models.layers import PatchEmbed

from ._base import BaseDistiller
from .registry import register_distiller
from .utils import GAP1d, TokenFilter, get_module_dict, init_weights, is_cnn_model, set_module_dict, kd_loss, MyPatchMerging, LambdaModule

# @register_distiller
# class XFD(BaseDistiller):
#     requires_feat = True
    
#     def __init__(self, student, teacher, criterion, args, **kwargs):
#         super(XFD, self).__init__(student, teacher, criterion, args)
        
#         assert is_cnn_model(student), 'current XAD implementation only support cnn students!'
        
#         is_cnn_student = is_cnn_model(student)
        
#         self.proj_q = nn.ModuleDict()
#         self.proj_v = nn.ModuleDict()
#         self.proj_k = nn.ModuleDict()
        
#         for stage in self.args.xfd_stage:
#             _, size_s = self.student.stage_info(stage)
#             _, size_t = self.teacher.stage_info(stage)
            
#             if is_cnn_student:
#                 in_chans_s, _, _ = size_s
#                 _, in_chans_t = size_t
                
#                 proj_q = nn.Conv2d(in_chans_s, in_chans_t, 1, 1, 0, bias=False)
#                 proj_v = nn.Conv2d(in_chans_s, in_chans_t, 1, 1, 0, bias=False)

#                 token_num = getattr(teacher, 'num_tokens', 0)  # cls tokens
#                 if token_num != 0:
#                     proj_k = nn.Sequential(
#                         TokenFilter(token_num, remove_mode=True),
#                         GAP1d(),
#                         nn.Linear(in_chans_t, in_chans_t)
#                     )
#                 else:
#                     proj_k = nn.Sequential(
#                         GAP1d(),
#                         nn.Linear(in_chans_t, in_chans_t)
#                     )

#                 set_module_dict(self.proj_k, stage, proj_k)
#                 set_module_dict(self.proj_q, stage, proj_q)
#                 set_module_dict(self.proj_v, stage, proj_v)

#         self.proj_k.apply(init_weights)
#         self.proj_q.apply(init_weights)
#         self.proj_v.apply(init_weights)

    
#     def forward(self, image, label, *args, **kwargs):
#         with torch.no_grad():
#             self.teacher.eval()
#             logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)

#         logits_student, feat_student = self.student(image, requires_feat=True)

#         xfd_losses = []
#         for stage in self.args.xfd_stage:
#             idx_s, _ = self.student.stage_info(stage)
#             idx_t, _ = self.teacher.stage_info(stage)

#             feat_q = get_module_dict(self.proj_q, stage)(feat_student[idx_s])
#             feat_v = get_module_dict(self.proj_v, stage)(feat_student[idx_s])
#             feat_k = get_module_dict(self.proj_k, stage)(feat_teacher[idx_t])

#             token_num = getattr(self.teacher, 'num_tokens', 0)  # cls tokens
#             if token_num != 0:
#                 feat_t = feat_teacher[idx_t][:, 1:, :].mean(1)
#             else: 
#                 feat_t = feat_teacher[idx_t].mean(1)

#             B, _, H, W = feat_q.size()
#             attn_map = F.softmax(torch.einsum('bchw,bc->bhw', [feat_q, feat_k]).flatten(1), dim=1).view(B, H, W)
#             feat_s = torch.einsum('bchw,bhw->bchw', [feat_v, attn_map]).flatten(2).mean(2)

#             xfd_losses.append(F.mse_loss(feat_s, feat_t))

#         loss_xfd = self.args.xfd_loss_weight * sum(xfd_losses)
#         loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
#         loss_kd = self.args.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.args.kd_temperature)

#         losses_dict = {
#             "loss_gt": loss_gt,
#             "loss_kd": loss_kd,
#             "loss_xfd": loss_xfd
#         }

#         return logits_student, losses_dict


@register_distiller
class XFD(BaseDistiller):
    requires_feat = True
    
    def __init__(self, student, teacher, criterion, args, **kwargs):
        super(XFD, self).__init__(student, teacher, criterion, args)
        
        assert is_cnn_model(student), 'current XAD implementation only support cnn students!'
        
        is_cnn_teacher = is_cnn_model(teacher)
        is_cnn_student = is_cnn_model(student)
        
        self.projector = nn.ModuleDict()
        
        for stage in self.args.xfd_stage:
            stage_modules = nn.ModuleDict()
            
            _, size_s = self.student.stage_info(stage)
            _, size_t = self.teacher.stage_info(stage)
            
            if is_cnn_student and not is_cnn_teacher:
                
                token_num = getattr(teacher, 'num_tokens', 1)  # cls tokens
                feature_filter_s = nn.Identity()
                in_chans, H, W = size_s
                patch_num, embed_dim = size_t
                patch_grid = int((patch_num - token_num) ** .5)
                
                if H >= patch_grid:
                    feature_filter_t = TokenFilter(token_num, remove_mode=True)  # remove cls token
                    patch_size = H // patch_grid
                    assert patch_size * patch_grid == H
                    aligner = PatchEmbed(H, patch_size, in_chans, embed_dim)
                    # proj_k = nn.Linear(embed_dim, embed_dim)
                    # proj_v = nn.Linear(embed_dim, embed_dim)
                    proj_q = nn.Linear(embed_dim, embed_dim)
                    proj_o = nn.Linear(embed_dim, embed_dim)
                else:
                    feature_filter_t = nn.Sequential(
                        TokenFilter(token_num, remove_mode=True),  # remove cls token
                        MyPatchMerging(H * W)
                    )
                    scale = patch_grid // H
                    assert scale * H == patch_grid
                    aligner = nn.Sequential(
                        nn.Conv2d(in_chans, embed_dim * scale ** 2, 1, 1, 0),
                        LambdaModule(lambda x: torch.einsum('nchw->nhwc', x)),
                        nn.Flatten(start_dim=1, end_dim=2)
                    )
                    # proj_k = nn.Linear(embed_dim * scale ** 2, embed_dim * scale ** 2)
                    # proj_v = nn.Linear(embed_dim * scale ** 2, embed_dim * scale ** 2)
                    proj_q = nn.Linear(embed_dim * scale ** 2, embed_dim * scale ** 2)
                    proj_o = nn.Linear(embed_dim * scale ** 2, embed_dim * scale ** 2)
                    
                    
            stage_modules['feature_filter_s'] = feature_filter_s
            stage_modules['feature_filter_t'] = feature_filter_t
            stage_modules['aligner'] = aligner
            # stage_modules['proj_k'] = proj_k
            # stage_modules['proj_v'] = proj_v
            stage_modules['proj_q'] = proj_q
            stage_modules['proj_o'] = proj_o
            
            set_module_dict(self.projector, stage, stage_modules)
            
        self.projector.apply(init_weights)

    
    def forward(self, image, label, *args, **kwargs):
        with torch.no_grad():
            self.teacher.eval()
            logits_teacher, feat_teacher = self.teacher(image, requires_feat=True)

        logits_student, feat_student = self.student(image, requires_feat=True)

        xfd_losses = []
        for stage in self.args.xfd_stage:
            idx_s, _ = self.student.stage_info(stage)
            idx_t, _ = self.teacher.stage_info(stage)
            
            feat_t = feat_teacher[idx_t]
            feat_s = feat_student[idx_s]

            feat_t = get_module_dict(self.projector, stage)['feature_filter_t'](feat_t)
            feat_s = get_module_dict(self.projector, stage)['feature_filter_s'](feat_s)
            feat_s_aligned = get_module_dict(self.projector, stage)['aligner'](feat_s)

            # feat_k = get_module_dict(self.projector, stage)['proj_k'](feat_s_aligned)
            # feat_v = get_module_dict(self.projector, stage)['proj_v'](feat_s_aligned)
            feat_q = get_module_dict(self.projector, stage)['proj_q'](feat_t)

            C = feat_q.size(-1)
            attn_score = torch.matmul(feat_q, feat_s_aligned.transpose(-1,-2)) # [B, L, C] * [B, C, L] => [B, L, L]
            attn_score = attn_score / math.sqrt(C)
            attn_prob = nn.Softmax(dim=-1)(attn_score)
            attn_map = torch.matmul(attn_prob, feat_s_aligned)
            attn_map = get_module_dict(self.projector, stage)['proj_o'](attn_map)
            
            feat_t = feat_t + attn_map

            xfd_losses.append(F.mse_loss(feat_s_aligned, feat_t))

        loss_xfd = self.args.xfd_loss_weight * sum(xfd_losses)
        loss_gt = self.args.gt_loss_weight * self.criterion(logits_student, label)
        # loss_kd = self.args.kd_loss_weight * kd_loss(logits_student, logits_teacher, self.args.kd_temperature)

        losses_dict = {
            "loss_gt": loss_gt,
            # "loss_kd": loss_kd,
            "loss_xfd": loss_xfd,
        }

        return logits_student, losses_dict
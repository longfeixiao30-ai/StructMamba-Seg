custom_imports = dict(
    imports=[
        'model',
        'classification.models.new_iou_boundary_loss',
    ],
    allow_failed_imports=False
)

_base_ = [
    '../_base_/models/upernet_swin.py',
    '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExponentialMovingAverage',
        begin_iter=0,
        momentum=0.0002,
        update_buffers=False
    )
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

head_norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='MM_VSSM',
        out_indices=(0, 1, 2, 3),
        pretrained=None,  # 这里保持 None，因为下面用 load_from 做 warm start
        dims=96,
        depths=(2, 2, 6, 2),
        ssm_d_state=16,
        ssm_dt_rank='auto',
        ssm_ratio=1.0,
        mlp_ratio=4.0,
        downsample_version='v3',
        patchembed_version='v2',
        drop_path_rate=0.3,

        # 改进1四个开关
        use_ep=True,
        use_mc=True,
        use_pc=True,
        use_sa=False,

        # 改进2接口
        use_lprc=False,
        lprc_stage_start=2,
        lprc_window_tokens=16,
        lprc_pairs_per_window=24,
        lprc_tau=1.0,
        lprc_margin=0.0,
        lprc_weight=0.010,
        use_ldso=False,
    ),
    decode_head=dict(
        in_channels=[96, 192, 384, 768],
        num_classes=150,
        norm_cfg=head_norm_cfg,
        loss_decode=[
            dict(
                type='CrossEntropyLoss',
                use_sigmoid=False,
                loss_weight=1.0,
                avg_non_ignore=True
            ),
            dict(
                type='IoUBoundaryLoss',
                loss_weight=1.0,
                ignore_index=255,
                lovasz_weight=0.78,
                boundary_weight=0.10,
                boundary_radius=2,
                boundary_bce_weight=1.0,
                boundary_dice_weight=1.0
            )
        ]
    ),
    auxiliary_head=dict(
        in_channels=384,
        num_classes=150,
        norm_cfg=head_norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,
            loss_weight=0.30,
            avg_non_ignore=True
        )
    )
)

# 用你当前 cityscapes 最优/已有权重做 warm start
# 注意：因为 ADE20K 的类别数是 150，decode_head / auxiliary_head 的分类层形状会不匹配，
# 加载时这些 head 参数会自动跳过，这是正常现象。
load_from = '/root/autodl-tmp/work_dirs/upernet_ms_defm_cityscapes_120k_fuse12_ce_ls_br_recipe_plus_v1/best_mIoU_iter_16000.pth'

data_root = '/root/autodl-tmp/datasets/ADEChallengeData2016'

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        data_root=data_root
    )
)

val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root
    )
)

test_dataloader = val_dataloader

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW',
        lr=0.00004,
        betas=(0.9, 0.999),
        weight_decay=0.01
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.0),
            'relative_position_bias_table': dict(decay_mult=0.0),
            'norm': dict(decay_mult=0.0),
            'conv_offset': dict(lr_mult=3.0),
            'decode_head': dict(lr_mult=1.5),
            'auxiliary_head': dict(lr_mult=1.5),
        }
    )
)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=False,
        interval=8000,
        save_best='mIoU',
        rule='greater',
        save_last=False,
        max_keep_ckpts=1
    )
)

train_cfg = dict(
    type='IterBasedTrainLoop',
    max_iters=160000,
    val_interval=8000
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-3,
        by_epoch=False,
        begin=0,
        end=1500
    ),
    dict(
        type='PolyLR',
        eta_min=1e-6,
        power=1.0,
        begin=1500,
        end=160000,
        by_epoch=False,
    )
]

work_dir = '/root/autodl-tmp/work_dirs/upernet_ms_defm_ade20k_160k_fuse12_ce_ls_br_recipe_plus_v1'
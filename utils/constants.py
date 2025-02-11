"""


Some constants to visualize distance metrics 


"""

# Dataset codes (MIDOG 2025 Test Set) ----------------------------------------------------------------------------- 

MIDOG_ABBREVATIONS = {
    0: 'HMEL',
    1: 'HAC',
    2: 'HBLC',
    3: 'CMC',
    4: 'CCMCT',
    5: 'HMEN',
    6: 'HCOC',
    7: 'CHAS',
    8: 'FSTS',
    9: 'FLYM'
}

# Yolov7 architeture  ---------------------------------------------------------------------------


YOLO_LAYERS = [
    # backbone layers 
    'model.11',     # last of P2
    'model.24',     # last of P3
    'model.37',     # last of P4
    'model.50',     # last of P5

    # neck layers
    'model.63',     # last of first top down ELAN (after P5 and connected to P4)
    'model.75',     # last of second top down (ELAN?) (after first ELAN and connected to P3)
    'model.88',     # last of first bottom down ELAN (P4)
    'model.101',    # last of second bottom down ELAN (P5)

    'model.102',    # last layer of neck connected to head P3
    'model.103',    # last layer of neck connected to head P4
    'model.104',    # last layer of neck connected to head P5

    # head layers 
    'model.105.m.0',    # head P3
    'model.105.m.1',    # head P4
    'model.105.m.2',    # head P5
]


YOLO_LAYER_CODES = {
    # backbone layers 
    'model.11': 'C2',
    'model.24': 'C3',
    'model.37': 'C4',
    'model.50': 'C5',

    # neck layers
    'model.63': 'Up-Neck-P4',
    'model.75': 'Up-Neck-P3',
    'model.88': 'Down-Neck-P4',
    'model.101': 'Down-Neck-P5',
    'model.102': 'P3',
    'model.103': 'P4',
    'model.104': 'P5',

    # head layers 
    'model.105.m.0': 'O3',   
    'model.105.m.1': 'O4',   
    'model.105.m.2': 'O5',   
}


# Yolov7 D6 architeture  ---------------------------------------------------------------------------

# layers for feature extraction
YOLO_D6_LAYERS = [
    # backbone layers 
    'model.14',     # last of P2
    'model.27',     # last of P3
    'model.40',     # last of P4
    'model.53',     # last of P5
    'model.66',     # last of P6

    # neck layers 
    'model.83',     # last of neck connected to P5
    'model.99',     # last of neck connected to P4
    'model.115',    # last of neck connected to P3
    'model.129',    # last of neck connected to P2 (?)
    'model.143',    # last of neck connected to P1 (?)
    'model.157',    
    'model.158',    # last layer of neck connected to head P3
    'model.159',    # last layer of neck connected to head P4
    'model.160',    # last layer of neck connected to head P5
    'model.161',    # last layer of neck connected to head P6

    # head layers 
    'model.166.m.0',    # head P3
    'model.166.m.1',    # head P4
    'model.166.m.2',    # head P5
    'model.166.m.3',    # head P6
]


YOLO_D6_LAYER_CODES = {
    # backbone layers 
    'model.14': 'C2',
    'model.27': 'C3',
    'model.40': 'C4',
    'model.53': 'C5',
    'model.66': 'C6',

    # neck layers
    'model.83': 'Up-Neck-P5',
    'model.99': 'Up-Neck-P4',
    'model.115': 'Up-Neck-P3',
    'model.129': 'Down-Neck-P4',
    'model.143': 'Down-Neck-P5',
    'model.157': 'Down-Neck-P6',
    'model.158': 'P3',
    'model.159': 'P4',
    'model.160': 'P5',
    'model.161': 'P6',

    # head layers 
    'model.166.m.0': 'O3',   
    'model.166.m.1': 'O4',   
    'model.166.m.2': 'O5',   
    'model.166.m.3': 'O6',   
}



# FCOS  & RetinaNet architeture  ---------------------------------------------------------------------------


FCOS_LAYER_CODES = {
        
        # common layers 
        'model.backbone.body.layer1': 'C2',
        'model.backbone.body.layer2': 'C3',
        'model.backbone.body.layer3': 'C4',
        'model.backbone.body.layer4': 'C5',
        'model.backbone.fpn.layer_blocks.0': 'P2',
        'model.backbone.fpn.layer_blocks.1': 'P3',
        'model.backbone.fpn.layer_blocks.2': 'P4',
        'model.backbone.fpn.layer_blocks.3': 'P5',
        'model.backbone.fpn.extra_blocks.pool': 'P6',

        # fcos, retinanet layers
        'model.head.classification_head.conv_0': 'H2',
        'model.head.classification_head.conv_1': 'H3',
        'model.head.classification_head.conv_2': 'H4',
        'model.head.classification_head.conv_3': 'H5',
        'model.head.classification_head.conv_4': 'H6',
        'model.head.classification_head.cls_logits_0': 'O2',
        'model.head.classification_head.cls_logits_1': 'O3',
        'model.head.classification_head.cls_logits_2': 'O4',
        'model.head.classification_head.cls_logits_3': 'O5',
        'model.head.classification_head.cls_logits_4': 'O6',

}
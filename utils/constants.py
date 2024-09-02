"""


Some constants to visualize distance metrics 


"""

# TODO: get layers from before and from head 
# 158-161 should be the layers attached to the DetectHead

# layers for feature extraction
YOLO_LAYERS = [
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


LAYER_CODES = {
    # backbone layers 
    'model.14': 'P2',
    'model.27': 'P3',
    'model.40': 'P4',
    'model.53': 'P5',
    'model.66': 'P6',

    # neck layers
    'model.83': 'Up-Neck-P5',
    'model.99': 'Up-Neck-P4',
    'model.115': 'Up-Neck-P3',
    'model.129': 'Down-Neck-P4',
    'model.143': 'Down-Neck-P5',
    'model.157': 'Down-Neck-P6',
    'model.158': 'Final-Neck-P3',
    'model.159': 'Final-Neck-P4',
    'model.160': 'Final-Neck-P5',
    'model.161': 'Final-Neck-P6',

    # head layers 
    'model.166.m.0': 'Head-P3',   
    'model.166.m.1': 'Head-P4',   
    'model.166.m.2': 'Head-P5',   
    'model.166.m.3': 'Head-P6',   
}
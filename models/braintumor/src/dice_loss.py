import torch
import numpy as np
def dice_loss(pred,target,e=1e-6):
    inter=2*(pred*target).sum()+e
    union=(pred).sum()+(target).sum()+e
    return 1-(inter/union).sum()

def bce_dice_loss(inputs, target):
    dicescore = dice_loss(inputs, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)
    
    return dicescore+bceloss
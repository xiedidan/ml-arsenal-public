from dependencies import *

def dice(input:Tensor, targs:Tensor, best_thr=0.2, iou:bool=False, eps:float=1e-8)->Rank0Tensor:
    n = targs.shape[0]
    
    input = torch.softmax(input, dim=1)[:,1,...].view(n,-1)
    input = (input > best_thr).long()
    
    input[input.sum(-1) < noise_th,...] = 0.0
    
    #input = input.argmax(dim=1).view(n,-1)
    targs = targs.view(n,-1)
    
    intersect = (input * targs).sum(-1).float()
    union = (input+targs).sum(-1).float()
    
    if not iou:
        return ((2.0 * intersect + eps) / (union+eps)).mean()
    else:
        return ((intersect + eps) / (union - intersect + eps)).mean()

def dice_accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    intersection = p & t
    union        = p | t
    dice = (intersection.float().sum(1)+EPS) / (union.float().sum(1)+EPS)

    if is_average:
        dice = dice.sum()/batch_size
        return dice
    else:
        return dice

def accuracy(prob, truth, threshold=0.5,  is_average=True):
    batch_size = prob.size(0)
    p = prob.detach().view(batch_size,-1)
    t = truth.detach().view(batch_size,-1)

    p = p>threshold
    t = t>0.5
    correct = ( p == t).float()
    accuracy = correct.sum(1)/p.size(1)

    if is_average:
        accuracy = accuracy.sum()/batch_size
        return accuracy
    else:
        return accuracy

# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
 
    print('\nsucess!')

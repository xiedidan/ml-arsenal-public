from dependencies import *

def dice_metric(input, targs, noise_th, best_thr=0.2, iou=False, eps=1e-8, logger=None):
    n = targs.shape[0]

    p = input.detach().view(n, -1)
    t = targs.detach().view(n, -1)
    
    if best_thr > 0:
        p = (p > best_thr).float()

    p[p.sum(-1) < noise_th,...] = 0.0
    
    t = (t > 0.5).float()

    intersect = (p * t).sum(-1).float()
    union = (p + t).sum(-1).float()
    
    if logger is not None:
        logger.debug('\np.sum(): {:.2f},\tt.sum(): {:.2f},\ti: {:.2f},\tu: {:.2f},\tdice: {:.8f}'.format(p.sum(), t.sum(), intersect.sum(), union.sum(), ((2.0 * intersect) / (union + eps)).mean()))
    
    if not iou:
        return ((2.0 * intersect + eps) / (union + eps)).mean()
        # return ((2.0 * intersect) / (union + eps)).mean()
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

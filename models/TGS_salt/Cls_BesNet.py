from dependencies import *

class ConvBn2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True, nonlinearity='relu'):
        super(ConvBn2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=groups, bias=False)
        #self.bn = SynchronizedBatchNorm2d(out_channels)
        self.bn = nn.BatchNorm2d(out_channels)
        
        # init
        if nonlinearity=='relu' or nonlinearity=='leaky-relu':
            nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity=nonlinearity)
        elif nonlinearity=='sigmoid':
            nn.init.xavier_normal_(self.conv.weight)
        else:
            print('unknown nonlinearity: {}'.format(nonlinearity))
            
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, z):
        x = self.conv(z)
        x = self.bn(x)
        return x

class sSE(nn.Module):
    def __init__(self, out_channels):
        super(sSE, self).__init__()
        self.conv = ConvBn2d(in_channels=out_channels,out_channels=1,kernel_size=1,padding=0,nonlinearity='sigmoid')
        
    def forward(self,x):
        x=self.conv(x)
        #print('spatial',x.size())
        x=F.sigmoid(x)
        return x

class cSE(nn.Module):
    def __init__(self, out_channels):
        super(cSE, self).__init__()
        self.conv1 = ConvBn2d(in_channels=out_channels,out_channels=int(out_channels/2),kernel_size=1,padding=0,nonlinearity='relu')
        self.conv2 = ConvBn2d(in_channels=int(out_channels/2),out_channels=out_channels,kernel_size=1,padding=0,nonlinearity='sigmoid')
    def forward(self,x):
        x=nn.AvgPool2d(x.size()[2:])(x)
        #print('channel',x.size())
        x=self.conv1(x)
        x=F.relu(x)
        x=self.conv2(x)
        x=F.sigmoid(x)
        return x



class Decoder(nn.Module):
    def __init__(self, in_channels, channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = ConvBn2d(in_channels, channels, kernel_size=3, padding=1,nonlinearity='relu')
        self.conv2 = ConvBn2d(channels, out_channels, kernel_size=3, padding=1,nonlinearity='relu')
        self.spatial_gate = sSE(out_channels)
        self.channel_gate = cSE(out_channels)

    def forward(self, x, e=None):
        x = F.upsample(x, scale_factor=2, mode='bilinear', align_corners=True)
        #print('x',x.size())
        #print('e',e.size())
        if e is not None:
            x = torch.cat([x,e],1)

        x = F.relu(self.conv1(x),inplace=True)
        x = F.relu(self.conv2(x),inplace=True)
        #print('x_new',x.size())
        g1 = self.spatial_gate(x)
        #print('g1',g1.size())
        g2 = self.channel_gate(x)
        #print('g2',g2.size())
        x = g1*x + g2*x

        return x

class ClsBesNet(nn.Module):

    def criterion3(self,logit, truth):
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)
        loss = L.lovasz_hinge_relu(logit, truth, per_image=True, ignore=None)
        return loss

    def criterion(self,logit, truth):
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)
        loss = L.lovasz_hinge(logit, truth, per_image=True, ignore=None)
        return loss
    
    def criterion2(self,logit, truth):
        metric = torch.nn.BCEWithLogitsLoss(size_average=True, reduction='none')
        logit = logit.squeeze(1)
        truth = truth.squeeze(1)
        loss = metric(logit, truth)
        return loss
    
    def focal_loss(self, output, target, alpha, gamma, OHEM_percent):
        output = output.contiguous().view(-1)
        target = target.contiguous().view(-1)

        max_val = (-output).clamp(min=0)
        loss = output - output * target + max_val + ((-max_val).exp() + (-output - max_val).exp()).log()

        # This formula gives us the log sigmoid of 1-p if y is 0 and of p if y is 1
        invprobs = F.logsigmoid(-output * (target * 2 - 1))
        focal_loss = alpha * (invprobs * gamma).exp() * loss

        # Online Hard Example Mining: top x% losses (pixel-wise). Refer to http://www.robots.ox.ac.uk/~tvg/publications/2017/0026.pdf
        OHEM, _ = focal_loss.topk(k=int(OHEM_percent * [*focal_loss.shape][0]))
        return OHEM.mean() 

    def __init__(self, dropout_rate=0.):
        super().__init__()
        self.dropout_rate = dropout_rate
        
        self.resnet = models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(
            self.resnet.conv1,
            self.resnet.bn1,
            self.resnet.relu,
        )

        self.encoder2 = self.resnet.layer1 # 64
        self.encoder3 = self.resnet.layer2 #128
        self.encoder4 = self.resnet.layer3 #256
        self.encoder5 = self.resnet.layer4 #512

        self.center = nn.Sequential(
            ConvBn2d(512,512,kernel_size=3,padding=1,nonlinearity='relu'),
            nn.ReLU(inplace=True),
            ConvBn2d(512,256,kernel_size=3,padding=1,nonlinearity='relu'),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2,stride=2),
        )

        self.decoder5 = Decoder(256+512,512,64)
        self.decoder4 = Decoder(64 +256,256,64)
        self.decoder3 = Decoder(64 +128,128,64)
        self.decoder2 = Decoder(64 +64 ,64 ,64)
        self.decoder1 = Decoder(64     ,32 ,64)

        self.mask_logit = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )
        
        self.boundary_logit = nn.Sequential(
            nn.Conv2d(384, 64, kernel_size=3, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, padding=0),
        )

        self.class_logit = nn.Linear(4096, 1)
        
        # init logits
        conv_count = 0
        for l in self.mask_logit.modules():
            if isinstance(l, nn.Conv2d):
                if conv_count == 0:
                    nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif conv_count == 1:
                    nn.init.xavier_normal_(l.weight)
                else:
                    print('got more Conv2d in logit: {}'.format(l))
                    
                conv_count += 1
                
        conv_count = 0
        for l in self.boundary_logit.modules():
            if isinstance(l, nn.Conv2d):
                if conv_count == 0:
                    nn.init.kaiming_normal_(l.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif conv_count == 1:
                    nn.init.xavier_normal_(l.weight)
                else:
                    print('got more Conv2d in logit: {}'.format(l))
                    
                conv_count += 1

        for l in self.class_logit.modules():
            if isinstance(l, nn.Linear):
                nn.init.normal_(l.weight)
                nn.init.constant_(l.bias, 0)

    def forward(self, x):
        mean=[0.491, 0.491, 0.491]
        std=[0.249,0.249,0.249]
        x=torch.cat([
           (x-mean[2])/std[2],
           (x-mean[1])/std[1],
           (x-mean[0])/std[0],
        ],1)
        '''
        x=torch.cat([
           x,
           x,
           x
        ],1)
        '''

        e1 = self.conv1(x)
        #print(e1.size())
        e2 = self.encoder2(e1)
        #print('e2',e2.size())
        e3 = self.encoder3(e2)
        #print('e3',e3.size())
        e4 = self.encoder4(e3)
        #print('e4',e4.size())
        e5 = self.encoder5(e4)
        #print('e5',e5.size())

        c = self.center(e5)
        #print('f',f.size())
        d5 = self.decoder5(c, e5)
        d4 = self.decoder4(d5,e4)
        d3 = self.decoder3(d4,e3)
        d2 = self.decoder2(d3,e2)
        d1 = self.decoder1(d2)
        #print('d1',d1.size())

        f = torch.cat((
            F.upsample(e1,scale_factor= 2, mode='bilinear',align_corners=False),
            d1,
            F.upsample(d2,scale_factor= 2, mode='bilinear',align_corners=False),
            F.upsample(d3,scale_factor= 4, mode='bilinear',align_corners=False),
            F.upsample(d4,scale_factor= 8, mode='bilinear',align_corners=False),
            F.upsample(d5,scale_factor=16, mode='bilinear',align_corners=False),
        ),1)
        #print('hc',f.size())
        
        if self.dropout_rate > 0:
            # TODO : add class_logit
            mask_f = F.dropout2d(f, p=self.dropout_rate, training=self.training)
            mask_logit = self.mask_logit(mask_f)
        
            boundary_f = F.dropout2d(f, p=self.dropout_rate, training=self.training)
            boundary_logit = self.boundary_logit(boundary_f)
        else:
            mask_logit = self.mask_logit(f)
            boundary_logit = self.boundary_logit(f)

            n = c.shape[0]
            c = F.adaptive_avg_pool2d(c, 4).view(n, -1)
            class_logit = self.class_logit(c)

        return mask_logit, boundary_logit, class_logit


    def criterion1(self, logit, truth ):
        loss = FocalLoss2d(gamma=0.5)(logit, truth, type='sigmoid')
        return loss

    # def criterion(self,logit, truth):
    #     loss = F.binary_cross_entropy_with_logits(logit, truth)
    #     return loss

    def class_criterion(self, c_logit, c_truth):
        loss = F.binary_cross_entropy_with_logits(c_logit, c_truth, reduction='none')

        return loss

    def boundary_criterion(self, b_logit, b_truth, weights=None, reduction=True):
        # wbce
        logit = b_logit.view(-1)
        truth = b_truth.view(-1).float()
        assert(logit.shape==truth.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        
        if weights is None:
            if reduction:
                loss = loss.mean()
        else:
            pos = (truth>0.5).float()
            neg = (truth<0.5).float()
            pos_weight = pos.sum().item() + 1e-12
            neg_weight = neg.sum().item() + 1e-12
            if reduction:
                loss = (weights[0]*pos*loss/pos_weight + weights[1]*neg*loss/neg_weight).sum()
            else:
                loss = (weights[0]*pos*loss/pos_weight + weights[1]*neg*loss/neg_weight)
                
        return loss

    def mask_criterion(self, m_logit, b_logit, m_truth, b_truth, alpha=2., beta=0.1, weights=None, reduction=True):
        # wbce
        logit = m_logit.view(-1)
        truth = m_truth.view(-1)
        
        # do NOT calc grad for boundary branch
        with torch.no_grad():
            blogit = b_logit.view(-1)
            bprob = torch.sigmoid(blogit)
            btruth = b_truth.view(-1).float()

            b_temp = (beta - bprob).float()
            b_mask = ((b_temp > 0.) * (btruth > 0.5)).float()
            b_enhance = b_mask * (alpha * b_temp)
        
        assert(logit.shape==truth.shape)
        assert(blogit.shape==btruth.shape)
        assert(logit.shape==blogit.shape)

        loss = F.binary_cross_entropy_with_logits(logit, truth, reduction='none')
        
        if weights is not None:
            # class balance
            pos = (truth>0.5).float()
            neg = (truth<0.5).float()
            pos_weight = pos.sum().item() + 1e-12
            neg_weight = neg.sum().item() + 1e-12
            loss = weights[0]*pos*loss/pos_weight + weights[1]*neg*loss/neg_weight
        
            # boundary enhancement - enchance boundary pixels with low prob
            if reduction:
                loss = ((1. + b_enhance) * loss).sum()
            else:
                loss = ((1. + b_enhance) * loss)
        else:
            if reduction:
                loss = loss.mean()

        return loss

    def metric(self, logit, truth, noise_th, threshold=0.2, logger=None):
        prob = torch.sigmoid(logit)
        # dice = dice_accuracy(prob, truth, threshold=threshold, is_average=True)
        # dice = accuracy(prob, truth, threshold=threshold, is_average=True)
        dice = dice_metric(prob, truth, noise_th, best_thr=threshold, iou=False, eps=1e-8, logger=logger)
        return dice

    def set_mode(self, mode, is_freeze_bn=False ):
        self.mode = mode
        if mode in ['eval', 'valid', 'test']:
            self.eval()
        elif mode in ['train']:
            self.train()
            if is_freeze_bn==True: ##freeze
                for m in self.modules():
                    if isinstance(m, nn.BatchNorm2d) or isinstance(m,SynchronizedBatchNorm2d):
                        m.eval()
                        m.weight.requires_grad = False
                        m.bias.requires_grad   = False

        else:   
        	raise NotImplementedError

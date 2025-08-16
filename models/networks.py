import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from .spectralNormalization import SpectralNorm
###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda num_features: nn.Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy in ('lambda', 'linear'): # Combined lambda and linear
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        T_total = max(1, getattr(opt, 'niter', 0) + getattr(opt, 'niter_decay', 0)) # Corrected T_max with protection
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_total, eta_min=0)
    else:
        raise NotImplementedError(f'learning rate policy [{opt.lr_policy}] is not implemented')
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        device = torch.device(f'cuda:{gpu_ids[0]}')
        net.to(device)
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[],
             csca_config=None, training_stage='full'):
    """
    定义CSCA增强的生成器网络

    Args:
        csca_config: CSCA配置对象
        training_stage: CSCA训练阶段
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'resnet_9blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                             csca_config=csca_config, training_stage=training_stage, norm_type_str=norm)
    elif netG == 'resnet_6blocks':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                             csca_config=csca_config, training_stage=training_stage, norm_type_str=norm)
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, norm_type_str=norm)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, norm_type_str=norm)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    if norm=="spectral":
        if netD == 'basic':
            net = NLayerDiscriminatorSN(input_nc, ndf, n_layers=3, use_sigmoid=use_sigmoid)
        elif netD == 'n_layers':
            net = NLayerDiscriminatorSN(input_nc, ndf, n_layers_D, use_sigmoid=use_sigmoid)
    else:
        norm_layer = get_norm_layer(norm_type=norm)

        if netD == 'basic':
            net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, norm_type_str=norm)
        elif netD == 'n_layers':
            net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, norm_type_str=norm)
        elif netD == 'pixel':
            net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid, norm_type_str=norm)
        else:
            raise NotImplementedError('Discriminator model name [%s] is not recognized' % netD)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()
    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
#################################################################################
#                    Critic Loss for Wassertein Gan GP                          #
#################################################################################

class GradPenalty(nn.Module):
    def __init__(self, use_cuda):
        super(GradPenalty, self).__init__()
        self.use_cuda = use_cuda
    def forward(self, critic, real_data, fake_data):
        alpha = torch.rand_like(real_data)

        assignGPU = lambda x: x.cuda() if self.use_cuda else x
        alpha = assignGPU(alpha)

        interpolates = alpha*real_data + (1-alpha)*fake_data.detach()
        interpolates = assignGPU(interpolates)
        interpolates = torch.autograd.Variable(interpolates, requires_grad = True)

        critic_interpolates = critic(interpolates)

        gradients = torch.autograd.grad(outputs=critic_interpolates, inputs=interpolates,
                                        grad_outputs=assignGPU(torch.ones(critic_interpolates.size())),
                                        create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1)-1)**2).mean()
        return gradient_penalty

#####
#####

# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect',
                 csca_config=None, training_stage='full', norm_type_str='batch'): # Added norm_type_str
        """
        CSCA增强的ResNet生成器

        Args:
            input_nc: 输入通道数
            output_nc: 输出通道数
            ngf: 生成器特征数
            norm_layer: 归一化层
            use_dropout: 是否使用dropout
            n_blocks: ResNet块数量
            padding_type: 填充类型
            csca_config: CSCA配置对象
            training_stage: CSCA训练阶段
            norm_type_str: 归一化类型字符串 (e.g., 'batch', 'instance', 'none')
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.training_stage = training_stage

        # 导入CSCA配置
        if csca_config is None:
            from .CSCA import CSCAConfig
            csca_config = CSCAConfig()
        self.csca_config = csca_config

        use_bias = (norm_type_str == 'instance' or norm_type_str == 'none')

        # 构建模型层
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        # 下采样层
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        # CSCA增强的ResNet块
        mult = 2**n_downsampling
        self.resnet_blocks = nn.ModuleList()
        for i in range(n_blocks):
            # 默认启用CSCA，后续可通过set_csca_mask关闭
            block = ResnetBlock(ngf * mult,
                              padding_type=padding_type,
                              norm_layer=norm_layer,
                              use_dropout=use_dropout,
                              use_bias=use_bias,
                              csca_config=self.csca_config,
                              training_stage=training_stage,
                              enable_csca=True)
            self.resnet_blocks.append(block)
            model.append(block)

        # 上采样层
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        # 输出层
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

    def set_csca_training_stage(self, stage):
        """设置所有ResNet块的CSCA训练阶段"""
        self.training_stage = stage
        for block in self.resnet_blocks:
            block.set_training_stage(stage)

    def set_csca_mask(self, csca_enabled_indices=None):
        """
        设置哪些ResNet块启用CSCA模块。

        Args:
            csca_enabled_indices (list or set): 启用CSCA的块的索引列表 (0-based)。
                                                如果为None，则不进行任何更改。
        """
        if csca_enabled_indices is None:
            return
        
        for i, block in enumerate(self.resnet_blocks):
            if i in csca_enabled_indices:
                block.enable_csca = True
            else:
                block.enable_csca = False

    def get_csca_losses(self):
        """获取所有CSCA块的正则化损失"""
        total_loss = 0.0
        for block in self.resnet_blocks:
            total_loss += block.get_csca_losses()
        return total_loss


# Define a resnet block with integrated CSCA attention
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias,
                 csca_config=None, training_stage='full', enable_csca=True):
        """
        CSCA增强的ResNet块

        Args:
            dim: 特征维度
            padding_type: 填充类型
            norm_layer: 归一化层
            use_dropout: 是否使用dropout
            use_bias: 是否使用bias
            csca_config: CSCA配置对象
            training_stage: CSCA训练阶段 ('mid_only', 'final_only', 'full')
            enable_csca: 是否启用CSCA模块
        """
        super(ResnetBlock, self).__init__()
        self.training_stage = training_stage
        self.enable_csca = enable_csca

        # 导入CSCA模块
        from .CSCA import MidAttentionModule, FinalAttentionModule, CSCAConfig

        # 使用默认配置或传入的配置
        if csca_config is None:
            csca_config = CSCAConfig()
        self.csca_config = csca_config

        # 构建分离的卷积层（便于插入注意力）
        self.conv1_block, self.conv2_block = self.build_conv_blocks(
            dim, padding_type, norm_layer, use_dropout, use_bias
        )

        # 集成CSCA注意力模块
        self.mid_attention = MidAttentionModule(
            dim,
            init_gate=csca_config.mid_init_gate,
            l1_weight=csca_config.l1_weight
        )

        self.final_attention = FinalAttentionModule(
            dim,
            init_gate=csca_config.final_init_gate,
            reduction=csca_config.reduction,
            l1_weight=csca_config.l1_weight,
            config=csca_config # Pass the whole config
        )

        self.res_scale = nn.Parameter(torch.tensor(1.0)) # Bugfix #7: 修正缺失的残差缩放参数

    def build_conv_blocks(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        构建分离的卷积块，便于插入CSCA注意力

        Returns:
            tuple: (conv1_block, conv2_block)
        """
        # 第一个卷积块: Pad -> Conv -> Norm -> ReLU -> (Dropout?)
        conv1_layers = []
        p = 0
        if padding_type == 'reflect':
            conv1_layers += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv1_layers += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv1_layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                        norm_layer(dim),
                        nn.ReLU(True)]
        if use_dropout:
            conv1_layers += [nn.Dropout(0.5)]

        # 第二个卷积块: Pad -> Conv -> Norm (注意：没有ReLU)
        conv2_layers = []
        p = 0
        if padding_type == 'reflect':
            conv2_layers += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv2_layers += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv2_layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                        norm_layer(dim)]

        return nn.Sequential(*conv1_layers), nn.Sequential(*conv2_layers)

    def forward(self, x):
        """
        CSCA增强的前向传播

        流程：
        1. 第一个卷积块 (Pad -> Conv -> Norm -> ReLU -> Dropout?)
        2. 中间注意力插入点 (可选)
        3. 第二个卷积块 (Pad -> Conv -> Norm)
        4. 最终注意力插入点 (可选)
        5. 残差连接
        """
        residual = x

        # 第一个卷积块
        out = self.conv1_block(residual)

        # 中间注意力增强
        if self.enable_csca and self.training_stage in ['mid_only', 'full']:
            mid_attention = self.mid_attention(out)
            out = out + mid_attention

        # 第二个卷积块
        out = self.conv2_block(out)

        # 最终注意力增强
        if self.enable_csca and self.training_stage in ['final_only', 'full']:
            final_attention = self.final_attention(out)
            out = out + final_attention

        # 残差连接
        return x + self.res_scale * out

    def set_training_stage(self, stage):
        """设置CSCA训练阶段"""
        self.training_stage = stage

    def get_csca_losses(self):
        """获取CSCA的L1正则化损失"""
        # 始终返回同设备 Tensor
        device = next(self.parameters()).device
        if not self.enable_csca:
            return torch.zeros((), device=device)
            
        total = torch.zeros((), device=device)
        seen = set()

        # 递归获取子模块的损失，并使用seen集合去重
        for module in [self.mid_attention, self.final_attention]:
            for submodule in module.modules():
                if hasattr(submodule, 'get_l1_loss'):
                    sid = id(submodule)
                    if sid not in seen:
                        seen.add(sid)
                        total = total + submodule.get_l1_loss()
        return total


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, norm_type_str='batch'): # Added norm_type_str
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True, norm_type_str=norm_type_str)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout, norm_type_str=norm_type_str)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, norm_type_str=norm_type_str)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer, norm_type_str=norm_type_str)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer, norm_type_str=norm_type_str)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, norm_type_str=norm_type_str)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, norm_type_str='batch'): # Added norm_type_str
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        use_bias = (norm_type_str == 'instance' or norm_type_str == 'none') # Updated use_bias logic
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, norm_type_str='batch'): # Added norm_type_str
        super(NLayerDiscriminator, self).__init__()
        use_bias = (norm_type_str == 'instance' or norm_type_str == 'none') # Updated use_bias logic

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class NLayerDiscriminatorSN(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False):
        super(NLayerDiscriminatorSN, self).__init__()
        use_bias = False

        kw = 4
        padw = 1
        sequence = [
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False, norm_type_str='batch'): # Added norm_type_str
        super(PixelDiscriminator, self).__init__()
        use_bias = (norm_type_str == 'instance' or norm_type_str == 'none') # Updated use_bias logic

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

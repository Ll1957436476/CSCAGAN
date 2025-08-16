import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class CycleGANModel(BaseModel):
    def name(self):
        return 'CycleGANModel'

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)
        parser.add_argument('--csca_enable_pattern', type=str, default='mid4',
                            help="CSCA enable pattern: all|none|mid4|indices:1,3,5 (1-based)|mask:011010")
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
            parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')

            # --- CSCA General Config ---
            parser.add_argument('--csca_training_stage', type=str, default='full', choices=['mid_only', 'final_only', 'full'], help='CSCA training stage')
            parser.add_argument('--csca_mid_init_gate', type=float, default=0.05, help='initial gate value for mid attention')
            parser.add_argument('--csca_final_init_gate', type=float, default=0.1, help='initial gate value for final attention')
            parser.add_argument('--csca_reduction', type=int, default=4, help='channel reduction ratio for CSCA')
            parser.add_argument('--csca_l1_weight', type=float, default=1e-4, help='L1 regularization weight for CSCA gates')
            parser.add_argument('--lambda_csca', type=float, default=0.001, help='overall loss weight for CSCA regularization')

            # --- CoT Branch Config ---
            parser.add_argument('--csca_cot_k', type=int, default=3, help='kernel size for CoT attention')
            parser.add_argument('--csca_cot_heads', type=int, default=4, help='number of heads for CoT attention')
            parser.add_argument('--csca_cot_temp', type=float, default=0.8, help='temperature tau for CoT attention')
            parser.add_argument('--csca_cot_lambda', type=float, default=0.7, help='mixing weight lambda for CoT attention')

            # --- Coord2H Branch Config ---
            parser.add_argument('--coord_heads', type=int, default=4, help='number of heads for Coord2H attention')
            parser.add_argument('--coord_tau', type=float, default=1.0, help='temperature tau for Coord2H axis-softmax')
            parser.add_argument('--coord_kappa', type=float, default=1.0, help='scaling factor kappa for Coord2H linear amplification path')
            parser.add_argument('--coord_use_softmax', dest='coord_use_softmax', action='store_true', help='use axis-softmax and amplification path in Coord2H')
            parser.add_argument('--no_coord_use_softmax', dest='coord_use_softmax', action='store_false', help='use stable sigmoid path in Coord2H')
            parser.set_defaults(coord_use_softmax=True)

        return parser

    @staticmethod
    def _parse_csca_enable_pattern(pattern_str, n_blocks):
        if pattern_str == 'all':
            return set(range(n_blocks))
        elif pattern_str == 'none':
            return set()
        elif pattern_str == 'mid4':
            if n_blocks < 4:
                print(f"Warning: 'mid4' pattern requested for generator with only {n_blocks} blocks. Enabling all blocks.")
                return set(range(n_blocks))
            start_idx = (n_blocks - 4) // 2
            return set(range(start_idx, start_idx + 4))
        elif pattern_str.startswith('indices:'):
            indices_str = pattern_str[len('indices:'):]
            try:
                # Expect 1-based indices, convert to 0-based
                parsed_indices = set(int(i.strip()) - 1 for i in indices_str.split(','))
                # Validate indices are within bounds [0, n_blocks-1]
                if any(idx < 0 or idx >= n_blocks for idx in parsed_indices):
                    raise ValueError(f"Indices out of bounds. Expected 1-based indices between 1 and {n_blocks}.")
                return parsed_indices
            except ValueError as e:
                raise ValueError(f"Invalid indices pattern: {pattern_str}. {e}")
        elif pattern_str.startswith('mask:'):
            mask_str = pattern_str[len('mask:'):]
            if len(mask_str) != n_blocks:
                raise ValueError(f"Mask length ({len(mask_str)}) does not match n_blocks ({n_blocks}).")
            return set(i for i, char in enumerate(mask_str) if char == '1')
        else:
            raise ValueError(f"Unknown CSCA enable pattern: {pattern_str}")

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.csca_training_stage = getattr(opt, 'csca_training_stage', 'full')
        self.lambda_csca = getattr(opt, 'lambda_csca', 0.001)

        # Create and populate the CSCA configuration object
        from .CSCA import CSCAConfig
        self.csca_config = CSCAConfig()
        
        # General
        self.csca_config.mid_init_gate = getattr(opt, 'csca_mid_init_gate', 0.05)
        self.csca_config.final_init_gate = getattr(opt, 'csca_final_init_gate', 0.1)
        self.csca_config.reduction = getattr(opt, 'csca_reduction', 4)
        self.csca_config.l1_weight = getattr(opt, 'csca_l1_weight', 1e-4)
        
        # CoT Branch
        self.csca_config.cot_k = getattr(opt, 'csca_cot_k', 3)
        self.csca_config.cot_heads = getattr(opt, 'csca_cot_heads', 4)
        self.csca_config.cot_temperature = getattr(opt, 'csca_cot_temp', 0.8)
        self.csca_config.cot_lambda = getattr(opt, 'csca_cot_lambda', 0.7)
        
        # Coord2H Branch
        self.csca_config.coord_heads = getattr(opt, 'coord_heads', 4)
        self.csca_config.coord_use_softmax = getattr(opt, 'coord_use_softmax', True)
        self.csca_config.coord_tau = getattr(opt, 'coord_tau', 1.0)
        self.csca_config.coord_kappa = getattr(opt, 'coord_kappa', 1.0)

        self.loss_names = ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B', 'csca_reg']
        visual_names_A = ['real_A', 'fake_B', 'rec_A']
        visual_names_B = ['real_B', 'fake_A', 'rec_B']
        if self.isTrain and self.opt.lambda_identity > 0.0:
            visual_names_A.append('idt_A')
            visual_names_B.append('idt_B')

        self.visual_names = visual_names_A + visual_names_B
        
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
        else:
            self.model_names = ['G_A', 'G_B']

        self.netG_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        csca_config=self.csca_config, training_stage=self.csca_training_stage)
        self.netG_B = networks.define_G(opt.output_nc, opt.input_nc, opt.ngf, opt.netG, opt.norm,
                                        not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids,
                                        csca_config=self.csca_config, training_stage=self.csca_training_stage)

        # Handle DataParallel wrapper
        netG_A_module = self.netG_A.module if hasattr(self.netG_A, 'module') else self.netG_A
        netG_B_module = self.netG_B.module if hasattr(self.netG_B, 'module') else self.netG_B

        # Get actual n_blocks from the generator module
        n_blocks_actual = getattr(netG_A_module, 'n_blocks', None)
        
        # Only apply CSCA mask if generator has n_blocks (i.e., it's a ResNet-based generator)
        if n_blocks_actual is not None:
            # Parse CSCA enable pattern and apply mask to generators
            csca_enabled_indices = self._parse_csca_enable_pattern(opt.csca_enable_pattern, n_blocks_actual)
            
            if hasattr(netG_A_module, 'set_csca_mask'):
                netG_A_module.set_csca_mask(csca_enabled_indices)
            if hasattr(netG_B_module, 'set_csca_mask'):
                netG_B_module.set_csca_mask(csca_enabled_indices)

            # Log CSCA configuration for verification
            print(f"")
            print(f"--- CSCA Configuration Summary ---")
            print(f"Total ResNet blocks (n_blocks): {n_blocks_actual}")
            print(f"CLI CSCA enable pattern: '{opt.csca_enable_pattern}'")
            print(f"CSCA enabled blocks (0-based): {sorted(list(csca_enabled_indices))}")
            print(f"CSCA enabled blocks (1-based): {[idx + 1 for idx in sorted(list(csca_enabled_indices))]}")
            print(f"----------------------------------")
            print(f"")
        else:
            print(f"")
            print(f"--- CSCA Configuration Summary ---")
            print(f"Generator type '{opt.netG}' does not support CSCA block selection (no 'n_blocks' attribute).")
            print(f"CSCA mask application skipped.")
            print(f"----------------------------------")
            print(f"")

        if self.isTrain:
            self.is_sn_gan = getattr(opt, 'sn_gan', 1)
            self.is_wgan = getattr(opt, 'wgan', 0)
            self.with_gp = getattr(opt, 'with_gp', 0)
            self.lambda_gp = getattr(opt, 'lambda_gp', 10)
            use_sigmoid = getattr(opt, 'no_lsgan', False) and not self.is_wgan
            if self.is_sn_gan:
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, "spectral", use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, "spectral", use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
            else:
                self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)
                self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, opt.init_gain, self.gpu_ids)

            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan).to(self.device)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers = [self.optimizer_G, self.optimizer_D]

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device, non_blocking=True)
        self.real_B = input['B' if AtoB else 'A'].to(self.device, non_blocking=True)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.fake_B = self.netG_A(self.real_A)
        self.rec_A = self.netG_B(self.fake_B)
        self.fake_A = self.netG_B(self.real_B)
        self.rec_B = self.netG_A(self.fake_A)

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        if self.is_wgan:
            loss_D_real = -pred_real.mean()
        else:
            loss_D_real = self.criterionGAN(pred_real, True)

        # Fake (detach so D-only grads)
        pred_fake = netD(fake.detach())
        if self.is_wgan:
            loss_D_fake = pred_fake.mean()
        else:
            loss_D_fake = self.criterionGAN(pred_fake, False)

        # Combined
        loss_D = 0.5 * (loss_D_real + loss_D_fake)

        # WGAN-GP (only if wgan + gp)
        if self.is_wgan and self.with_gp:
            with torch.enable_grad():
                eps = torch.rand(real.size(0), 1, 1, 1, device=self.device)
                x_tilde = (eps * real + (1 - eps) * fake.detach()).requires_grad_(True)
                pred_tilde = netD(x_tilde)
                grad = torch.autograd.grad(
                    outputs=pred_tilde, inputs=x_tilde,
                    grad_outputs=torch.ones_like(pred_tilde),
                    create_graph=True, retain_graph=True, only_inputs=True
                )[0]
                gp = ((grad.view(grad.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
                loss_D = loss_D + self.lambda_gp * gp

        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)

    def backward_G(self):
        lambda_idt = self.opt.lambda_identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        
        if lambda_idt > 0:
            self.idt_A = self.netG_A(self.real_B)
            self.loss_idt_A = self.criterionIdt(self.idt_A, self.real_B) * lambda_B * lambda_idt
            self.idt_B = self.netG_B(self.real_A)
            self.loss_idt_B = self.criterionIdt(self.idt_B, self.real_A) * lambda_A * lambda_idt
        else:
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        if self.is_wgan:
            self.loss_G_A = -self.netD_A(self.fake_B).mean()
            self.loss_G_B = -self.netD_B(self.fake_A).mean()
        else:
            self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
            self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        
        def get_generator_csca_loss(net):
            if hasattr(net, 'module'):
                return net.module.get_csca_losses() if hasattr(net.module, 'get_csca_losses') else 0.0
            else:
                return net.get_csca_losses() if hasattr(net, 'get_csca_losses') else 0.0

        csca_loss_A = get_generator_csca_loss(self.netG_A)
        csca_loss_B = get_generator_csca_loss(self.netG_B)
        self.loss_csca_reg = (csca_loss_A + csca_loss_B) * self.lambda_csca

        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B + self.loss_idt_A + self.loss_idt_B + self.loss_csca_reg
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad([self.netD_A, self.netD_B], False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()
        self.backward_D_A()
        self.backward_D_B()
        self.optimizer_D.step()
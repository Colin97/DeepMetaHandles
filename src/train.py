import datasets
import torch
import numpy as np
import visdom
import network
from utils import weights_init, bn_momentum_adjust, weights_init1
import argparse
import shutil
import logging
from tqdm import tqdm
import os
from pathlib import Path
import datetime
from losses import laplacian_loss, normal_loss
import pytorch3d
from torch_batch_svd import svd
from render import render_mesh
from torch.autograd import Variable
from torchvision.utils import save_image

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--batch_size', type=int, default=42,
                        help='Batch Size during training')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='number of workers for data loader')
    parser.add_argument('--epoch',  default=30, type=int, help='Epoch to run')
    parser.add_argument('--learning_rate', default=0.0001,
                        type=float, help='Initial learning rate')
    parser.add_argument('--log_dir', type=str, default=None,
                        help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float,
                        default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--step_size', type=int,  default=5,
                        help='Decay step for lr decay')
    parser.add_argument('--lr_decay', type=float,
                        default=0.5, help='Decay rate for lr decay')
    parser.add_argument('--gpu', type=str, default='0,1,2', help='GPU to use') 
    parser.add_argument('--display', type=int, default=50,
                        help='number of iteration per display')
    parser.add_argument('--env_name', type=str,
                        default='train_chair', help='enviroment name of visdom')
    parser.add_argument('--num_basis', type=int, default=15,
                        help='number of basis vectors')
    return parser.parse_args()

opt = parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

experiment_dir = Path('./log/')
experiment_dir.mkdir(exist_ok=True)

if opt.log_dir is None:
    log_name = str(datetime.datetime.now().strftime(
        '%m-%d_%H-%M_')) + opt.env_name
    experiment_dir = experiment_dir.joinpath(log_name)
else:
    experiment_dir = experiment_dir.joinpath(opt.log_dir)
experiment_dir.mkdir(exist_ok=True)
checkpoints_dir = experiment_dir.joinpath('checkpoints/')
checkpoints_dir.mkdir(exist_ok=True)
log_dir = experiment_dir.joinpath('logs/')
log_dir.mkdir(exist_ok=True)
if opt.log_dir is None:
    shutil.copy('datasets.py', str(experiment_dir))
    shutil.copy('losses.py', str(experiment_dir))
    shutil.copy('network.py', str(experiment_dir))
    shutil.copy('pointnet_utils.py', str(experiment_dir))
    shutil.copy('train.py', str(experiment_dir))
    shutil.copy('utils.py', str(experiment_dir))
    shutil.copy('laplacian.py', str(experiment_dir))


def log_string(str):
    logger.info(str)
    print(str)


logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler = logging.FileHandler('%s/log.txt' % log_dir)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
log_string('PARAMETER ...')
log_string(opt)

vis = visdom.Visdom(port=8097, env=opt.env_name)

dataset_train = datasets.ChairDataset("train")
dataloader_train = torch.utils.data.DataLoader(dataset_train, collate_fn=datasets.my_collate, drop_last=True,
                                                shuffle=True, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                                worker_init_fn=lambda id: np.random.seed(np.random.get_state()[1][0] + id))

net = network.model(opt.num_basis).cuda()
net = torch.nn.DataParallel(net)

netD = network.Discriminator().cuda()  # discriminator network
netD = torch.nn.DataParallel(netD)

try:
    checkpoint = torch.load(str(experiment_dir) +
                            '/checkpoints/last_model.pth')
    start_epoch = checkpoint['epoch'] + 1
    cd_curve = checkpoint['cd_curve']
    sym_curve = checkpoint['sym_curve']
    nor_curve = checkpoint['nor_curve']
    lap_curve = checkpoint['lap_curve']
    tot_curve = []
    g_curve = []
    d_curve = []
    net.load_state_dict(checkpoint['model_state_dict'])
    netD.load_state_dict(checkpoint['modelD_state_dict'])
    log_string('Use pretrain model')
except:
    log_string('No existing model, starting training from scratch...')
    start_epoch = 0
    cd_curve = []
    sym_curve = []
    lap_curve = []
    nor_curve = []
    tot_curve = []
    g_curve = []
    d_curve = []
    net = net.apply(weights_init)
    netD = netD.apply(weights_init1)

optimizer = torch.optim.Adam(
    net.parameters(),
    lr=opt.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=opt.decay_rate)

optimizerD = torch.optim.Adam(
    netD.parameters(),
    lr=opt.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=opt.decay_rate * 10)

LEARNING_RATE_CLIP = 1e-7
MOMENTUM_ORIGINAL = 0.1
MOMENTUM_DECCAY = 0.5
MOMENTUM_DECCAY_STEP = opt.step_size

current_epoch = 0
adversarial_loss = torch.nn.BCELoss()

for epoch in range(start_epoch, opt.epoch):
    log_string('Epoch %d (%d/%s):' % (current_epoch + 1, epoch + 1, opt.epoch))
    '''Adjust learning rate and BN momentum'''
    lr = max(opt.learning_rate * (opt.lr_decay **
             (epoch // opt.step_size)), LEARNING_RATE_CLIP)
    log_string('Learning rate:%f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    for param_group in optimizerD.param_groups:
        param_group['lr'] = lr

    momentum = MOMENTUM_ORIGINAL * \
        (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
    if momentum < 0.01:
        momentum = 0.01
    print('BN momentum updated to: %f' % momentum)
    net = net.apply(lambda x: bn_momentum_adjust(x, momentum))
    netD = netD.apply(lambda x: bn_momentum_adjust(x, momentum))

    net.train()
    netD.train()
    buf = {"cd" : [], "sym" : [], "nor" : [], "lap" : [], "sp" : [], "cov" : [],
            "ortho" : [], "svd" : [], "dis" : [], "gen" : [], "real" : [], "fake" : [],
            "tot" : [], "acc" : []}

    src_names = []
    tar_names = []
    for i, data in tqdm(enumerate(dataloader_train), total=len(dataloader_train), smoothing=0.9):
        src_pc = data['src_pc']
        tar_pc = data['tar_pc']
        key_pts = data['key_pts']
        w_pc = data['w_pc']

        def_key_pts, def_pc, cd_loss, basis, coef, sample_def_key_pts, sym_loss, coef_range\
            = net(src_pc, tar_pc, key_pts, w_pc)

        vs = []
        fs = []
        def_vs = []
        real_vs = []
        real_fs = []
        for j in range(opt.batch_size):
            vs.append(data['src_ver'][j].cuda())
            fs.append(data['src_face'][j].cuda())
            real_vs.append(data['real_ver'][j].cuda())
            real_fs.append(data['real_face'][j].cuda())
            def_vs.append(torch.matmul(
                data['w_mesh'][j].cuda(), sample_def_key_pts[j]).contiguous())
        src_meshes = pytorch3d.structures.Meshes(verts=vs, faces=fs)
        def_meshes = pytorch3d.structures.Meshes(verts=def_vs, faces=fs)

        '''differentiable rendering & adversarial_loss'''
        images_real = render_mesh(real_vs, real_fs)
        images_fake = render_mesh(def_vs, fs)

        #save_image(images_real[0].permute(2, 0, 1), 'real.png')
        #save_image(images_fake[0].permute(2, 0, 1), 'fake.png')

        valid = Variable(torch.rand(opt.batch_size, 1).cuda()
                        * 0.1 + 0.9, requires_grad=False)
        fake = Variable(torch.rand(opt.batch_size, 1).cuda()
                        * 0.1, requires_grad=False)

        log_prob0 = netD(images_real.detach())
        log_prob1 = netD(images_fake.detach())

        real_loss = adversarial_loss(log_prob0, valid)
        fake_loss = adversarial_loss(log_prob1, fake)

        d_loss = (real_loss + fake_loss) / 2
        correct = (log_prob0 > 0.5).sum().item() + \
            (log_prob1 < 0.5).sum().item()

        optimizerD.zero_grad()
        real_loss.backward()
        fake_loss.backward()
        optimizerD.step()

        valid1 = Variable(torch.ones(
            opt.batch_size, 1).cuda(), requires_grad=False)
        log_prob2 = netD(images_fake)
        g_loss = adversarial_loss(log_prob2, valid1)

        buf['dis'].append(d_loss.item())
        buf['gen'].append(g_loss.item())
        buf['real'].append(real_loss.item())
        buf['fake'].append(fake_loss.item())
        buf['acc'].append(correct / (2 * opt.batch_size))

        '''fitting loss & geometric loss'''
        cd_loss = cd_loss.mean()
        sym_loss = sym_loss.mean()
        lap_loss = laplacian_loss(src_meshes, def_meshes)
        nor_loss = normal_loss(src_meshes, def_meshes)

        buf['cd'].append(cd_loss.cpu().item())
        buf['sym'].append(sym_loss.cpu().item())
        buf['nor'].append(nor_loss.cpu().item())
        buf['lap'].append(lap_loss.cpu().item())

        '''disentanglement loss'''
        dot = torch.bmm(basis.abs(), basis.transpose(1, 2).abs())
        dot[:, range(opt.num_basis), range(opt.num_basis)] = 0
        ortho_loss = dot.norm(p=2, dim=(1, 2)).mean()

        sp_loss = basis.view(opt.batch_size, opt.num_basis, 150).norm(p=1, dim=2).mean() \
            + coef.view(opt.batch_size, opt.num_basis).norm(p=1, dim=-1).mean()

        basis = basis.reshape(opt.batch_size * opt.num_basis, 50, 3)
        tmp = torch.bmm(basis.transpose(1, 2), basis)
        _, s, _ = svd(tmp)
        svd_loss = s[:, 2].mean()

        coef = coef.view(opt.batch_size, opt.num_basis) - \
            coef.view(opt.batch_size, opt.num_basis).mean(dim=0)
        cov = torch.bmm(coef.view(opt.batch_size, opt.num_basis, 1),
                        coef.view(opt.batch_size, 1, opt.num_basis))
        cov = cov.sum(dim=0) / (opt.batch_size - 1)
        cov_loss = cov.norm(p=1, dim=(0, 1))

        buf['sp'].append(sp_loss.cpu().item())
        buf['svd'].append(svd_loss.cpu().item())
        buf['ortho'].append(ortho_loss.cpu().item())
        buf['cov'].append(cov_loss.cpu().item())

        loss = cd_loss + sym_loss * 1 + lap_loss * 3 + nor_loss * 0.1 + g_loss * 0.006 \
            + svd_loss * 0.4 + sp_loss * 0.001 + ortho_loss * 0.001 + cov_loss * 0.001
        

        buf['tot'].append(loss.cpu().item())
        src_names += data['src_name']
        tar_names += data['tar_name']

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % opt.display == 0 and i != 0:
            cd_curve.append(np.mean(buf['cd']))
            sym_curve.append(np.mean(buf['sym']))
            nor_curve.append(np.mean(buf['nor']))
            lap_curve.append(np.mean(buf['lap']))
            tot_curve.append(np.mean(buf['tot']))

            log_string("cd: %f, sym: %f, nor: %f, lap: %f, sp: %f, svd %f, cov %f, ortho %f, tot %f" %
                        (np.mean(buf['cd']),
                        np.mean(buf['sym']),
                        np.mean(buf['nor']),
                        np.mean(buf['lap']),
                        np.mean(buf['sp']),
                        np.mean(buf['svd']),
                        np.mean(buf['cov']),
                        np.mean(buf['ortho']),
                        np.mean(buf['tot'])))
            
            g_curve.append(np.mean(buf['gen']))
            d_curve.append(np.mean(buf['dis']))
            log_string("d_loss: %f, g_loss: %f,  acc: %f, real: %f, fake: %f" %
                        (np.mean(buf['dis']),
                        np.mean(buf['gen']),
                        np.mean(buf['acc']),
                        np.mean(buf['real']),
                        np.mean(buf['fake'])))
            for b in buf:
                buf[b] = []
            src_names = []
            tar_names = []

            vis.scatter(X=src_pc.contiguous()[0].data.cpu(),
                        win='src_pc',
                        opts=dict(title="source shape", markersize=2))
            vis.scatter(X=tar_pc.contiguous()[0].data.cpu(),
                        win='tar_pc',
                        opts=dict(title="target shape", markersize=2))
            vis.scatter(X=def_pc.contiguous()[0].data.cpu(),
                        win='def_pc',
                        opts=dict(title="deformed shape", markersize=2))
            vis.line(X=np.arange(len(cd_curve)),
                    Y=np.array(cd_curve),
                    win='fit_loss',
                    opts=dict(title="fitting loss", markersize=2))
            vis.line(X=np.arange(len(sym_curve)),
                    Y=np.array(sym_curve),
                    win='symm_loss',
                    opts=dict(title="symmetry loss", markersize=2))
            vis.line(X=np.arange(len(lap_curve)),
                    Y=np.array(lap_curve),
                    win='lap_loss',
                    opts=dict(title="laplacian loss", markersize=2))
            vis.line(X=np.arange(len(nor_curve)),
                    Y=np.array(nor_curve),
                    win='nor_loss',
                    opts=dict(title="normal loss", markersize=2))
            vis.line(X=np.arange(len(tot_curve)),
                    Y=np.array(tot_curve),
                    win='tot_loss',
                    opts=dict(title="tot loss", markersize=2))
            vis.line(X=np.arange(len(g_curve)),
                    Y=np.array(g_curve),
                    win='gen_loss',
                    opts=dict(title="generator loss", markersize=2))
            vis.line(X=np.arange(len(d_curve)),
                    Y=np.array(d_curve),
                    win='dis_loss',
                    opts=dict(title="discriminator loss", markersize=2))

    savepath = str(checkpoints_dir) + '/last_model.pth'
    state = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'modelD_state_dict': netD.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
        'cd_curve': cd_curve,
        'sym_curve': sym_curve,
        'lap_curve': lap_curve,
        'nor_curve': nor_curve
    }
    torch.save(state, savepath)
    current_epoch += 1

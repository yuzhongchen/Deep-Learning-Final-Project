import argparse
import os
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from data_utils.trainDataLoader_two import trainDataLoader
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils.utils import test, save_checkpoint,show_point_cloud
from model.baseline import Baseline as Base
import provider
import numpy as np 
from loss import oriloss

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Baseline')
    parser.add_argument('--data_path', type=str, default='', help='Benchmark path')
    parser.add_argument('--batchsize', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch',  default=70, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.005, type=float, help='learning rate in training')
    parser.add_argument('--gpu', type=str, default='0,1,2,3,4,5,6,7', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
    parser.add_argument('--num_workers', type=int, default=8, help='Worker Number [default: 8]')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--pretrain', type=str, default=None,help='whether use pretrain model')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate of learning rate')
    parser.add_argument('--model_name', default='', help='model name')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
    return parser.parse_args()
def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    basepath=os.getcwd()
    '''CREATE DIR'''
    experiment_dir = Path(os.path.join(basepath,'experiment'))
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s_Baseline-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'train_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------TRANING---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = args.data_path

    TRAIN_DATASET = trainDataLoader(root=DATA_PATH,num=args.num_point)
    
    traindataloader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batchsize, shuffle=True, num_workers=args.num_workers)

    logger.info("The number of training data is: %d", len(TRAIN_DATASET))

    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier_grab = Base(args.batchsize).train()
    classifier_release = Base(args.batchsize).train()

    if torch.cuda.device_count() > 1:
        classifier_grab = torch.nn.DataParallel(classifier_grab)
        classifier_release = torch.nn.DataParallel(classifier_release)
    
    classifier_grab.to(device)
    classifier_release.to(device)

    if args.pretrain is not None:
        print('Use pretrain model...')
        logger.info('Use pretrain model')
        grab_path = args.pretrain+'_grab.pth'
        release_path = args.pretrain+'_release.pth'
        start_epoch = torch.load(grab_path)['epoch']
        classifier_grab.module.load_state_dict(torch.load(grab_path)['model_state_dict'])
        classifier_release.module.load_state_dict(torch.load(release_path)['model_state_dict'])
    else:
        print('No existing model, starting training from scratch...')
        start_epoch = 0


    if args.optimizer == 'SGD':
        optimizer_grab = torch.optim.SGD(classifier_grab.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        params = list(map(id, classifier_grab.module.gru.parameters()))

        trainable_params_grab = []
        trainable_params_grab += [{'params': filter(lambda x: x.requires_grad,
                                            classifier_grab.module.gru.parameters()),
                          'lr': args.learning_rate}]
        trainable_params_grab += [{'params': filter(lambda x:id(x) not in params,classifier.module.parameters()),
                          'lr': args.learning_rate}]
        optimizer_grab =  torch.optim.Adam(trainable_params_grab,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=args.decay_rate)

    scheduler_grab = torch.optim.lr_scheduler.StepLR(optimizer_grab, step_size=5, gamma=0.7)


    if args.optimizer == 'SGD':
        optimizer_release = torch.optim.SGD(classifier_release.parameters(), lr=args.learning_rate, momentum=0.9)
    elif args.optimizer == 'Adam':
        params = list(map(id, classifier_release.module.gru.parameters()))

        trainable_params_release = []
        trainable_params_release += [{'params': filter(lambda x: x.requires_grad,
                                            classifier_release.module.gru.parameters()),
                          'lr': args.learning_rate}]
        trainable_params_release += [{'params': filter(lambda x:id(x) not in params,classifier.module.parameters()),
                          'lr': args.learning_rate}]
        optimizer_release =  torch.optim.Adam(trainable_params_release,
                           betas=(0.9, 0.999),
                           eps=1e-08,
                           weight_decay=args.decay_rate)

    scheduler_release = torch.optim.lr_scheduler.StepLR(optimizer_release, step_size=5, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_tst_accuracy = 0.0
    blue = lambda x: '\033[94m' + x + '\033[0m'

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch,args.epoch):
        print('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        logger.info('Epoch %d (%d/%s):' ,global_epoch + 1, epoch + 1, args.epoch)
        print('lr=',optimizer_grab.state_dict()['param_groups'][0]['lr'])
        totalloss_grab = 0
        totalloss_release = 0
        scheduler_grab.step()
        scheduler_release.step()
        for batch_id, data in tqdm(enumerate(traindataloader, 0), total=len(traindataloader), smoothing=0.9):
            (gt_xyz_grab,pointcloud_grab,geometry_grab,LENGTH_grab,_),(gt_xyz_release,pointcloud_release,geometry_release,LENGTH_release,_)= data
            

            pointcloud_grab=pointcloud_grab.transpose(3,2)
            
            gt_xyz_grab,pointcloud_grab,geometry_grab=gt_xyz_grab.to(device),pointcloud_grab.to(device),geometry_grab.to(device)

            optimizer_grab.zero_grad()
            
            if gt_xyz_grab.size()[0]!=args.batchsize:
                break
            pred = classifier_grab(pointcloud_grab[:,:,:3,:],pointcloud_grab[:,:,3:,:],geometry_grab,LENGTH_grab.max().repeat(torch.cuda.device_count()).to(device))
            
            
            loss =oriloss(pred,gt_xyz_grab,LENGTH_grab,device)


            loss.backward()
            optimizer_grab.step()
            global_step += 1
            totalloss_grab=loss+totalloss_grab


            pointcloud_release=pointcloud_release.transpose(3,2)
            
            gt_xyz_release,pointcloud_release,geometry_release=gt_xyz_release.to(device),pointcloud_release.to(device),geometry_release.to(device)

            optimizer_release.zero_grad()
            
            if gt_xyz_release.size()[0]!=args.batchsize:
                break
            pred = classifier_release(pointcloud_release[:,:,:3,:],pointcloud_release[:,:,3:,:],geometry_release,LENGTH_release.max().repeat(torch.cuda.device_count()).to(device))
            
            
            loss =oriloss(pred,gt_xyz_release,LENGTH_release,device)


            loss.backward()
            optimizer_release.step()
            totalloss_release=loss+totalloss_release
            
            nnum=1
            if (batch_id+1)==nnum:
            
                print('\r Grab Loss: %f' % float(totalloss_grab/nnum))
                preloss_grab=totalloss_grab
                print('\r Release Loss: %f' % float(totalloss_release/nnum))
                preloss_release=totalloss_release
            elif (batch_id+1)%nnum==0:
                print('\r Grab Loss: %f' % float((totalloss_grab-preloss_grab)/nnum))
                preloss_grab=totalloss_grab
                print('\r Release Loss: %f' % float((totalloss_release-preloss_release)/nnum))
                preloss_release=totalloss_release
            else:
                None

        if (global_epoch+1)%1==0 :
            save_checkpoint(
                global_epoch + 1,
                classifier_grab.module,
                optimizer_grab,
                str(checkpoints_dir),
                args.model_name+"_grab")
            save_checkpoint(
                global_epoch + 1,
                classifier_release.module,
                optimizer_release,
                str(checkpoints_dir),
                args.model_name+"_release")
            print('Saving model....')

        print('\r Loss: %f' % loss.data)
        logger.info('Grab Loss: %.2f', totalloss_grab)
        logger.info('Release Loss: %.2f', totalloss_release)
        global_epoch += 1

    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)
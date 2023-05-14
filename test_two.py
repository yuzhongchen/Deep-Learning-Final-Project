import argparse
import os
import sys
import numpy as np 
import torch
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import cv2
from data_utils.testDataLoader_two import testDataLoader
import datetime
import logging
from pathlib import Path
from tqdm import tqdm
from utils.utils import test, save_checkpoint
from model.baseline import Baseline as Base
import provider
import open3d as o3d

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Baseline')
    parser.add_argument('--data_path', type=str, default='', help='Benchmark path')
    parser.add_argument('--batchsize', type=int, default=1, help='batch size')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint')
    parser.add_argument('--num_point', type=int, default=8192, help='Point Number [default: 8192]')
    parser.add_argument('--num_workers', type=int, default=0, help='Worker Number [default: 0]')
    parser.add_argument('--model_name', default='', help='model name')
    parser.add_argument('--normal', action='store_true', default=True, help='Whether to use normal information [default: False]')
    parser.add_argument('--length', action='store_true', default=1, help='Whether to use normal information [default: False]')
    parser.add_argument('--vis', action='store_true', default=0, help='Whether to use normal information [default: False]')

    return parser.parse_args()

def main(args):
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    result_path='./result/'
    '''CREATE DIR'''
    experiment_dir = Path('./eval_experiment/')
    experiment_dir.mkdir(exist_ok=True)
    file_dir = Path(str(experiment_dir) + '/%s_Baseline-'%args.model_name + str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')))
    file_dir.mkdir(exist_ok=True)
    checkpoints_dir = file_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    os.system('cp %s %s' % (args.checkpoint+'_grab.pth', checkpoints_dir))
    os.system('cp %s %s' % (args.checkpoint+'_release.pth', checkpoints_dir))
    log_dir = file_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger(args.model_name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(str(log_dir) + 'eval_%s_cls.txt'%args.model_name)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('---------------------------------------------------EVAL---------------------------------------------------')
    logger.info('PARAMETER ...')
    logger.info(args)

    '''DATA LOADING'''
    logger.info('Load dataset ...')
    DATA_PATH = args.data_path

    TEST_DATASET = testDataLoader(root=DATA_PATH,num=args.num_point)
    finaltestDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batchsize,shuffle=False)
    logger.info("The number of test data is: %d", len(TEST_DATASET))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    seed = 3
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    '''MODEL LOADING'''

    classifier_grab = Base(args.batchsize).to(device).eval()
    classifier_release = Base(args.batchsize).to(device).eval()

  
    print('Load CheckPoint...')
    logger.info('Load CheckPoint')
    #need to cancel comment!!!!!!!!!!!!!!!!!!!!
    checkpoint_grab = torch.load(args.checkpoint+'_grab.pth')
    checkpoint_release = torch.load(args.checkpoint+'_release.pth')
    classifier_grab.load_state_dict(checkpoint_grab['model_state_dict'])
    classifier_release.load_state_dict(checkpoint_release['model_state_dict'])


    '''EVAL'''
    logger.info('Start evaluating...')
    print('Start evaluating...')

    
    for batch_id, data in tqdm(enumerate(finaltestDataLoader, 0), total=len(finaltestDataLoader), smoothing=0.9):
           
        
            (gt_xyz_grab,pointcloud_grab,geometry_grab,LENGTH_grab,clipsource_grab),(gt_xyz_release,pointcloud_release,geometry_release,LENGTH_release,clipsource_release)= data
            
            pointcloud_grab=pointcloud_grab.transpose(3,2)
            pointcloud_release=pointcloud_release.transpose(3,2)
            
            gt_xyz_grab,pointcloud_grab,geometry_grab=gt_xyz_grab.to(device),pointcloud_grab.to(device),geometry_grab.to(device)
            gt_xyz_release,pointcloud_release,geometry_release=gt_xyz_release.to(device),pointcloud_release.to(device),geometry_release.to(device)
            
            tic=cv2.getTickCount()
            pred_grab = classifier_grab(pointcloud_grab[:,:,:3,:],pointcloud_grab[:,:,3:,:],geometry_grab,LENGTH_grab.max().repeat(torch.cuda.device_count()).to(device))
            pred_release = classifier_release(pointcloud_release[:,:,:3,:],pointcloud_release[:,:,3:,:],geometry_release,LENGTH_release.max().repeat(torch.cuda.device_count()).to(device))

            toc=cv2.getTickCount()-tic    
            toc /= cv2.getTickFrequency()
            print('speed:',(LENGTH_grab+LENGTH_release)/toc,'FPS')
            model_path_grab = os.path.join('./results', args.model_name, clipsource_grab[0][0],clipsource_grab[1][0])
            model_path_release = os.path.join('./results', args.model_name, clipsource_release[0][0],clipsource_release[1][0])
            if not os.path.isdir(model_path_grab):
                os.makedirs(model_path_grab)
            if not os.path.isdir(model_path_release):
                os.makedirs(model_path_release)
            result_path_grab=os.path.join(model_path_grab,clipsource_grab[2][0]+'-'+clipsource_grab[3][0]+'_grab.txt')
            result_path_release=os.path.join(model_path_release,clipsource_release[2][0]+'-'+clipsource_release[3][0]+'_rele.txt')
            gt_path_grab=os.path.join(model_path_grab,clipsource_grab[2][0]+'-'+clipsource_grab[3][0]+'_grab_gt.txt')
            gt_path_release=os.path.join(model_path_release,clipsource_release[2][0]+'-'+clipsource_release[3][0]+'_rele_gt.txt')
            np.savetxt(gt_path_grab,gt_xyz_grab[0][:len(pred_grab)].cpu().numpy())
            np.savetxt(gt_path_release,gt_xyz_release[0][:len(pred_release)].cpu().numpy())

            with open(result_path_grab, 'w') as f:
                for xx in pred_grab:
                    
                    def dcon(x):
                        resultlist=torch.linspace(-1,1,1024*5).cuda()
                        x=x/x.max()

                        x[torch.where(x<=0.5)]=0

                        return (x*resultlist).sum()/x.sum()
  

                    data=str(float(dcon(xx[0][0])))+','+str(float(dcon(xx[0][1])))+','+str(float(dcon(xx[0][2])))
                    f.write(data+'\n')
            with open(result_path_release, 'w') as f:
                for xx in pred_release:
                    
                    def dcon(x):
                        resultlist=torch.linspace(-1,1,1024*5).cuda()
                        x=x/x.max()

                        x[torch.where(x<=0.5)]=0

                        return (x*resultlist).sum()/x.sum()
  

                    data=str(float(dcon(xx[0][0])))+','+str(float(dcon(xx[0][1])))+','+str(float(dcon(xx[0][2])))
                    f.write(data+'\n')
          
                
            
            
    logger.info('End of evaluation...')

if __name__ == '__main__':
    args = parse_args()
    main(args)

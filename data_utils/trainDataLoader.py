import numpy as np
import warnings
import os
import open3d as o3d
import cv2
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
import re
from tqdm import tqdm
from functools import reduce
from math import floor
import mediapipe as mp

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


def tryint(s):                    
    try:
        return int(s)
    except ValueError:
        return s

def str2int(v_str):            
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]


class trainDataLoader(Dataset):
    def __init__(self, root,num,max_it):
        self.root = root
        self.scenepath=os.path.join(root,'sequences')
        self.gtpath=os.path.join(root,'annotrain')
        self.num=num
        self.numscene=os.listdir(self.gtpath)
        self.indexlist=[]
        self.cliplength=[]
        self.max_it=max_it

        for scene in self.numscene:
            
            recordpath=os.path.join(self.gtpath,scene)
            recordname=os.listdir(recordpath)
            recordname.sort(key=str2int)
            
            for txtnum in recordname:
                txtpath=os.path.join(recordpath,txtnum)
                f=open(txtpath)
                data=f.readlines()
                for linenum in range(len(data)):
                    line=data[linenum].strip('\n').split(',')
                    if len(line)==9:
                        for ran in range(2):
                            if ran==0:
                                self.indexlist.append([scene,txtnum[:-4],line[0],line[1],line[3:6]])
                                self.cliplength.append(int(int(line[1])-int(line[0])))
                            else:
                                self.indexlist.append([scene,txtnum[:-4],line[1],line[2],line[6:]])
                                self.cliplength.append(int(int(line[2])-int(line[1])))
                                
                    elif len(line)==13:
                        for ran in range(3):
                            if ran==0:
                                self.indexlist.append([scene,txtnum[:-4],line[0],line[1],line[4:7]])
                                self.cliplength.append(int(int(line[1])-int(line[0])))
                            elif ran==1:
                                self.indexlist.append([scene,txtnum[:-4],line[1],line[2],line[7:10]])
                                self.cliplength.append(int(int(line[2])-int(line[1])))
                            else:
                                self.indexlist.append([scene,txtnum[:-4],line[2],line[3],line[10:]])
                                self.cliplength.append(int(int(line[3])-int(line[2])))
                                
                    elif len(line)==5:
                        for ran in range(1):
                            self.indexlist.append([scene,txtnum[:-4],line[0],line[1],line[2:]])
                            self.cliplength.append(int(int(line[1])-int(line[0])))
                    else:
                        print('cliperror')
                f.close()

        self.indexoff=np.where((np.array(self.cliplength)<=25)==1)[0]      
        self.length=len(self.indexoff)
        self.maxclip=25
            
                    
 
    def __len__(self):
        return len(self.indexoff)
    def __getitem__(self, index):
        return self._get_item(index)

    def getallimudata(self,imupath,start,end):
        imudata={}
        f=open(imupath,'r')
        alldata=f.readlines()
        for line in alldata:
            data=line.strip('\n').split(',')
            if float(data[0])>0.03*(start-1) and float(data[0])<0.03*(end):
                imudata[data[0]]=np.array([float(data[1]),float(data[2]),float(data[3]),\
                    float(data[4]),float(data[5]),float(data[6])]) 




        return imudata

    def getimudata(self,imupath,start):
        imudata={}
        f=open(imupath,'r')
        alldata=f.readlines()
        for line in alldata:
            data=line.strip('\n').split(',')
            if float(data[0])>0.03*(start-1) and float(data[0])<0.03*(start):
                imudata[data[0]]=np.array([float(data[1]),float(data[2]),float(data[3]),\
                    float(data[4]),float(data[5]),float(data[6])]) 



        each=imudata[list(imudata.keys())[-1]]
        return each


    def _get_item(self, index):
         
        finalsource=self.indexlist[self.indexoff[index]]
        imupath=os.path.join(self.scenepath,finalsource[0],finalsource[1],'data.txt')
        newpointpath=os.path.join(self.scenepath,finalsource[0],finalsource[1],'pointcloud')
        videopath=os.path.join(self.scenepath,finalsource[0],finalsource[1],'rgb_video.mp4')

        mp_hands = mp.solutions.hands

        transfomationsourcepath=os.path.join(self.scenepath,finalsource[0],finalsource[1],'transformation','odometry')
        
        rangenum=int(finalsource[3])-int(finalsource[2])
        
        pointcloud=np.zeros((self.maxclip,self.num,6))
        pointdir=np.zeros((self.num,3))
        
        geometry=np.zeros((self.maxclip,18))  

        positions=np.zeros((self.maxclip,63)) 
        curr_pos=np.zeros((self.num,3)) 
        middle_hand=np.zeros((self.maxclip,3)) 
        
        gt_xyz=np.zeros((self.maxclip,3)) 
        gt_xyzs=np.zeros((self.maxclip,self.max_it,8)) 
        centers=np.zeros((self.maxclip,self.max_it,8,3)) 

        first=np.array([float(finalsource[4][0]),float(finalsource[4][1]),float(finalsource[4][2])])
        odometrylist=[]

        cap = cv2.VideoCapture(videopath)
        cap.set(cv2.CAP_PROP_POS_FRAMES,1+int(finalsource[2]))

        
        for idx in range(rangenum):

            point=o3d.io.read_point_cloud(os.path.join(newpointpath,str(idx+1+int(finalsource[2]))+'.ply'))
            pointxyz=np.asarray(point.points)
            pointcolor=np.asarray(point.colors)

            randomlist=np.random.choice(pointxyz.shape[0],size=(self.num))
            pointcloud[idx,:,:3]=pointxyz[randomlist]
            pointcloud[idx,:,3:]=pointcolor[randomlist]
            pointdir[:,0] = -pointcloud[idx,:,0]
            pointdir[:,1] = -pointcloud[idx,:,1]
            pointdir[:,2] = np.abs(pointcloud[idx,:,2])

            odometrylist.append(np.load(os.path.join(transfomationsourcepath,str(idx+int(finalsource[2]))+'.npy')))
            odometry=reduce(np.dot, odometrylist)
            with mp_hands.Hands(
                min_detection_confidence=0.5,
                max_num_hands=1,
                min_tracking_confidence=0.5) as hands:
                success, image = cap.read()
                if success:
                    results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
                    if results.multi_hand_landmarks is not None:
                        for landmarks in results.multi_hand_landmarks:
                            id = 0
                            for lm in landmarks.landmark:
                                u = lm.x
                                v = lm.y
                                positions1[idx,id] = lm.x
                                positions1[idx,id+1] = lm.y
                                positions1[idx,id+2] = lm.z
                                u = int(u*3840)
                                v = int(v*2160)
                                # curr_pos[:,0] = (u-1.94228662e+03)/1.80820276e+03
                                # curr_pos[:,1] = (v-1.12382178e+03)/1.80794556e+03
                                # print(curr_pos[0])
                                curr_pos[:,0] = (u-1.94228662e+03)/1.80820276e+03*np.abs(pointcloud[idx,:,2])
                                curr_pos[:,1] = (v-1.12382178e+03)/1.80794556e+03*np.abs(pointcloud[idx,:,2])
                                curr_pos[:,2] = np.abs(pointcloud[idx,:,2])
                                # curr_pos[:,0] = (u-1.94228662e+03)/1.80820276e+03*pointcloud[idx,:,2]
                                # curr_pos[:,1] = (v-1.12382178e+03)/1.80794556e+03*pointcloud[idx,:,2]
                                # curr_pos[:,2] = pointcloud[idx,:,2]
                                pos = np.argmin(np.linalg.norm(curr_pos-pointdir,axis=1))
                                z = np.abs(pointcloud[idx,pos,2])
                                x = (u-1.94228662e+03)*z/1.80820276e+03
                                y = (v-1.12382178e+03)*z/1.80794556e+03
                                positions[idx,id] = x
                                positions[idx,id+1] = y
                                positions[idx,id+2] = z
                                # print(pointcloud[idx,pos,0:3])
                                id+=3
                                if id==15:
                                    middle_hand[idx,:] = middle_hand[idx,:]+positions[idx,id-3:id]
                                # print(x,y,z)
                                # middle_hand[idx,0] = middle_hand[idx,0]+x/21
                                # middle_hand[idx,1] = middle_hand[idx,1]+y/21
                                # middle_hand[idx,2] = middle_hand[idx,2]+z/21
                    else:
                        if (idx!=0):
                            positions[idx,:] = positions[idx-1,:]
                            middle_hand[idx,:] = middle_hand[idx-1,:]                
                else:
                    print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                    continue

            if idx ==0:
                gt_xyz[idx,:]=first
            else:
                gt_xyz[idx,:]=np.dot(np.linalg.inv(odometry),np.array([first[0], first[1], first[2], 1]))[:3]
            
            transfomationsource=np.load(os.path.join(transfomationsourcepath,str(idx+int(finalsource[2]))+'.npy'))[:3].reshape(-1)
            imudata=self.getimudata(imupath,idx+int(finalsource[2])).reshape(-1)
            
            geometry[idx]=np.concatenate((transfomationsource,imudata),0) 
            for it in range(self.max_it):
                for dirc in range(8):
                    x=0
                    y=0
                    z=0
                    if gt_xyz[idx,0]>centers[idx,it,dirc,0]:
                        x=1
                    if gt_xyz[idx,1]>centers[idx,it,dirc,1]:
                        y=1
                    if gt_xyz[idx,2]>centers[idx,it,dirc,2]:
                        z=1
                    gt_xyzs[idx,it,dirc]=x*4+y*2+z*1
                    if it==0:
                        x=floor((dirc)/4)
                        z=(dirc)%2
                        y=floor((dirc-4*x-z)/2)
                    x=x*2-1
                    y=y*2-1
                    z=z*2-1
                    if(it<self.max_it-1):
                        centers[idx,it+1,dirc,:]=np.asarray([centers[idx,it,dirc,0]+x*0.1*0.6**(it),centers[idx,it,dirc,1]+y*0.1*0.6**(it),centers[idx,it,dirc,2]+z*0.3*0.6**(it)])
        return gt_xyzs,centers,pointcloud,geometry,rangenum,finalsource,positions



if __name__ == '__main__':
    import torch
    DATA_PATH = './Benchmark/'
    TRAIN_DATASET = RGBDDataLoader(root=DATA_PATH,num=8192)
    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=8, shuffle=True, num_workers=16)
    for batch_id, data in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):

        print(batch_id)



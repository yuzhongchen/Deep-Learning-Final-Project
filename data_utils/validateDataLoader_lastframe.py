import numpy as np
import warnings
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')
from functools import reduce
import h5py



class validateDataLoader_lastframe(Dataset):
    def __init__(self, root, num):  # root="/scratch/yw5458/EgoPAT3D/dataset.hdf5"
        self.root = root
        self.rgb_video_path = "/scratch/yw5458/EgoPAT3D/videos"
        self.num = num
        self.indexlist = []
        self.cliplength = []
        self.mode = "annovalidate"
        self.maxclip=0

        dataset_file = h5py.File(self.root, "r")
        gt_grp = dataset_file[self.mode]
        for scene_name in gt_grp:
            scene_grp = gt_grp[scene_name]
            for video_name in scene_grp:
                video = scene_grp[video_name]
                video = np.asarray(video)
                for line in video:
                    self.indexlist.append([scene_name, video_name, str(int(float(line[0]))), str(int(float(line[1]))), [str(line[3]),str(line[4]), str(line[5]) ] ])
                    # self.cliplength.append(int( int(line[1])-int(line[0]) ))
                    self.maxclip = max(self.maxclip, int( int(line[1])-int(line[0]) ))
                    self.indexlist.append([scene_name, video_name, str(int(float(line[1]))), str(int(float(line[2]))), [str(line[6]),str(line[7]), str(line[8]) ] ])
                    # self.cliplength.append(int( int(line[2])-int(line[1]) ))
                    self.maxclip = max(self.maxclip, int( int(line[2])-int(line[1]) ))
        dataset_file.close()
        
        # self.indexoff = np.where((np.array(self.cliplength)<=25)==1)[0]
        self.length = len(self.indexlist)
 
    def __len__(self):
        return len(self.indexlist)
    
    def __getitem__(self, index):
        # import cv2
        finalsource = self.indexlist[index]
        dataset_file = h5py.File(self.root, "r")
        video_path = f"sequences/{finalsource[0]}/{finalsource[1]}"
        video = dataset_file[f"{video_path}"]
        
        #
        pointcloud_grp = video[f"pointcloud"]
        
        odometry_grp = video[f"transformation/odometry"]
        
        imu_file = np.array(video[f"imu"])

        # pointcloud=np.zeros((self.maxclip,self.num,6))
        # geometry=np.zeros((self.maxclip,18))  
        gt_xyz=np.zeros((self.maxclip,3), dtype=np.float32) 

        first=np.array([float(finalsource[4][0]),float(finalsource[4][1]),float(finalsource[4][2])])
        odometrylist=[]
        rangenum=int(finalsource[3])-int(finalsource[2])

        for idx in range(rangenum):
            # pointxyz = np.array(pointcloud_grp[f"pointxyz{idx+1+int(finalsource[2])}"])
            # pointcolor = np.array(pointcloud_grp[f"pointcolor{idx+1+int(finalsource[2])}"])

            # randomlist=np.random.choice(pointxyz.shape[0],size=(self.num))      # [8192, ]
            # pointcloud[idx,:,:3] = pointxyz[randomlist]                           # [8192, 3]
            # pointcloud[idx,:,3:] = pointcolor[randomlist]                         # [8192, 3]
            
            # frame = idx+int(finalsource[2])
            

            if idx!=0:
                odometrylist.append(np.array(odometry_grp[f"odometry{idx+int(finalsource[2])}"]))
                odometry=reduce(np.dot, odometrylist)
            if idx==0:
                gt_xyz[idx,:]=first
            else:
                gt_xyz[idx,:]=np.dot(np.linalg.inv(odometry),np.array([first[0], first[1], first[2], 1]))[:3]
            # transformationsource = np.array(odometry_grp[f"odometry{idx+int(finalsource[2])}"])[:3].reshape(-1)
            
            # imudata=imu_file[0][1:].reshape(-1)
            # imu_sign = 0
            # for imu_data in imu_file:
            #     if int(imu_data[0])==frame:
            #         imudata = imu_data[1:].reshape(-1)
            #         imu_sign = 1
            #     else:
            #         if imu_sign == 1:
            #             break


            # geometry[idx]=np.concatenate((transformationsource,imudata),0)

        final_pointcloud=np.zeros((self.num,6), dtype=np.float32)
        '''
        say we want a clip of length 2 (frame 0, frame 1). fianlsource[2] = 0. rangenum = finalsource[3]-finalsource[2] = 1-0 = 1
        so if we want frame 1, we need rangenum+1+int(finalsource[2]) = 1+1+0 = 2, because pointcloud is 1-indexed
        '''
        final_pointxyz = np.array(pointcloud_grp[f"pointxyz{rangenum+int(finalsource[2])}"]) 
        final_pointcolor = np.array(pointcloud_grp[f"pointcolor{rangenum+int(finalsource[2])}"])
        
        dataset_file.close()
        randomlist = np.random.choice(final_pointxyz.shape[0],size=(self.num))        # [8192, ]
        final_pointcloud[:,:3] = final_pointxyz[randomlist]                     # [8192, 3]
        final_pointcloud[:,3:] = final_pointcolor[randomlist]

        return gt_xyz[rangenum-1, :], final_pointcloud
 
 
 
        



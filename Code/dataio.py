from utils import *
import numpy as np
import torch
import skimage 
from skimage.transform import resize
from skimage.io import imread, imsave
from skimage import data,img_as_float
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import os

class ScalarDataSet():
	def __init__(self,args):
		self.dataset = args.dataset
		self.batch_size = args.batch_size
		self.application = args.application
		self.interval = args.interval
		self.scale = args.scale
		self.factor = args.factor
		if self.dataset == 'MF':
			self.dim = [480,720,120]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/jet_mixfrac_'
		elif self.dataset == 'CHI':
			self.dim = [480,720,120]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/jet_chi_'
		elif self.dataset == 'HR':
			self.dim = [480,720,120]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/jet_hr_'                
		elif self.dataset == 'YOH':
			self.dim = [480,720,120]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/jet_Y_OH_'
		elif self.dataset == 'VORT':
			self.dim = [480,720,120]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/jet_vort_'                 	                  
		elif self.dataset == 'H':
			self.dim = [600,248,248]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/H-'
		elif self.dataset == 'H+':
			self.dim = [600,248,248]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/H+-'
		elif self.dataset == 'PD':
			self.dim = [600,248,248]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/PD-'
		elif self.dataset == 'T':
			self.dim = [600,248,248]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/GT-'
		elif self.dataset == 'H2':
			self.dim = [600,248,248]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/H2-'             	                 	                 
		elif self.dataset == 'Vortex':
			self.dim = [128,128,128]
			self.total_samples = 90
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/vorts'
		elif self.dataset == 'Jet':
			self.dim = [128,128,128]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/Jet'
		elif self.dataset == 'Earthquake':
			self.dim = [256,256,96]
			self.total_samples = 598
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/amp'
		elif self.dataset == '640':
			self.dim = [640,240,80]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/half-cylinder-640-'
		elif self.dataset == '6400':
			self.dim = [640,240,80]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/half-cylinder-6400-'
		elif self.dataset == '320':
			self.dim = [640,240,80]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/half-cylinder-320-'
		elif self.dataset == '160':
			self.dim = [640,240,80]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/half-cylinder-160-'
		elif self.dataset == 'tangaroa':
			self.dim = [300,180,120]
			self.total_samples = 150
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/tangaroa-'
		elif self.dataset == 'Tangaroa-M':
			self.dim = [300,180,120]
			self.total_samples = 150
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/tangaroa'
		elif self.dataset == 'supernova':
			self.dim = [256,256,256]
			self.total_samples = 60
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/Supernova_E'
		elif self.dataset == 'supercurrent':
			self.dim = [256,128,32]
			self.total_samples = 200
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/supercurrent-vorticity-'
		elif self.dataset == '160-M':
			self.dim = [640,240,80]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/half-cylinder-160-magnitude-'
		elif self.dataset == '320-M':
			self.dim = [640,240,80]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/half-cylinder-320-magnitude-'
		elif self.dataset == '640-M':
			self.dim = [640,240,80]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/half-cylinder-640-magnitude-'
		elif self.dataset == '6400-M':
			self.dim = [640,240,80]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/half-cylinder-6400-magnitude-'		
		elif self.dataset == 'Bubble':
			self.dim = [320,128,128]
			self.total_samples = 100
			self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/UA_TSSR_Data/Bubble-'

		if not os.path.exists(args.result_path+args.dataset):
			os.mkdir(args.result_path+args.dataset)

		if not os.path.exists(args.model_path+args.dataset):
			os.mkdir(args.model_path+args.dataset)

		if self.application == 'extrapolation':
			self.training_samples = self.total_samples*8//10
			self.samples = range(1,self.training_samples+1)
		elif self.application == 'temporal':
			if self.dataset != 'Earthquake':
				self.samples = [i for i in range(1,self.total_samples+1,self.interval+1)]
				self.total_samples = self.samples[-1]
			else:
				self.samples = np.fromfile('/afs/crc.nd.edu/user/j/jhan5/CoordNet/earthquake_select.dat',dtype='int16')
				self.samples[0] = 1
		elif self.application == 'spatial':
			self.samples = range(1,self.total_samples+1)
		elif self.application == 'super-spatial':
			if self.dataset != 'PD':
				self.samples = range(1,self.total_samples+1)
			else:
				self.samples = range(51,self.total_samples+1)
				self.total_samples = 50


	def GetCoords(self):
		if self.application == 'extrapolation':
			self.coords = get_mgrid([self.total_samples,self.dim[0],self.dim[1],self.dim[2]],dim=4,s=1,t=0)
			self.coords = self.coords[0:self.training_samples*self.dim[0]*self.dim[1]*self.dim[2]:,]
		elif self.application == "temporal":
			if self.dataset != 'Earthquake':
				self.coords = get_mgrid([self.total_samples,self.dim[0],self.dim[1],self.dim[2]],dim=4,s=1,t=self.interval)
			else:
				coords = []
				time = np.zeros((self.dim[0]*self.dim[1]*self.dim[2],1))
				self.coords = get_mgrid([self.dim[0],self.dim[1],self.dim[2]],dim=3)
				for t in self.samples:
					print(t)
					if self.dataset == 'Earthquake':
						t = (t-1)/(self.total_samples-1)
					else:
						t = t/(self.total_samples-1)
					t -= 0.5
					t *= 2
					time.fill(t)
					coords += list(np.concatenate((time,self.coords),axis=1))
				self.coords = np.asarray(coords)
				print(self.coords.shape)
		elif self.application == 'spatial':
			self.Subsample()
			self.coords = get_mgrid([self.total_samples,self.dim[0],self.dim[1],self.dim[2]],dim=4,s=self.scale,t=0)
		elif self.application == 'super-spatial':
			self.coords = get_mgrid([self.total_samples,self.dim[0]*self.scale,self.dim[1]*self.scale,self.dim[2]*self.scale],dim=4,s=self.scale,t=0)

	def ReadData(self):
		self.GetCoords()
		self.data = []
		for i in self.samples:
			print(i)
			if self.dataset!='Tangaroa-M':
				d = np.fromfile(self.data_path+'{:04d}'.format(i)+'.dat',dtype='<f')
			else:
				d = np.fromfile(self.data_path+'{:04d}'.format(i+50)+'.dat',dtype='<f')
			d = 2*(d-np.min(d))/(np.max(d)-np.min(d))-1
			if self.application == 'spatial':
				self.data += list(d[self.coords_indices])
			else:
				self.data += list(d)
		self.data = np.asarray(self.data)


	def Subsample(self):
		self.coords_indices = []
		for z in range(0,self.dim[2],self.scale):
			for y in range(0,self.dim[1],self.scale):
				for x in range(0,self.dim[0],self.scale):
					index = (((z) * self.dim[1] + y) * self.dim[0] + x)
					self.coords_indices.append(index)
		self.coords_indices = np.asarray(self.coords_indices)


	def GetTrainingData(self):
		indices = []
		if self.application == 'spatial':
			samples = (self.dim[0]*self.dim[1]*self.dim[2])//(self.scale*self.scale*self.scale)
		elif self.application == 'super-spatial':
			samples = (self.dim[0]*self.dim[1]*self.dim[2])
		else:
			samples = self.dim[0]*self.dim[1]*self.dim[2]


		if self.application == 'extrapolation':
			for i in range(0,self.training_samples):
				index = np.random.choice(np.arange(i*samples,(i+1)*samples), self.factor*self.batch_size, replace=False)
				indices += list(index)
		elif self.application == 'temporal':
			for i in range(0,len(self.samples)):
				index = np.random.choice(np.arange(i*samples,(i+1)*samples), self.factor*self.batch_size, replace=False)
				indices += list(index)
		elif self.application == 'super-spatial':
			for i in range(0,len(self.samples)):
				index = np.random.choice(np.arange(i*samples,(i+1)*samples), self.factor*self.batch_size, replace=False)
				indices += list(index)


		if self.application in ['temporal','completion','super-spatial','extrapolation']:
			training_data_input = torch.FloatTensor(self.coords[indices])
			training_data_output = torch.FloatTensor(self.data[indices])
		elif self.application == 'spatial':
			if self.factor*self.batch_size >= samples:
				training_data_input = torch.FloatTensor(self.coords)
				training_data_output = torch.FloatTensor(self.data)
			else:
				for i in range(0,len(self.samples)):
					index = np.random.randint(low=i*samples,high=(i+1)*samples,size=self.factor*self.batch_size)
					indices += list(index)
				training_data_input = torch.FloatTensor(self.coords[indices])
				training_data_output = torch.FloatTensor(self.data[indices])
		data = torch.utils.data.TensorDataset(training_data_input,training_data_output)
		train_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)
		return train_loader

	def GetTestingData(self):
		return get_mgrid([self.total_samples,self.dim[0],self.dim[1],self.dim[2]],dim=4)
		


class AODataSet():
	def __init__(self,args):
		self.dataset = args.dataset
		self.batch_size = args.batch_size
		self.interval = 7
		self.factor = args.factor            	                 	                 
		if self.dataset == 'Vortex':
			self.dim = [128,128,128]
			self.total_samples = 90
		elif self.dataset == '6400':
			self.dim = [640,240,80]
			self.total_samples = 100
		elif self.dataset == 'Tangaroa':
			self.dim = [300,180,120]
			self.total_samples = 150
		elif self.dataset == 'Bubble':
			self.dim = [320,128,128]
			self.total_samples = 100
		elif self.dataset == 'Earthquake':
			self.dim = [256,256,96]
			self.total_samples = 598
		self.data_path = '/afs/crc.nd.edu/user/j/jhan5/vis/VIS20/AO/'+args.dataset+'/'


		if not os.path.exists(args.result_path+args.dataset):
			os.mkdir(args.result_path+args.dataset)

		if not os.path.exists(args.model_path+args.dataset):
			os.mkdir(args.model_path+args.dataset)

		self.samples = [i for i in range(1,self.total_samples+1,self.interval+1)] + [self.total_samples]
		self.total_samples = self.samples[-1]


	def GetCoords(self):
		coords = []
		time = np.zeros((self.dim[0]*self.dim[1]*self.dim[2],1))
		self.coords = get_mgrid([self.dim[0],self.dim[1],self.dim[2]],dim=3)
		for t in self.samples:
			print(t)
			t = (t-1)/(self.total_samples-1)
			t -= 0.5
			t *= 2
			time.fill(t)
			coords += list(np.concatenate((time,self.coords),axis=1))
		self.coords = np.asarray(coords)
		print(self.coords.shape)

	def ReadData(self):
		self.GetCoords()
		self.data = []
		for i in self.samples:
			print(i)
			d = np.fromfile(self.data_path+'{:04d}'.format(i)+'.dat',dtype='<f')
			d = 2*(d-np.min(d))/(np.max(d)-np.min(d))-1
			self.data += list(d)
		self.data = np.asarray(self.data)


	def GetTrainingData(self):
		indices = []
		samples = self.dim[0]*self.dim[1]*self.dim[2]
		for i in range(0,len(self.samples)):
			index = np.random.choice(np.arange(i*samples,(i+1)*samples), self.factor*self.batch_size, replace=False)
			indices += list(index)

		training_data_input = torch.FloatTensor(self.coords[indices])
		training_data_output = torch.FloatTensor(self.data[indices])
		data = torch.utils.data.TensorDataset(training_data_input,training_data_output)
		train_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)
		return train_loader

class ViewSynthesis():
	def __init__(self,args):
		self.dataset = args.dataset
		self.batch_size = args.batch_size
		self.factor = args.factor
		self.res = args.res
		self.angle = args.angle
		self.path = '../CoordNet/Data/'+self.dataset+'/'+str(self.res)+'/'
		self.theta = [i for i in range(0,180,self.angle)]+[179]
		self.phi = [i for i in range(0,360,self.angle)]+[359]
		if not os.path.exists(args.result_path+args.dataset):
			os.mkdir(args.result_path+args.dataset)

		if not os.path.exists(args.model_path+args.dataset):
			os.mkdir(args.model_path+args.dataset)

	def ReadData(self):
		self.count = 0
		self.coords = get_mgrid([self.res,self.res],dim=2)
		theta = np.zeros((self.res*self.res,1))
		phi = np.zeros((self.res*self.res,1))
		self.R = []
		self.G = []
		self.B = []
		coords = []
		for t in self.theta:
			theta_ = t/179.0
			theta_ -= 0.5
			theta_ *= 2.0
			theta.fill(theta_)
			for p in self.phi:
				phi_ = p/359.0
				phi_ -= 0.5
				phi_ *= 2.0
				phi.fill(phi_)
				coords += list(np.concatenate((self.coords,theta,phi),axis=1))
				img = img_as_float(imread(self.path+'save_'+'{:05d}'.format(self.count)+'.png'))
				img = img.transpose(2,0,1)
				img -= 0.5
				img *= 2.0
				self.R += list(img[0].flatten('F'))
				self.G += list(img[1].flatten('F'))
				self.B += list(img[2].flatten('F'))
				self.count += 1
				print(self.count)
		self.pixels = np.asarray([self.R,self.G,self.B])
		self.pixels = np.transpose(self.pixels,(1,0))
		self.coords = np.asarray(coords)

	def GetTrainingData(self):
		samples = self.res*self.res
		indices = []
		for i in range(0,self.count):
			index = np.random.choice(np.arange(i*samples,(i+1)*samples), self.factor*self.batch_size, replace=False)
			indices += list(index)
		training_data_input = torch.FloatTensor(self.coords[indices])
		training_data_output = torch.FloatTensor(self.pixels[indices])
		
		data = torch.utils.data.TensorDataset(training_data_input,training_data_output)
		train_loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=True)
		return train_loader

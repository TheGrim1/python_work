import sys
import numpy as np
import pylab as pl
import os,datetime
import h5py
from scipy.optimize import curve_fit

"""
Flatfield class - SJL 20131103

For analysis of a flatfield according to C. Schleputz (2009)

TODO:
# make detector and beamline independent
# generally clean it up.

output - flatfield (ff)
         flatfield uncertainty (ff_unc)
"""

class Flatfield:
	'''
	Class to provide necessary tools to: 
	analyse a flatfield
	generate a flatfield mask
	incorporate error bars

	input: 
		monitor corrected data
	output:
		ff - flatfield correction array
		ff_unc - relative uncertainties

	'''
	def __init__(self, data, path, detector = 'mpx4',mask = None,tolerance = 80, auto = False, id=''):
		'''
		Generate necessary memory blocks
		'''
		self.data = np.float32(data)
		self.no_ims = self.data.shape[0]
		self.data_dev = np.zeros(self.data.shape)
		self.I_lims = [0.,0.]
		self.check_I_lims = True
		self.check_tolerance = True
		self.tot_mask = np.ones(self.data[0,:,:].shape)
		self.K = 0.0
		time = datetime.datetime.utcnow()
		self.ff_path = path+id+'ff_'+time.strftime("%Y%m%d-%H%M/")
		self.tolerance = tolerance
		self.detector=detector
		self.mask_det=mask

		try:
			os.mkdir(self.ff_path)
		except:
			print("directory already exists!! no protection from overwriting")

		self.auto=auto
		if self.auto==True:
			print('Automated Flatfield analysis')
			self.check_I_lims = False
			self.check_tolerance = False

	def set_data(self,data):
		self.data = data

	def get_data(self,data):
		return self.data

	def apply_monitor2data(self,monitor):
		self.monitor = monitor
		for ii in range(self.data.shape[0]):
		    self.data[ii,:,:]=self.data[ii,:,:]/(self.monitor[ii]/np.mean(self.monitor))

	def calc_ff_ID01(self,user_mask=None):
		'''
		calculate a flatfield - takes all necessary steps
		check for dead pixels/ hot pixels
		choose the counting limits
		choose the required counting rates
		choose the tolerance for pixel counting
		generate ff and ff_unc, 2D array for flatfield correction and relative uncertainties
		
		'''
		# catch dead pixels i.e counting zero or ludicrously hot pixels - skewing mean

		#if user_mask!=None:
		#    self.data[user_mask>0]=np.nan 


        # find the mean value of each pixel
		self.calc_I_bar_sigma()

		# find the dead or hot pixels
		#self.dead_pixel_mask = ((np.sum((self.data_dev)**2,axis=0))==0) | \
        #                        (np.isnan(self.I_bar)) | \
        #                        (np.isnan(np.sum(self.data_dev,axis=0))) | \
        #                        (self.I_bar<(np.median(np.round(self.I_bar))*0.1)) | \
        #                        (self.I_bar>(np.median(np.round(self.I_bar))*4))   
		
		self.dead_pixel_mask =  (np.isnan(self.I_bar)) | \
								(np.isnan(np.sum(self.data_dev,axis=0))) | \
								(self.I_bar>(np.median(np.round(self.I_bar))*100)) 
		
		# plot the integrated intensity in the detector as a function of image
		self.plot_int_det_cts(mask=self.dead_pixel_mask)
		
		# mask_1: take only those pixels whose count rate lies within the user defined min/max count rate
		self.set_I_min_max()
		print("set I min/max")
		self.mask_1 = ((self.I_bar>= self.I_lims[0]) & \
                        (self.I_bar <= self.I_lims[1])) & \
                        (self.dead_pixel_mask == False)

		self.plot_bad_pxl_mask(self.mask_1,id='1')
		print("plot bad pixel mask")
		# mask_2 based on acceptable counting rates
		self.make_mask_2()
		print("made mask 2")
		self.plot_bad_pxl_mask(self.mask_2,id='2')
        # mask_3 based on tolerance ~ 98%
		self.mask_3 = self.set_tolerance()
		print("set tolerance")
		self.plot_bad_pxl_mask(self.mask_3,id='3')

		self.final_mask([self.mask_3,self.mask_2,self.mask_1])
		self.tot_mask=self.dead_pixel_mask
		print("final mask")
		self.plot_bad_pxl_mask(self.tot_mask,id='final')

        # look at the standard deviation across the detector
		self.apply_mask2data(np.invert(self.tot_mask))
		self.plot_SD_image()
		print("plot SD image")
		self.gen_ff()
		print("generate ff")
		self.plot_ff()
		print("plot ff")
		self.plot_rnd_ff_im()
		print("plot rnd ff im")
		self.plot_worst_pixel()
		print("plot worst pixel")
		#self.get_params()

	def calc_ff_x04sa(self,):
		'''
		calculate a flatfield - takes all necessary steps
		check for dead pixels/ hot pixels
		choose the counting limits
		choose the required counting rates
		choose the tolerance for pixel counting
		generate ff and ff_unc, 2D array for flatfield correction and relative uncertainties
		
		'''
		self.calc_I_bar_sigma()
		# catch dead pixels i.e counting zero or ludicrously hot pixels - skewing mean
		self.dead_pixel_mask = ((np.sum((self.data_dev)**2,axis=0))==0) | (self.I_bar>(np.nanmean(self.I_bar)*4))
		self.plot_int_det_cts()

		# mask_1 based on min/max count rate
		self.set_I_min_max()
		self.mask_1 = ((self.I_bar>= self.I_lims[0]) & (self.I_bar <= self.I_lims[1])) & (self.dead_pixel_mask ==False)

		#peak = x[y.tolist().index(np.max(y))]
		#sigma_peak = np.sqrt(peak)
		#self.set_I_min_max()
		print("set I min/max")
		self.plot_bad_pxl_mask(self.mask_1)
		print("plot bad pixel mask")
		# mask_2 based  on acceptable counting rates
		self.make_mask_2()
		print("made mask 2")
		self.set_tolerance()
		print("set tolerance")
		self.final_mask([self.mask_3,self.mask_2,self.mask_1])
		print("final mask")
		self.plot_SD_image()
		print("plot SD image")
		self.gen_ff()
		print("generate ff")
		self.plot_ff()
		print("plot ff")
		self.plot_rnd_ff_im()
		print("plot rnd ff im")
		self.plot_worst_pixel()
		print("plot worst pixel")
		self.get_params()

	def calc_I_bar_sigma(self,):
		"""
		"""
		self.data[self.data<1]=np.nan
		self.I_bar = np.nanmean(self.data,axis=0)

		for i in np.arange(self.data.shape[0]):
			self.data_dev[i,:,:] = self.data[i,:,:]-self.I_bar

		self.I_sigma = np.sqrt(np.sum((self.data_dev)**2,axis=0)/(self.no_ims-1))
		return self.I_bar, self.I_sigma

	def plot_int_det_cts(self,mask):
		pl.figure(1)
		tmp = self.data.copy()
		for i in np.arange(tmp.shape[0]):
		     tmp[i,:,:][mask]=np.nan 
		data2plot = np.nansum(np.nansum(tmp,axis=1),axis=1)
		pl.plot(((data2plot-np.nanmean(data2plot))/np.nanmean(data2plot))*100)
		pl.xlabel('Image no.')
		pl.ylabel('Total Intensity Variation (%)')
		pl.savefig(self.ff_path+'stats_1_int_det_vs_im.pdf')
		print(self.ff_path+"stats_1_int_det_vs_im.pdf \n If this is not a horizontal line with a small deviation of less than a few percent you are in trouble")
		pl.clf()

	def apply_mask2data(self,mask):
		for i in np.arange(self.data.shape[0]):
		     self.data[i,:,:][mask]=np.nan 
          # recalc values
		self.calc_I_bar_sigma()

	def plot_hist_avg_ph_cts(self,save=True):
		pl.figure(1)
		pl.title('Avg. Photon counts over %i exposures'%self.data.shape[0])
		self.hist_n, self.hist_bins, self.hist_patches = pl.hist(self.data.flatten(), np.arange(0,np.nanmean(np.compress(self.I_bar.flatten()>0,self.I_bar))*3,20), normed=0, facecolor='green', alpha=0.5)

		# gaussian fit to sigma_n

		self.hist_bin_centres = (self.hist_bins[:-1] + self.hist_bins[1:])/2

		# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
		p0 = [self.hist_n.max(), self.hist_bin_centres[self.hist_n.tolist().index(self.hist_n.max())], np.sqrt(self.hist_bin_centres[self.hist_n.tolist().index(self.hist_n.max())])]

		self.hist_coeff, var_matrix = curve_fit(gauss, self.hist_bin_centres, self.hist_n, p0=p0)

		# Get the fitted curve
		self.hist_hist_fit = gauss(self.hist_bin_centres, *self.hist_coeff)

		pl.plot(self.hist_bin_centres, self.hist_n,'g',label='Measured data')
		pl.plot(self.hist_bin_centres, self.hist_hist_fit,'b', label='Fitted data')
		tex = 'Fit pars - Amp:%i ,\n               Pos:%.2f ,\n              SD:%.2f'%(self.hist_coeff[0],self.hist_coeff[1],self.hist_coeff[2])
		ax = pl.gca()
		ax.text(self.I_lims[1]+np.sqrt(self.hist_bin_centres[self.hist_n.tolist().index(self.hist_n.max())]), self.hist_n.max()/2,tex,fontsize=10)
		pl.legend()

		# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
		print('Fit Parameters:')
		print('Peak ph cts: = ', self.hist_coeff[0])
		print('Fitted mean = ', self.hist_coeff[1])
		print('Fitted standard deviation = ', self.hist_coeff[2])

		self.I_lims = [self.hist_coeff[1]-3*self.hist_coeff[2],self.hist_coeff[1]+3*self.hist_coeff[2]]

		pl.vlines(self.I_lims[0],self.hist_n.min(),self.hist_n.max()/2.,color='r')
		pl.vlines(self.I_lims[1],self.hist_n.min(),self.hist_n.max()/2.,color='r')
		pl.xlim([self.hist_coeff[1]-8*self.hist_coeff[2],self.hist_coeff[1]+8*self.hist_coeff[2]])
		pl.xlabel('Photon counts')
		pl.ylabel('Frequency')
		if save | self.auto:
			pl.savefig(self.ff_path+"stats_2_hist_ph_cts.pdf")
		else:
			#print "Close figure 1 to continue - note the I_min, Imax"
			pl.clf()
		pl.clf()

	def plot_bad_pxl_mask(self,mask,id='0'):
		pl.figure(1)
		pl.title("%i pixels excluded"%(np.prod(self.data[0,:,:].shape)-mask.sum()))
		pl.imshow(mask,cmap='spectral')
		pl.savefig(self.ff_path+"image-bad_pixel_mask_"+id+".pdf")
		pl.clf()

	def set_I_min_max(self,):
		self.plot_hist_avg_ph_cts(save=True)
		if ((self.I_lims[0]==0.0) & (self.I_lims[1]==0.0)):
				self.check_I_lims = True
				"Choose limits: "
		while self.check_I_lims:
			try:
				pl.figure(1)
				pl.plot(self.hist_bin_centres, self.hist_n,'g',label='Measured data')
				pl.plot(self.hist_bin_centres, self.hist_hist_fit,'b', label='Fitted data')
				tex = 'Fit pars - Amp:%i ,\n               Pos:%.2f ,\n              SD:%.2f'%(self.hist_coeff[0],self.hist_coeff[1],self.hist_coeff[2])
				ax = pl.gca()
				ax.text(self.I_lims[1]+np.sqrt(self.hist_bin_centres[self.hist_n.tolist().index(self.hist_n.max())]), self.hist_n.max()/2,tex,fontsize=10)
				pl.legend()
				pl.vlines(self.I_lims[0],self.hist_n.min(),self.hist_n.max()/2.,color='r')
				pl.vlines(self.I_lims[1],self.hist_n.min(),self.hist_n.max()/2.,color='r')
				pl.xlim([self.hist_coeff[1]-8*self.hist_coeff[2],self.hist_coeff[1]+8*self.hist_coeff[2]])
				pl.xlabel('Photon counts')
				pl.ylabel('Frequency')
				pl.hold()
				pl.show()
				q = raw_input("are you happy with I_min/I_max [y/n]\n")
				if q=="n":
					self.I_lims[0] = int(input('\nI_min: '))
					self.I_lims[1] = int(input('\nI_max: '))
					print self.I_lims
				else:
					self.check_I_lims = False
					pl.clf()
					self.plot_hist_avg_ph_cts(save=True)
					print("\n ... it seems you are ... ")
			except:
				print("please enter a number")
				#self.check_I_lims = False
				#print self.check_I_lims

	def make_mask_2(self):
		np.seterr(all='ignore') # removes the division by zero error
		sigma_n = np.where(self.mask_1,self.I_sigma/np.sqrt(self.I_bar),0)
        #sn_dist,bins = np.histogram(sigma_n,np.arange(0,1.5,0.01))
        #
		#sigma_n = np.where(sigma_n>0)[0]
		sn_dist,bins,patches = pl.hist(sigma_n.flatten(), np.arange(0,1.5,0.01), normed=True, facecolor='green', alpha=0.5)
		# gaussian fit to sigma_n

		bin_centres = (bins[:-1] + bins[1:])/2

		# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
		max_sn = sn_dist[1:].max()
		p0 = [max_sn, bin_centres[sn_dist.tolist().index(max_sn)], 0.1]


		coeff, var_matrix = curve_fit(gauss, bin_centres, sn_dist, p0=p0)

		# Get the fitted curve
		hist_fit = gauss(bin_centres, *coeff)

		pl.figure(1)
		pl.plot(bin_centres, sn_dist,label='Measured data')
		pl.plot(bin_centres, hist_fit, label='Fitted data')
		tex = 'Fit pars - Amp:%i ,\n               Pos:%.2f ,\n              SD:%.2f'%(coeff[0],coeff[1],coeff[2])
		ax = pl.gca()
		ax.text(np.sqrt(bin_centres[sn_dist.tolist().index(sn_dist.max())]), sn_dist.max()/2,tex,fontsize=10)
		pl.legend()

		# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
		print('Fitted mean = ', coeff[1])
		print('Fitted standard deviation = ', coeff[2])

		pl.savefig(self.ff_path+'stats_3_ct_rate_corr.pdf')
		pl.clf()
		# add this
		sn_peak = coeff[1]
		sn_sigma = coeff[2]
		self.K= sn_peak + 4*sn_sigma
		self.mask_2 = self.I_sigma<(np.sqrt(self.I_bar)*self.K)

	def set_tolerance(self,):
		while self.check_tolerance:
			try:
				self.tolerance = int(input('Choose Tolerance (98% standard): '))
				self.check_tolerance = False
			except:
				print("please enter a number")

		tot_mask = self.mask_1*self.mask_2
		tmp_data = ((((self.data>=self.I_lims[0])& (self.data<=self.I_lims[1])).sum(axis=0)/float(self.no_ims))*tot_mask*100)
		# i.e normalise for exposures and corrected for the masking of pixels
		n, bins, patches = pl.hist(tmp_data.flatten(), np.arange(1,101,0.5),cumulative=True, normed=0, facecolor='green', alpha=0.5)
		pl.clf()
		data = np.prod(tmp_data.shape)-(((tmp_data>0).sum())-n).flatten()[::-1]
		pl.figure(1)
		pl.title('Tolerance versus Bad Pixels')
		pl.plot(np.arange(99.5,0,-0.5),data,label='data')
		ax = pl.gca()
		ax.set_xlim(ax.get_xlim()[::-1])
		pl.xlim([100,80])
		pl.ylim([0,(np.prod(tmp_data.shape)-(tmp_data>0).sum())*3])
		pl.xlabel('Tolerance')
		pl.ylabel('No. Bad Pixels')
		pl.vlines(self.tolerance,0,(np.prod(tmp_data.shape)-(tmp_data>0).sum())*3,color='r',label = 'Tolerance %.2f'%self.tolerance)
		pl.legend(loc='best')
		pl.savefig(self.ff_path+'stats_4_tol_plot.pdf')
		pl.clf()

		pl.figure(1)
		pl.title('Dead pixels')
		offset = 92.5
		step=0.3
		for i in np.arange(1,6):
			for j in np.arange(1,6):
				pl.subplot(5,5,((i-1)*5)+j)
				#print i,',',j,',',((i-1)*5)+j
				tmp_tolerance = ((i-1)*5+j)*step+offset
				pl.axis('off')
				#pl.tight_layout()
				pl.title('%.2f'%tmp_tolerance+r'%',fontsize=8)
				pl.imshow((tmp_data>=tmp_tolerance),cmap='spectral')#*ff_data[1,:,:])
		pl.savefig(self.ff_path+'stats_4_tol_as_ims.pdf')
		pl.clf()

		return (tmp_data>=self.tolerance)

	def final_mask(self,list_masks):
		for i in list_masks:
			self.tot_mask *= i

		pl.figure(1)
		pl.imshow(self.tot_mask,cmap='spectral')
		pl.title('Final Mask - %i pixels excluded'%(np.prod(self.tot_mask.shape)-self.tot_mask.sum()))
		pl.colorbar()
		pl.savefig(self.ff_path+'image-ff_final_mask.pdf')
		pl.clf()

	def plot_SD_image(self,clim=[0,0.05]):
		# hist of SD error and image of SD error
		pl.figure(1)
		pl.subplot(221)
		tmp = self.I_sigma*self.tot_mask/(np.nansum((self.I_bar*self.tot_mask))/np.nansum(self.tot_mask))#self.I_sigma/self.I_bar.sum()*(np.prod(self.I_bar.shape)-(self.I_bar==0).sum()))
		mean_I_sigma = np.nanmean(tmp.flatten())
		n, bins, patches = pl.hist(tmp.flatten(), np.arange(0,np.sqrt(mean_I_sigma),np.sqrt(mean_I_sigma)*2/100), normed=0, facecolor='green', alpha=0.5)
		pl.title('normalised SD distribution')
		pl.xlabel('SD')
		pl.ylabel('Frequency')
		pl.subplot(222)
		pl.title('normalised SD map')
		pl.imshow(tmp,cmap='spectral')
		pl.colorbar()
		pl.clim(clim)
		pl.xlabel('pixel')
		pl.ylabel('pixel')
		#pl.clim(mean_I_sigma +np.sqrt(mean_I_sigma)*-3, mean_I_sigma +np.sqrt(mean_I_sigma)*3)
		#pl.colorbar()
		pl.savefig(self.ff_path+'stats_5_SD.pdf')
		pl.clf()

#	def gen_ff(self,):
#		self.I_bar_tot = (self.I_bar*self.tot_mask).sum()/self.tot_mask.sum()
#		self.ff = np.where(self.tot_mask,self.I_bar_tot.sum()/self.I_bar,0)
#		self.ff_unc = np.where(self.tot_mask,np.sqrt(1.0/self.data.shape[2])*self.I_sigma/self.I_bar,0)
	def gen_ff(self,):
        # find the scale factor to normalise the data
		self.I_bar_sf = np.nansum((self.I_bar*self.tot_mask))/np.nansum(self.tot_mask)
		self.ff = np.where(self.tot_mask,self.I_bar_sf/self.I_bar,0)
		self.ff_unc = np.where(self.tot_mask,np.sqrt(1.0/self.data.shape[2])*self.I_sigma/self.I_bar,0)

	def plot_ff(self,clim=[0.95,1.05],clim_unc=[0,0.025]):
		pl.figure(1)
		pl.subplot(221)
		pl.title('Flatfield')
		pl.imshow(self.ff,cmap='spectral')
		pl.clim(clim)
		pl.colorbar()
		pl.subplot(222)
		pl.title('Flatfield Uncertainty')
		pl.imshow(self.ff_unc,cmap='spectral')
		#pl.clim((np.compress(self.ff_unc.flatten()>0,self.ff_unc.flatten())).min(),(np.compress(self.ff_unc.flatten()>0,self.ff_unc.flatten())).max())
		pl.colorbar()
		pl.clim(clim_unc)
		pl.savefig(self.ff_path+"stats_6_ff.pdf")
		pl.clf()

	def plot_worst_pixel(self,):
		# worst pixel versus a typical pixel as a function of image number
		pl.figure(1)
		pl.subplot(211)
		x,y = np.where(self.I_sigma*self.tot_mask==np.nanmax(self.I_sigma*self.tot_mask))
		tmp_data1 = self.data[:,x,y]
		#tmp_data1 = self.data_dev[x[0],y[0]]**2

		n, bins, patches = pl.hist(tmp_data1.flatten(),np.arange(np.nanmin(tmp_data1),np.nanmax(tmp_data1),(np.nanmax(tmp_data1)-np.nanmin(tmp_data1))/10) , normed=0, facecolor='green', alpha=0.5)
		#pl.xlim(self.I_lims[0]-10,self.I_lims[1]+10)
		mn1=np.nanmean(tmp_data1)
		std1=np.nanstd(tmp_data1)
		pl.xlim(mn1-3*std1,mn1+3*std1)
		pl.ylabel('Frequency')
		pl.title('Worst Pixel')
		pl.subplot(212)

		a = find_nearest(self.I_sigma*self.tot_mask,np.nanmean(self.I_sigma*self.tot_mask))
		x,y = np.where(self.I_sigma*self.tot_mask==a)
		print x,y
		#tmp_data2 = (self.data_dev[x,y]/self.data[x,y])**2
		#tmp_data2 = self.I_sigma*self.tot_mask
		#tmp_data2 = self.data_dev[x,y]**2
		tmp_data2 = self.data[:,x[0],y[0]]

		n, bins, patches = pl.hist(tmp_data2.flatten(),np.arange(np.nanmin(tmp_data2),np.nanmax(tmp_data2),(np.nanmax(tmp_data2)-np.nanmin(tmp_data2))/10) , normed=0, facecolor='black', alpha=0.5)
		pl.title('Average Pixel')
		#pl.xlim(self.I_lims[0]-10,self.I_lims[1]+10)
		mn2=np.nanmean(tmp_data1)
		std2=np.nanstd(tmp_data1)
		pl.xlim(mn2-3*std2,mn2+3*std2)
		pl.xlabel('Photon counts')
		pl.ylabel('Frequency')
		pl.savefig(self.ff_path+'stats_7_avg_wPix_hist.pdf')
		pl.clf()

		pl.figure(1)
		pl.plot(tmp_data1.flatten(),'g', label = 'Worst Pixel')
		pl.plot(tmp_data2.flatten(),'k', label = 'Average Pixel')
		pl.ylim(self.I_lims[0],self.I_lims[1])
		pl.xlabel('Exposure Number')
		pl.ylabel('Photons counts')
		pl.legend(loc='best')
		pl.savefig(self.ff_path+'stats_7_avg_wPix_time.pdf')
		pl.clf()

	def plot_rnd_ff_im(self,n_std=1):
		# normalised mean flatfield image,
		# a randomly chosen flatfield. could include line profiles or not as you wish.
		pl.figure(1)

		pl.subplot(221)
		pl.title('Random image')
		rand_num = int(np.random.rand()*self.no_ims)
		tmp = self.data[rand_num,:,:]*self.tot_mask
		data2plot = tmp/np.nansum(tmp)*(np.nansum((tmp>0)))
		pl.imshow(data2plot)
		pl.clim([np.nanmean(data2plot)-np.nanstd(data2plot)*n_std,np.nanmean(data2plot)+np.nanstd(data2plot)*n_std])
		#pl.clim(-data2plot.max()+2,data2plot.max())
		pl.colorbar()

		pl.subplot(222)
		pl.title('corrected w/ ff corr')
		data2plot = data2plot*self.ff
		pl.imshow(data2plot)
		pl.clim([np.nanmean(data2plot)-np.nanstd(data2plot)*n_std,np.nanmean(data2plot)+np.nanstd(data2plot)*n_std])
		pl.colorbar()

		pl.subplot(223)
		pl.title('flatfield')
		data2plot = self.ff
		pl.imshow(data2plot,cmap='spectral')
		pl.clim([np.nanmean(data2plot)-np.nanstd(data2plot)*n_std,np.nanmean(data2plot)+np.nanstd(data2plot)*n_std])
		pl.colorbar()

		pl.subplot(224)
		pl.title('mean w/ ff corr')
		data2plot = self.I_bar*self.tot_mask/self.I_bar_sf
		pl.imshow(data2plot*self.ff,cmap='spectral')
		pl.clim([np.nanmean(data2plot)-np.nanstd(data2plot)*n_std,np.nanmean(data2plot)+np.nanstd(data2plot)*n_std])
		pl.colorbar()

		pl.savefig(self.ff_path+'image-ff_im_correction.pdf')
		pl.clf()

	def get_ff(self):
		'''
		return the flatfield and relative uncertainties
		'''
		return self.ff, self.ff_unc

	def get_params(self):
		file = open(self.ff_path+'params.txt','w')
		file.write(str(self.__dict__))
		file.close()


	def apply_ff(self,data):
		'''
		Apply the ff to a dataset
		'''
		# apply the flatfield correction to the real data, calculate absolute uncertainties
		corr_data = np.zeros(data.shape)
		abs_unc = np.zeros(data.shape)
		rel_unc = np.sqrt(np.copy(data))/data
		for i in np.arange(data.shape[2]):
			corr_data[i:,:] = data[i,:,:]*self.ff
			abs_unc[i,:,:] = (rel_unc[i,:,:]+self.ff_unc)*corr_data[i,:,:,i]
		return corr_data, abs_unc

	def make_ff_h5(self, fn='ff.h5', save_raw_ims=False):
		# check the file doesn't exist
		try:
			os.listdir(self.ff_path).index(fn)
			print("File exists")
			q=input("would you like to overwrite it? [y/n]")
			if q=='y':
				ff_h5 = h5py.File(self.ff_path+fn,'w') 
			else:
				new_fn = input("New filename: ")
				ff_h5 = h5py.File(self.ff_path+fn,'w')

		except ValueError:
			print("File doesn't exist: Creating file")
			ff_h5 = h5py.File(self.ff_path+fn,'w')
	
		# add data to the file
		ff_h5.create_group('/ff')
		if save_raw_ims:
			ff_h5.create_dataset('ff/image_data', data = self.data,compression='gzip', compression_opts=9)
		ff_h5.create_dataset('ff/ff', data = self.ff,compression='gzip', compression_opts=9)
		ff_h5.create_dataset('ff/ff_rel_unc', data = self.ff_unc,compression='gzip', compression_opts=9)
		ff_h5.create_dataset('ff/bad_pix_mask', data = self.tot_mask,compression='gzip', compression_opts=9)
		for attr in list(self.__dict__.keys()):
			try:
				code ="ff_h5['ff/ff'].attrs['%s'] = self.%s"%(attr,attr)
				exec(code)
				print(code)
			except:
				pass
		ff_h5.close()

	def read_ff_h5(self, fn):

		try:
			ff_h5 = h5py.File(self.ff_path+fn,'r')
		except:
			print("File does not exist: ", self.ff_path)
			print(os.listdir(self.ff_path))

		ff = ff_h5['ff/ff'].value
		ff_unc = ff_h5['ff/ff_rel_unc'].value
		ff_attrs = ff_h5['ff/ff'].attrs
		#print("FF Attributes: ", end=' ') 
		for attr in list(ff_attrs.keys()):
			print(attr) #, ' : ', ff_attrs[attr]
		ff_h5.close()
		return ff,ff_unc

class Flatfieldv2:
	'''
	Class to provide necessary tools to: 
	analyse a flatfield
	generate a flatfield mask
	incorporate error bars

	input: 
		monitor corrected data
	output:
		ff - flatfield correction array
		ff_unc - relative uncertainties

	'''
	def __init__(self, data, path, detector = 'mpx4',mask = None,tolerance = 80, auto = False, id=''):
		'''
		Generate necessary memory blocks
		'''
		self.data = np.float32(data)
		self.no_ims = self.data.shape[0]
		self.data_dev = np.zeros(self.data.shape)
		self.I_lims = [0.,0.]
		self.check_I_lims = True
		self.check_tolerance = True
		self.tot_mask = np.ones(self.data[0,:,:].shape)
		self.K = 0.0
		time = datetime.datetime.utcnow()
		self.ff_path = path+id+'ff_'+time.strftime("%Y%m%d-%H%M/")
		self.tolerance = tolerance
		self.detector=detector
		self.mask_det=mask

		try:
			os.mkdir(self.ff_path)
		except:
			print("directory already exists!! no protection from overwriting")

		self.auto=auto
		if self.auto==True:
			print('Automated Flatfield analysis')
			self.check_I_lims = False
			self.check_tolerance = False

	def set_data(self,data):
		self.data = data

	def get_data(self,data):
		return self.data

	def apply_monitor2data(self,monitor):
		self.monitor = monitor
		for ii in range(self.data.shape[0]):
		    self.data[ii,:,:]=self.data[ii,:,:]/(self.monitor[ii]/np.mean(self.monitor))

	def calc_ff_ID01(self,user_mask=None):
		'''
		calculate a flatfield - takes all necessary steps
		check for dead pixels/ hot pixels
		choose the counting limits
		choose the required counting rates
		choose the tolerance for pixel counting
		generate ff and ff_unc, 2D array for flatfield correction and relative uncertainties
		
		'''
		# catch dead pixels i.e counting zero or ludicrously hot pixels - skewing mean

		#if user_mask!=None:
		#    self.data[user_mask>0]=np.nan 


        # find the mean value of each pixel
		self.calc_I_bar_sigma()

		# find the dead or hot pixels
		#self.dead_pixel_mask = ((np.sum((self.data_dev)**2,axis=0))==0) | \
        #                        (np.isnan(self.I_bar)) | \
        #                        (np.isnan(np.sum(self.data_dev,axis=0))) | \
        #                        (self.I_bar<(np.median(np.round(self.I_bar))*0.1)) | \
        #                        (self.I_bar>(np.median(np.round(self.I_bar))*4))   
		
		self.dead_pixel_mask =  (np.isnan(self.I_bar)) | \
								(np.isnan(np.sum(self.data_dev,axis=0))) | \
								(self.I_bar>(np.median(np.round(self.I_bar))*100)) 
		
		# plot the integrated intensity in the detector as a function of image
		self.plot_int_det_cts(mask=self.dead_pixel_mask)
		
		# mask_1: take only those pixels whose count rate lies within the user defined min/max count rate
		self.set_I_min_max()
		print("set I min/max")
		self.mask_1 = ((self.I_bar>= self.I_lims[0]) & \
                        (self.I_bar <= self.I_lims[1])) & \
                        (self.dead_pixel_mask == False)

		self.plot_bad_pxl_mask(self.mask_1,id='1')
		print("plot bad pixel mask")
		# mask_2 based on acceptable counting rates
		self.make_mask_2()
		print("made mask 2")
		self.plot_bad_pxl_mask(self.mask_2,id='2')
        # mask_3 based on tolerance ~ 98%
		self.mask_3 = self.set_tolerance()
		print("set tolerance")
		self.plot_bad_pxl_mask(self.mask_3,id='3')

		self.final_mask([self.mask_3,self.mask_2,self.mask_1])
		self.tot_mask=self.dead_pixel_mask
		print("final mask")
		self.plot_bad_pxl_mask(self.tot_mask,id='final')

        # look at the standard deviation across the detector
		self.apply_mask2data(np.invert(self.tot_mask))
		self.plot_SD_image()
		print("plot SD image")
		self.gen_ff()
		print("generate ff")
		self.plot_ff()
		print("plot ff")
		self.plot_rnd_ff_im()
		print("plot rnd ff im")
		self.plot_worst_pixel()
		print("plot worst pixel")
		#self.get_params()

	def deadpixels2nan(self):
		self.data[self.data==0]=np.nan


	def calc_I_bar_sigma(self,):
		"""
		"""
		self.deadpixels2nan()
		self.I_bar = np.nanmean(self.data,axis=0)

		for i in np.arange(self.data.shape[0]):
			self.data_dev[i,:,:] = self.data[i,:,:]-self.I_bar

		self.I_sigma = np.sqrt(np.sum((self.data_dev)**2,axis=0)/(self.no_ims-1))
		return self.I_bar, self.I_sigma

	def plot_int_det_cts(self,mask):
		pl.figure(1)
		tmp = self.data.copy()
		for i in np.arange(tmp.shape[0]):
		     tmp[i,:,:][mask==False]=np.nan 
		data2plot = np.nansum(np.nansum(tmp,axis=1),axis=1)
		pl.plot(((data2plot-np.nanmean(data2plot))/np.nanmean(data2plot))*100)
		pl.xlabel('Image no.')
		pl.ylabel('Total Intensity Variation (%)')
		pl.savefig(self.ff_path+'stats_1_int_det_vs_im.pdf')
		print(self.ff_path+"stats_1_int_det_vs_im.pdf \n If this is not a horizontal line with a small deviation of less than a few percent you are in trouble")
		pl.clf()

	def scale_data(self,mask):
		pl.figure(1)
		tmp = self.data.copy()
		for i in np.arange(tmp.shape[0]):
		     tmp[i,:,:][mask==False]=np.nan 
		data2plot = np.nansum(np.nansum(tmp,axis=1),axis=1)
		self.sf=data2plot/np.nanmean(data2plot)
		
		pl.plot(((data2plot-np.nanmean(data2plot))/np.nanmean(data2plot))/self.sf*100)
		self.data=self.data/self.sf[:,None,None]
		pl.xlabel('Image no.')
		pl.ylabel('Total Intensity Variation (%)')
		pl.savefig(self.ff_path+'stats_1_int_det_vs_im.pdf')
		print(self.ff_path+"stats_1_int_det_vs_im.pdf \n If this is not a horizontal line with a small deviation of less than a few percent you are in trouble")
		pl.clf()
		
	def apply_mask2data(self,mask):
		for i in np.arange(self.data.shape[0]):
		     self.data[i,:,:][mask]=np.nan 
          # recalc values
		self.calc_I_bar_sigma()
		
	def normalise(self):
		"""
		"""
		self.data = self.data/self.hist_coeff[1]
		self.calc_I_bar_sigma()


	def plot_hist_avg_ph_cts(self,save=True):
		pl.figure(1)
		pl.title('Avg. Photon counts over %i exposures'%self.data.shape[0])
		mean=np.nanmean(np.compress(self.I_bar.flatten()>0,self.I_bar))
		sigma=np.nanmean(np.compress(self.I_bar.flatten()>0,self.I_sigma))*3
		#self.hist_n, self.hist_bins, self.hist_patches = pl.hist(self.data.flatten()[np.isfinite(self.data.flatten())], np.arange(0,np.nanmean(np.compress(self.I_bar.flatten()>0,self.I_bar))*3,np.nanmean(np.compress(self.I_bar.flatten()>0,self.I_bar))*3/100), normed=0,facecolor='green', alpha=0.5)
		self.hist_n, self.hist_bins, self.hist_patches = pl.hist(self.data.flatten()[np.isfinite(self.data.flatten())], np.arange(mean-sigma*5,mean+sigma*5,sigma*10/50), normed=0,facecolor='green', alpha=0.5)

		# gaussian fit to sigma_n

		self.hist_bin_centres = (self.hist_bins[:-1] + self.hist_bins[1:])/2

		# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
		p0 = [self.hist_n.max(), self.hist_bin_centres[self.hist_n.tolist().index(self.hist_n.max())], np.sqrt(self.hist_bin_centres[self.hist_n.tolist().index(self.hist_n.max())])]

		self.hist_coeff, var_matrix = curve_fit(gauss, self.hist_bin_centres, self.hist_n, p0=p0)

		# Get the fitted curve
		self.hist_hist_fit = gauss(self.hist_bin_centres, *self.hist_coeff)

		pl.plot(self.hist_bin_centres, self.hist_n,'g',label='Measured data')
		pl.plot(self.hist_bin_centres, self.hist_hist_fit,'b', label='Fitted data')
		tex = 'Fit pars - Amp:%i ,\n               Pos:%.2f ,\n              SD:%.2f'%(self.hist_coeff[0],self.hist_coeff[1],self.hist_coeff[2])
		ax = pl.gca()
		ax.text(self.I_lims[1]+np.sqrt(self.hist_bin_centres[self.hist_n.tolist().index(self.hist_n.max())]), self.hist_n.max()/2,tex,fontsize=10)
		pl.legend()

		# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
		print('Fit Parameters:')
		print('Peak ph cts: = ', self.hist_coeff[0])
		print('Fitted mean = ', self.hist_coeff[1])
		print('Fitted standard deviation = ', self.hist_coeff[2])

		self.I_lims = [self.hist_coeff[1]-3*self.hist_coeff[2],self.hist_coeff[1]+3*self.hist_coeff[2]]

		pl.vlines(self.I_lims[0],self.hist_n.min(),self.hist_n.max()/2.,color='r')
		pl.vlines(self.I_lims[1],self.hist_n.min(),self.hist_n.max()/2.,color='r')
		pl.xlim([self.hist_coeff[1]-8*self.hist_coeff[2],self.hist_coeff[1]+8*self.hist_coeff[2]])
		pl.xlabel('Photon counts')
		pl.ylabel('Frequency')
		if save | self.auto:
			pl.savefig(self.ff_path+"stats_2_hist_ph_cts.pdf")
		else:
			#print "Close figure 1 to continue - note the I_min, Imax"
			pl.clf()
		pl.clf()

	def plot_bad_pxl_mask(self,mask,id='0'):
		pl.figure(1)
		pl.title("%i pixels excluded"%(np.prod(self.data[0,:,:].shape)-mask.sum()))
		pl.imshow(mask,cmap='spectral')
		pl.savefig(self.ff_path+"image-bad_pixel_mask_"+id+".pdf")
		pl.clf()

	def set_I_min_max(self,):
		self.plot_hist_avg_ph_cts(save=True)
		if ((self.I_lims[0]==0.0) & (self.I_lims[1]==0.0)):
				self.check_I_lims = True
				"Choose limits: "
		while self.check_I_lims:
			try:
				pl.figure(1)
				pl.plot(self.hist_bin_centres, self.hist_n,'g',label='Measured data')
				pl.plot(self.hist_bin_centres, self.hist_hist_fit,'b', label='Fitted data')
				tex = 'Fit pars - Amp:%i ,\n               Pos:%.2f ,\n              SD:%.2f'%(self.hist_coeff[0],self.hist_coeff[1],self.hist_coeff[2])
				ax = pl.gca()
				ax.text(self.I_lims[1]+np.sqrt(self.hist_bin_centres[self.hist_n.tolist().index(self.hist_n.max())]), self.hist_n.max()/2,tex,fontsize=10)
				pl.legend()
				pl.vlines(self.I_lims[0],self.hist_n.min(),self.hist_n.max()/2.,color='r')
				pl.vlines(self.I_lims[1],self.hist_n.min(),self.hist_n.max()/2.,color='r')
				pl.xlim([self.hist_coeff[1]-8*self.hist_coeff[2],self.hist_coeff[1]+8*self.hist_coeff[2]])
				pl.xlabel('Photon counts')
				pl.ylabel('Frequency')
				pl.hold()
				pl.show()
				q = raw_input("are you happy with I_min/I_max [y/n]\n")
				if q=="n":
					self.I_lims[0] = int(input('\nI_min: '))
					self.I_lims[1] = int(input('\nI_max: '))
					print self.I_lims
				else:
					self.check_I_lims = False
					pl.clf()
					self.plot_hist_avg_ph_cts(save=True)
					print("\n ... it seems you are ... ")
			except:
				print("please enter a number")
				#self.check_I_lims = False
				#print self.check_I_lims

	def make_mask_2(self):
		np.seterr(all='ignore') # removes the division by zero error
		sigma_n = np.where(self.mask_1,self.I_sigma/np.sqrt(self.I_bar),0)
        #sn_dist,bins = np.histogram(sigma_n,np.arange(0,1.5,0.01))
        #
		#sigma_n = np.where(sigma_n>0)[0]
		sn_dist,bins,patches = pl.hist(sigma_n.flatten()[np.isfinite(sigma_n.flatten())], np.arange(0,1.5,0.01), normed=True, facecolor='green', alpha=0.5)
		# gaussian fit to sigma_n

		bin_centres = (bins[:-1] + bins[1:])/2

		# p0 is the initial guess for the fitting coefficients (A, mu and sigma above)
		max_sn = sn_dist[1:].max()
		p0 = [max_sn, bin_centres[sn_dist.tolist().index(max_sn)], 0.1]


		coeff, var_matrix = curve_fit(gauss, bin_centres, sn_dist, p0=p0)

		# Get the fitted curve
		hist_fit = gauss(bin_centres, *coeff)

		pl.figure(1)
		pl.plot(bin_centres, sn_dist,label='Measured data')
		pl.plot(bin_centres, hist_fit, label='Fitted data')
		tex = 'Fit pars - Amp:%i ,\n               Pos:%.2f ,\n              SD:%.2f'%(coeff[0],coeff[1],coeff[2])
		ax = pl.gca()
		ax.text(np.sqrt(bin_centres[sn_dist.tolist().index(sn_dist.max())]), sn_dist.max()/2,tex,fontsize=10)
		pl.legend()

		# Finally, lets get the fitting parameters, i.e. the mean and standard deviation:
		print('Fitted mean = ', coeff[1])
		print('Fitted standard deviation = ', coeff[2])

		pl.savefig(self.ff_path+'stats_3_ct_rate_corr.pdf')
		pl.clf()
		# add this
		sn_peak = coeff[1]
		sn_sigma = coeff[2]
		self.K= sn_peak + 4*sn_sigma
		self.mask_2 = self.I_sigma<(np.sqrt(self.I_bar)*self.K)

	def set_tolerance(self,):
		while self.check_tolerance:
			try:
				self.tolerance = int(input('Choose Tolerance (98% standard): '))
				self.check_tolerance = False
			except:
				print("please enter a number")

		tot_mask = self.mask_1*self.mask_2
		tmp_data = ((((self.data>=self.I_lims[0])& (self.data<=self.I_lims[1])).sum(axis=0)/float(self.no_ims))*tot_mask*100)
		# i.e normalise for exposures and corrected for the masking of pixels
		n, bins, patches = pl.hist(tmp_data.flatten()[np.isfinite(tmp_data.flatten())], np.arange(1,101,0.5),cumulative=True, normed=0, facecolor='green', alpha=0.5)
		pl.clf()
		data = np.prod(tmp_data.shape)-(((tmp_data>0).sum())-n).flatten()[::-1]
		pl.figure(1)
		pl.title('Tolerance versus Bad Pixels')
		pl.plot(np.arange(99.5,0,-0.5),data,label='data')
		ax = pl.gca()
		ax.set_xlim(ax.get_xlim()[::-1])
		pl.xlim([100,80])
		pl.ylim([0,(np.prod(tmp_data.shape)-(tmp_data>0).sum())*3])
		pl.xlabel('Tolerance')
		pl.ylabel('No. Bad Pixels')
		pl.vlines(self.tolerance,0,(np.prod(tmp_data.shape)-(tmp_data>0).sum())*3,color='r',label = 'Tolerance %.2f'%self.tolerance)
		pl.legend(loc='best')
		pl.savefig(self.ff_path+'stats_4_tol_plot.pdf')
		pl.clf()

		pl.figure(1)
		pl.title('Dead pixels')
		offset = 92.5
		step=0.3
		for i in np.arange(1,6):
			for j in np.arange(1,6):
				pl.subplot(5,5,((i-1)*5)+j)
				#print i,',',j,',',((i-1)*5)+j
				tmp_tolerance = ((i-1)*5+j)*step+offset
				pl.axis('off')
				#pl.tight_layout()
				pl.title('%.2f'%tmp_tolerance+r'%',fontsize=8)
				pl.imshow((tmp_data>=tmp_tolerance),cmap='spectral')#*ff_data[1,:,:])
		pl.savefig(self.ff_path+'stats_4_tol_as_ims.pdf')
		pl.clf()

		return (tmp_data>=self.tolerance)

	def final_mask(self,list_masks):
		for i in list_masks:
			self.tot_mask *= i

		pl.figure(1)
		pl.imshow(self.tot_mask,cmap='spectral')
		pl.title('Final Mask - %i pixels excluded'%(np.prod(self.tot_mask.shape)-self.tot_mask.sum()))
		pl.colorbar()
		pl.savefig(self.ff_path+'image-ff_final_mask.pdf')
		pl.clf()

	def plot_SD_image(self,clim=[0,0.05]):
		# hist of SD error and image of SD error
		pl.figure(1)
		pl.subplot(221)
		tmp = self.I_sigma*self.tot_mask/(np.nansum((self.I_bar*self.tot_mask))/np.nansum(self.tot_mask))#self.I_sigma/self.I_bar.sum()*(np.prod(self.I_bar.shape)-(self.I_bar==0).sum()))
		mean_I_sigma = np.nanmean(tmp.flatten())
		n, bins, patches = pl.hist(tmp.flatten()[np.isfinite(tmp.flatten())], np.arange(0,np.sqrt(mean_I_sigma),np.sqrt(mean_I_sigma)*2/100), normed=0, facecolor='green', alpha=0.5)
		pl.title('normalised SD distribution')
		pl.xlabel('SD')
		pl.ylabel('Frequency')
		pl.subplot(222)
		pl.title('normalised SD map')
		pl.imshow(tmp,cmap='spectral')
		pl.colorbar()
		pl.clim(clim)
		pl.xlabel('pixel')
		pl.ylabel('pixel')
		#pl.clim(mean_I_sigma +np.sqrt(mean_I_sigma)*-3, mean_I_sigma +np.sqrt(mean_I_sigma)*3)
		#pl.colorbar()
		pl.savefig(self.ff_path+'stats_5_SD.pdf')
		pl.clf()

#	def gen_ff(self,):
#		self.I_bar_tot = (self.I_bar*self.tot_mask).sum()/self.tot_mask.sum()
#		self.ff = np.where(self.tot_mask,self.I_bar_tot.sum()/self.I_bar,0)
#		self.ff_unc = np.where(self.tot_mask,np.sqrt(1.0/self.data.shape[2])*self.I_sigma/self.I_bar,0)
	def gen_ff(self,):
        # find the scale factor to normalise the data
		self.I_bar_sf = np.nansum((self.I_bar*self.tot_mask))/np.nansum(self.tot_mask)
		#self.I_bar_sf = self.I_bar/self.hist_coeff[1]
		self.ff = np.where(self.tot_mask,self.I_bar_sf/self.I_bar,0)
		self.ff_unc = np.where(self.tot_mask,np.sqrt(1.0/self.data.shape[2])*self.I_sigma/self.I_bar,0)

	def plot_ff(self,clim=[0.95,1.05],clim_unc=[0,0.025]):
		pl.figure(1)
		pl.subplot(221)
		pl.title('Flatfield')
		pl.imshow(self.ff,cmap='spectral')
		pl.clim(clim)
		pl.colorbar()
		pl.subplot(222)
		pl.title('Flatfield Uncertainty')
		pl.imshow(self.ff_unc,cmap='spectral')
		#pl.clim((np.compress(self.ff_unc.flatten()>0,self.ff_unc.flatten())).min(),(np.compress(self.ff_unc.flatten()>0,self.ff_unc.flatten())).max())
		pl.colorbar()
		pl.clim(clim_unc)
		pl.savefig(self.ff_path+"stats_6_ff.pdf")
		pl.clf()

	def plot_worst_pixel(self,):
		# worst pixel versus a typical pixel as a function of image number
		pl.figure(1)
		pl.subplot(211)
		x,y = np.where(self.I_sigma*self.tot_mask==np.nanmax(self.I_sigma*self.tot_mask))
		tmp_data1 = self.data[:,x,y]
		#tmp_data1 = self.data_dev[x[0],y[0]]**2

		n, bins, patches = pl.hist(tmp_data1.flatten()[np.isfinite(tmp_data1.flatten())],np.arange(np.nanmin(tmp_data1),np.nanmax(tmp_data1),(np.nanmax(tmp_data1)-np.nanmin(tmp_data1))/10) , normed=0, facecolor='green', alpha=0.5)
		#pl.xlim(self.I_lims[0]-10,self.I_lims[1]+10)
		mn1=np.nanmean(tmp_data1)
		std1=np.nanstd(tmp_data1)
		pl.xlim(mn1-3*std1,mn1+3*std1)
		pl.ylabel('Frequency')
		pl.title('Worst Pixel')
		pl.subplot(212)

		a = find_nearest(self.I_sigma*self.tot_mask,np.nanmean(self.I_sigma*self.tot_mask))
		x,y = np.where(self.I_sigma*self.tot_mask==a)
		print x,y
		#tmp_data2 = (self.data_dev[x,y]/self.data[x,y])**2
		#tmp_data2 = self.I_sigma*self.tot_mask
		#tmp_data2 = self.data_dev[x,y]**2
		tmp_data2 = self.data[:,x[0],y[0]]

		n, bins, patches = pl.hist(tmp_data2.flatten()[np.isfinite(tmp_data2.flatten())],np.arange(np.nanmin(tmp_data2),np.nanmax(tmp_data2),(np.nanmax(tmp_data2)-np.nanmin(tmp_data2))/10) , normed=0, facecolor='black', alpha=0.5)
		pl.title('Average Pixel')
		#pl.xlim(self.I_lims[0]-10,self.I_lims[1]+10)
		mn2=np.nanmean(tmp_data1)
		std2=np.nanstd(tmp_data1)
		pl.xlim(mn2-3*std2,mn2+3*std2)
		pl.xlabel('Photon counts')
		pl.ylabel('Frequency')
		pl.savefig(self.ff_path+'stats_7_avg_wPix_hist.pdf')
		pl.clf()

		pl.figure(1)
		pl.plot(tmp_data1.flatten(),'g', label = 'Worst Pixel')
		pl.plot(tmp_data2.flatten(),'k', label = 'Average Pixel')
		pl.ylim(self.I_lims[0],self.I_lims[1])
		pl.xlabel('Exposure Number')
		pl.ylabel('Photons counts')
		pl.legend(loc='best')
		pl.savefig(self.ff_path+'stats_7_avg_wPix_time.pdf')
		pl.clf()

	def plot_rnd_ff_im(self,n_std=1):
		# normalised mean flatfield image,
		# a randomly chosen flatfield. could include line profiles or not as you wish.
		pl.figure(1)

		pl.subplot(221)
		pl.title('Random image')
		rand_num = int(np.random.rand()*self.no_ims)
		tmp = self.data[rand_num,:,:]*self.tot_mask
		data2plot = tmp/np.nansum(tmp)*(np.nansum((tmp>0)))
		pl.imshow(data2plot)
		pl.clim([np.nanmean(data2plot)-np.nanstd(data2plot)*n_std,np.nanmean(data2plot)+np.nanstd(data2plot)*n_std])
		#pl.clim(-data2plot.max()+2,data2plot.max())
		pl.colorbar()

		pl.subplot(222)
		pl.title('corrected w/ ff corr')
		data2plot = data2plot*self.ff
		pl.imshow(data2plot)
		pl.clim([np.nanmean(data2plot)-np.nanstd(data2plot)*n_std,np.nanmean(data2plot)+np.nanstd(data2plot)*n_std])
		pl.colorbar()

		pl.subplot(223)
		pl.title('flatfield')
		data2plot = self.ff
		pl.imshow(data2plot,cmap='spectral')
		pl.clim([np.nanmean(data2plot)-np.nanstd(data2plot)*n_std,np.nanmean(data2plot)+np.nanstd(data2plot)*n_std])
		pl.colorbar()

		pl.subplot(224)
		pl.title('mean w/ ff corr')
		data2plot = self.I_bar*self.tot_mask/self.I_bar_sf
		pl.imshow(data2plot*self.ff,cmap='spectral')
		pl.clim([np.nanmean(data2plot)-np.nanstd(data2plot)*n_std,np.nanmean(data2plot)+np.nanstd(data2plot)*n_std])
		pl.colorbar()

		pl.savefig(self.ff_path+'image-ff_im_correction.pdf')
		pl.clf()

	def get_ff(self):
		'''
		return the flatfield and relative uncertainties
		'''
		return self.ff, self.ff_unc

	def get_params(self):
		file = open(self.ff_path+'params.txt','w')
		file.write(str(self.__dict__))
		file.close()


	def apply_ff(self,data):
		'''
		Apply the ff to a dataset
		'''
		# apply the flatfield correction to the real data, calculate absolute uncertainties
		corr_data = np.zeros(data.shape)
		abs_unc = np.zeros(data.shape)
		rel_unc = np.sqrt(np.copy(data))/data
		for i in np.arange(data.shape[2]):
			corr_data[i:,:] = data[i,:,:]*self.ff
			abs_unc[i,:,:] = (rel_unc[i,:,:]+self.ff_unc)*corr_data[i,:,:,i]
		return corr_data, abs_unc

	def make_ff_h5(self, fn='ff.h5', save_raw_ims=False):
		# check the file doesn't exist
		try:
			os.listdir(self.ff_path).index(fn)
			print("File exists")
			q=input("would you like to overwrite it? [y/n]")
			if q=='y':
				ff_h5 = h5py.File(self.ff_path+fn,'w') 
			else:
				new_fn = input("New filename: ")
				ff_h5 = h5py.File(self.ff_path+fn,'w')

		except ValueError:
			print("File doesn't exist: Creating file")
			ff_h5 = h5py.File(self.ff_path+fn,'w')
	
		# add data to the file
		ff_h5.create_group('/ff')
		if save_raw_ims:
			ff_h5.create_dataset('ff/image_data', data = self.data,compression='gzip', compression_opts=9)
		ff_h5.create_dataset('ff/ff', data = self.ff,compression='gzip', compression_opts=9)
		ff_h5.create_dataset('ff/ff_rel_unc', data = self.ff_unc,compression='gzip', compression_opts=9)
		ff_h5.create_dataset('ff/bad_pix_mask', data = self.tot_mask,compression='gzip', compression_opts=9)
		for attr in list(self.__dict__.keys()):
			try:
				code ="ff_h5['ff/ff'].attrs['%s'] = self.%s"%(attr,attr)
				exec(code)
				print(code)
			except:
				pass
		ff_h5.close()

	def read_ff_h5(self, fn):

		try:
			ff_h5 = h5py.File(self.ff_path+fn,'r')
		except:
			print("File does not exist: ", self.ff_path)
			print(os.listdir(self.ff_path))

		ff = ff_h5['ff/ff'].value
		ff_unc = ff_h5['ff/ff_rel_unc'].value
		ff_attrs = ff_h5['ff/ff'].attrs
		#print("FF Attributes: ", end=' ') 
		for attr in list(ff_attrs.keys()):
			print(attr) #, ' : ', ff_attrs[attr]
		ff_h5.close()
		return ff,ff_unc




# useful functions
def find_nearest(a, a0):
    "Element in nd array `a` closest to the scalar value `a0`"
    idx = np.abs(np.nan_to_num(a) - np.nan_to_num(a0)).argmin()
    return np.nan_to_num(a).flat[idx]

# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))


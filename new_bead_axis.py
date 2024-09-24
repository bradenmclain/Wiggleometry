import trimesh
import glob
import numpy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from multiprocessing import Pool, cpu_count
import os
import copy
from scipy.optimize import minimize
from scipy.ndimage import binary_erosion,binary_dilation
import sys
from scipy.interpolate import interp1d
from icecream import ic
from scipy.interpolate import LinearNDInterpolator, CloughTocher2DInterpolator,griddata
import json
from numpyencoder import NumpyEncoder
from scipy.signal import find_peaks, peak_prominences, peak_widths
from sklearn import linear_model, datasets
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.optimize import root_scalar
from scipy import interpolate
from find_peaks_click import select_rois_and_find_peaks


def rotationMatrix(angle,axis=numpy.array([0,0,1])):
	"""Returns the rotation matrix for rotating 'angle' radians about 'axis'."""
	ux=axis[0]
	uy=axis[1]
	uz=axis[2]
	ux2=axis[0]**2
	uy2=axis[1]**2
	uz2=axis[2]**2
	uxuy=axis[0]*axis[1]
	uxuz=axis[0]*axis[2]
	uyuz=axis[1]*axis[2]
	c=numpy.cos(angle)
	omc=1.-c
	s=numpy.sin(angle)
	R=numpy.array([[c+ux2*omc,     uxuy*omc-uz*s, uxuz*omc+uy*s],
		       [uxuy*omc+uz*s, c+uy2*omc,     uyuz*omc-ux*s],
		       [uxuz*omc-uy*s, uyuz*omc+ux*s, c+uz2*omc    ]])
	return R

def rotateUnitVector(angle,n=numpy.array([1.,0.,0.]),axis=numpy.array([0,0,1])):
	"""Returns a unit vector rotated 'angle' radians about 'axis'."""
	M=rotationMatrix(angle,axis)
	Rnew=numpy.matmul(n,M)
	Rnew=Rnew.reshape(3)
	return Rnew

def angleBetweenVectors(a,b):
	"""calculates the angle between vectors a and b, in radians."""
	na=numpy.linalg.norm(a)
	nb=numpy.linalg.norm(b)
	if (na==0)|(nb==0)|(numpy.allclose(a,b)):
		return 0
	else:
		return numpy.arccos(numpy.dot(a,b)/na/nb)

def align(mesh):
	"""Returns an aligned mesh object with its shortest axis aligned to Z and its longest axis aligned to X."""
	# pull the vertices out of the mesh
	verts=mesh.vertices
	
	# initial plane fit
	p,xdir,ydir,zdir=planeFit(verts)
	
	# sometimes it is upside down..
	if zdir[2]<0:
		zdir*=-1
		xdir*=-1
		
	# transform the vertex coordinates into this new orientation
	newx=numpy.sum((verts-p)*xdir,axis=1)
	newy=numpy.sum((verts-p)*ydir,axis=1)
	newz=numpy.sum((verts-p)*zdir,axis=1)
	newverts=numpy.stack((newx,newy,newz),axis=1)
	mesh=trimesh.Trimesh(vertices=newverts,faces=mesh.faces)
	
#	# cut at z=0
#	lower=mesh.slice_plane(numpy.array([0,0,0.1]),numpy.array([0,0,-1]))
#	dot_cutoff=0.9

#	flat_idx=numpy.nonzero(lower.vertex_normals[:,2]>dot_cutoff)
#	flat=lower.vertices[flat_idx,:][0]

#	substrate_interpolator=LinearNDInterpolator(flat[:,:2], flat[:,2], fill_value=0, rescale=True)
#	zflat=mesh.vertices[:,2]-substrate_interpolator(mesh.vertices[:,:2])
#	mesh=trimesh.Trimesh(vertices=numpy.stack((mesh.vertices[:,0],mesh.vertices[:,1],zflat),axis=1),faces=mesh.faces)
	return mesh

def planeFit(cloud):
	"""Principal component analysis (PCA) plane fitting algorithm.  If indices are provided, a subset of the cloud is used."""
	p=cloud.mean(axis=0) # plane point as the cloud mean
	cloud2=cloud-p
	matrix = numpy.cov(cloud2.T)
	eigenvalues, eigenvectors = numpy.linalg.eig(matrix)
	sort = eigenvalues.argsort()[::-1]
	eigenvalues = eigenvalues[sort]
	eigenvectors = eigenvectors[:,sort]
	zdir=eigenvectors[:,2]
	ydir=eigenvectors[:,1]
	xdir=eigenvectors[:,0]
	return p,xdir,ydir,zdir

def allthethings(F):
	"""Perform bead measurements on the filename F."""
	results={}
	a=trimesh.exchange.load.load_mesh(F) # load the STL file
	a=align(a) # align the mesh
	#a.export(F[:-4]+"_aligned.stl") # export the aligned mesh
	results["path"],zheight,final_state=measure(a,F)
	results["stats"]=analyze(results["path"],F)
	return results,zheight,final_state

def moving_average(x, w):
	return numpy.convolve(x, numpy.ones(w), 'valid') / w

def find_stub_indecies(binary_change,peaks):

	stubs = numpy.asarray(peaks)
	der = numpy.gradient((binary_change))

	x_org = numpy.arange(0,len(der))

	interp_function = interpolate.interp1d(x_org,binary_change)

	der_interp_function = interpolate.interp1d(x_org,der)
	xnew = numpy.arange(0,len(der)-1,.001)

	new_der = der_interp_function(xnew)
	new_sec_der = numpy.gradient(new_der)
	sec_der_interp_function = interpolate.interp1d(xnew,new_sec_der)

	zero_crossings = []
	der_peaks = []
	der_valleys = []

	for i in range(1, len(xnew)):
		if new_der[i-1] * new_der[i] < 0:  # Sign change detected
			# Step 2: Refine the zero crossing using root_scalar
			root_result = root_scalar(der_interp_function, bracket=[xnew[i-1], xnew[i]])
			if root_result.converged:
				zero_crossings.append((root_result.root))

	for zero_crossing in zero_crossings:
		if sec_der_interp_function(zero_crossing) >= 0:
			plt.scatter(zero_crossing, der_interp_function(numpy.array(zero_crossing)), color='red', zorder=5, label='Zero Crossings')
			der_valleys.append(zero_crossing)
		else:
			plt.scatter(zero_crossing, der_interp_function(numpy.array(zero_crossing)), color='blue', zorder=5, label='Zero Crossings')
			der_peaks.append(zero_crossing)

	#plt.show()

	lengths = []
	positions = []

	for peak in stubs:

		left = (numpy.where(((numpy.diff(numpy.sign(der_valleys-peak)) != 0)*1)==1)[0])
		if left.size > 0:
			left = int(left[0])
		if left != None:
			right = (left + 1)

			local_x = numpy.arange((der_valleys[int(left)]),(der_valleys[int(right)]),.001)
			local_y = interp_function(local_x)

			#if the left point is higher up, use that and the right point locally horizontal to it
			if interp_function(der_valleys[int(left)]) > interp_function(der_valleys[int(right)]):
				new_point = (numpy.where(((numpy.diff(numpy.sign(interp_function(der_valleys[int(left)])-local_y)) != 0)*1)==1))
				if new_point[0].size > 1:
					x1_pos = local_x[new_point[0][0]]
					x2_pos = local_x[new_point[0][-1]]

				else:
					x1_pos = der_valleys[int(left)]
					x2_pos = local_x[new_point[0][0]]
			#if the right point is higher up, use that and the left point locally horizontal to it
			else:
				new_point = (numpy.where(((numpy.diff(numpy.sign(local_y-interp_function(der_valleys[int(right)]))) != 0)*1)==1))
				
				#if the newly found line crosses over itself, take the inside points
				if new_point[0].size > 1:
					x1_pos = local_x[new_point[0][0]]
					x2_pos = local_x[new_point[0][-1]]

				else:
					x1_pos = local_x[new_point[0][0]]
					x2_pos = der_valleys[int(right)]
					
			lengths.append((x2_pos-x1_pos))
			positions.append([x1_pos,x2_pos,interp_function(x1_pos),interp_function(x2_pos)]) 



	return stubs, lengths, positions

def calculate_stub_time(peaks,points):

	peaks = numpy.sort(peaks)
	inital_stub_offset = 38
	stub_continue = False
	stub_count = 0
	stub_timing_factor = 1.1
	peak_masks = []

	for i,peak in enumerate(peaks):
		mask = []
		current_peak = peaks[i]
		previous_peak = peaks[i-1]
		peak_mask = [peak,mask]

		if i == len(peaks)-1:
			next_peak = current_peak
		else:
			next_peak = peaks[i +1]

		if i == 0:
			if next_peak == current_peak:
					stub_count += inital_stub_offset
					peak_mask = [peak,peak +inital_stub_offset]

			elif next_peak <= (current_peak + inital_stub_offset*stub_timing_factor):
				stub_count += (next_peak - current_peak)* stub_timing_factor
				peak_mask = [peak,(next_peak - current_peak)* stub_timing_factor]
				dwell = (next_peak - current_peak) 
				stub_continue = True
			else:
				stub_count += inital_stub_offset
				peak_mask = [peak,peak +inital_stub_offset]
				stub_continue = False
				dwell = inital_stub_offset
			
		elif current_peak != next_peak:
			if (next_peak - current_peak) <= dwell*stub_timing_factor:
				stub_count += (next_peak - current_peak)*stub_timing_factor
				peak_mask = [peak,(next_peak - current_peak)* stub_timing_factor]
				stub_continue = True
				dwell = (next_peak - current_peak)

			elif stub_continue:
				stub_count += dwell
				peak_mask = [peak,current_peak+dwell]
				stub_continue = False
				dwell = inital_stub_offset
			else:
				stub_count += inital_stub_offset
				dwell = inital_stub_offset
				peak_mask = [peak,current_peak+inital_stub_offset]
				stub_continue = False
			
		else:
			if stub_continue:
				if current_peak + dwell >len(points):
					stub_count += (len(points) - current_peak)
				else:
					stub_count += dwell
			else:
				if current_peak + inital_stub_offset >len(points):
					stub_count += (len(points)- current_peak)
				else:
					stub_count += inital_stub_offset
		peak_masks.append(peak_mask)
		# print(f'dwell is {dwell}')
	print(stub_count)
	print(f'percent time stubbing is {stub_count/730}')
	return peak_masks


def measure(mesh,F):
	# first trim off the substrate
	cut=mesh.slice_plane(numpy.array([0,0,0.25]),numpy.array([0,0,1]))
	test=cut.split(only_watertight=False)
	threshold=500
	keepers=[]
	for i in range(len(test)):
		if len(test[i].vertices)>threshold:
			keepers.append(test[i])
	bead=trimesh.util.concatenate(keepers)

	cut_interval=.1
	mesh.visual.face_colors = [200, 200, 250, 100]
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d',aspect='auto')
	ax.set(xlim=(-50, 50), ylim=(-3, 3),zlim=(0,2))
	
	xmin=numpy.min(bead.vertices[:,0])
	xmax=numpy.max(bead.vertices[:,0])
	sections=[(xval,bead.section(plane_origin=numpy.array([xval,0,0]),plane_normal=numpy.array([-1,0,0]))) for xval in numpy.linspace(xmin,xmax,int((xmax-xmin)/cut_interval))] 
	pnts=[]
	areas = []
	peaks = []
	for S in sections:
		if S[1]:
			ax.plot(S[1].vertices[:,0],S[1].vertices[:,1],S[1].vertices[:,2],'g.')
			weights=S[1].vertices[:,2]
			area=numpy.sum(weights)
			moments=numpy.array([numpy.array([S[1].vertices[i,0],S[1].vertices[i,1],S[1].vertices[i,2]/2])*weights[i] for i in range(len(weights))])
			centroid=numpy.sum(moments,axis=0)/area
			pnts.append(centroid)
			areas.append(area)
			peak_idx = numpy.argmax(S[1].vertices[:,2])
			peaks.append(S[1].vertices[peak_idx,:])

		# else:
		# 	pnts.append(numpy.array([S[0],0,0]))
	peaks = numpy.array(peaks)
	pnts = numpy.array(pnts)
	areas = numpy.array(areas)

	ax.plot(pnts[:,0],pnts[:,1],pnts[:,2],'r-')
	plt.clf()

	# Assuming you're fitting a line based on the 1D points in pnts[:, 0]
	x_values = pnts[:,0].reshape(-1,1) # Index or any other 1D inumpyut
	y_values = pnts[:, 1]  # The actual 1D data you want to fit

	# RANSAC fitting on the 1D data
	ransac = linear_model.RANSACRegressor(max_trials=5000, min_samples=12,stop_n_inliers=int(0.8*len(x_values)),residual_threshold=.05)
	ransac.fit(x_values, y_values)

	# Generate new x values for predicting the line
	ransac_adjust_line = ransac.predict(x_values)
	smooth_y_data_adjusted = moving_average(pnts[:,1]-ransac_adjust_line,5)
	first_peaks, properties2 = find_peaks(numpy.abs(smooth_y_data_adjusted),prominence=0.1,height=0.,wlen=40)
	
	plt.plot(smooth_y_data_adjusted)
	
	for peak in first_peaks:
		plt.plot(peak,smooth_y_data_adjusted[peak],'or')

	period = 30
	smooth_y_data_adjusted_nan = smooth_y_data_adjusted.copy()
	mask = numpy.ones(len(smooth_y_data_adjusted),dtype=bool)
	for peak in first_peaks:
		start = max(0, peak - period)
		end = min(len(smooth_y_data_adjusted), peak + period + 1)
		mask[start:end] = False
		
		# Set the values within the window to NaN
		smooth_y_data_adjusted_nan[start:end] = numpy.nan



	plt.plot(numpy.arange(0,len(smooth_y_data_adjusted),1),smooth_y_data_adjusted_nan,'orange')
	plt.ylim(-1.5,1.5)

	# Apply the mask to remove unwanted data points
	
	x = numpy.arange(0,len(smooth_y_data_adjusted),1)
	x_masked = x[mask]
	smooth_y_data_adjusted_masked = smooth_y_data_adjusted[mask]	

	ransac = linear_model.RANSACRegressor(max_trials=5000, min_samples=6,stop_n_inliers=int(0.8*len(x_values)),residual_threshold=.05)
	ransac.fit(x_masked.reshape(-1,1), smooth_y_data_adjusted_masked)
	ransac_adjust_line = ransac.predict(x.reshape(-1,1))
	
	plt.plot(ransac_adjust_line,'black')
	
	plt.ylim(-1.5,1.5)
	#plt.show()
	
	ransac_second_adjust = smooth_y_data_adjusted_nan[:]-ransac_adjust_line[:]
	plt.plot(numpy.arange(0,len(smooth_y_data_adjusted),1),ransac_second_adjust,'orange')
	plt.ylim(-1.5,1.5)
	plt.plot(numpy.zeros(len(smooth_y_data_adjusted)),'black')
	plt.plot(numpy.abs(smooth_y_data_adjusted))
	

	peaks, properties2 = find_peaks(numpy.abs(ransac_second_adjust),prominence=0.1,height=0.,wlen=40)

	for peak in peaks:
		plt.plot(peak,ransac_second_adjust[peak],'or')

	#plt.show()
	plt.plot(moving_average(pnts[:,2],10))
	#plt.show()
	total_peaks = numpy.append(first_peaks,peaks)
	plt.plot(smooth_y_data_adjusted)
	plt.plot(smooth_y_data_adjusted_nan)
	plt.xlabel('Distance in mm')
	plt.ylabel('Y centroild position in mm')
	thresholded_data = numpy.convolve(numpy.abs(smooth_y_data_adjusted)>0.15,numpy.ones(period))>2
	plt.plot(thresholded_data[:len(smooth_y_data_adjusted)])

	state_drip =  numpy.max(moving_average(pnts[:,2],10)>0.6)*1

	state_stub = numpy.max(thresholded_data)
	
	if state_drip == 1:
		final_state = 'fail drip'
	elif state_stub == 1:
		final_state = 'fail stub'
	else:
		final_state = 'pass'

	plt.ylim(-1.5,1.5)
	#plt.show()	
	plt.clf()
	
	return pnts,moving_average(pnts[:,2],10),final_state
	

def analyze(path,F):
	beadcenter=numpy.average(path,axis=0)
	yerror=[abs(pnt[1]-beadcenter[1]) for pnt in path]
	zerror=[abs(pnt[2]-beadcenter[2]) for pnt in path]
	results={}
	results["XY error range"]=[numpy.min(yerror), numpy.max(yerror)]
	results["XY error average"]=numpy.mean(yerror)
	results["XY error stdev"]=numpy.std(yerror)
	results["Z error range"]=[numpy.min(zerror), numpy.max(zerror)]
	results["Z error average"]=numpy.mean(zerror)
	results["Z error stdev"]=numpy.std(zerror)
	d = numpy.diff(path,axis=0)
	xy_norm = numpy.apply_along_axis(numpy.linalg.norm,1,d[:,:2])
	z_norm=numpy.abs(d[:,2])


	# results["X avg diff"] = numpy.average(xdiff)
	# results["Y avg diff"] = numpy.average(ydiff)
	# results["Z avg diff"] = numpy.average(zdiff)
	# results["X abs diff"] = numpy.sum(abs(xdiff))
	# results["Y abs diff"] = numpy.sum(abs(ydiff))
	# results["Z abs diff"] = numpy.sum(abs(zdiff))
	# results["X rms diff"] = numpy.sqrt(numpy.average((xdiff)**2))
	# results["Y rms diff"] = numpy.sqrt(numpy.average((ydiff)**2))
	# results["Z rms diff"] = numpy.sqrt(numpy.average((zdiff)**2))
	results["XY Norm Avg"] = numpy.average(xy_norm)
	results["Z Norm Avg"] = numpy.average(z_norm)
	results["XY Norm stdev"] = numpy.std(xy_norm)
	results["Z Norm stdev"] = numpy.std(z_norm)
	return results

if __name__=="__main__":
	if os.path.isdir(sys.argv[1]):
		final_info = []
		path = sys.argv[1]
		for filename in glob.glob(os.path.join(path, '*.stl')):
			
			bead_number = filename.split('_')[1].split('.')[0]
			with open(filename, 'r') as f:
				results,zheight,final_state=allthethings(filename)
				f=open(filename[:-4]+"_data.txt",'w')
				f.write(json.dumps(results,cls=NumpyEncoder))
				f.close()
				final_info.append([bead_number,final_state])


	final_info = sorted(final_info, key=lambda x: x[0])

	with open("output2.txt", "w") as file:
		for info in final_info:
			file.write(str(info)+'\n')
				

	if os.path.isfile(sys.argv[1]):
		with open(sys.argv[1], 'r') as f:
			results=allthethings(sys.argv[1])
			f=open(sys.argv[1][:-4]+"_data.txt",'w')
			f.write(json.dumps(results,cls=NumpyEncoder))
			f.close()

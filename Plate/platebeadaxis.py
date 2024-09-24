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
	
	
	# transform the vertex coordinates into this new orientation
	newx=numpy.sum((verts-p)*xdir,axis=1)
	newy=numpy.sum((verts-p)*ydir,axis=1)
	newz=numpy.sum((verts-p)*zdir,axis=1)
	newverts=numpy.stack((newx,newy,newz),axis=1)
	mesh=trimesh.Trimesh(vertices=newverts,faces=mesh.faces)


	substrate=mesh.slice_plane(numpy.array([0,0,0.2]),numpy.array([0,0,-1]))

	dot_cutoff=0.9
	flat_idx=numpy.nonzero(numpy.abs(substrate.vertex_normals[:,2])>dot_cutoff)
	flat=substrate.vertices[flat_idx,:][0]
	substrate_interpolator=LinearNDInterpolator(flat[:,:2], flat[:,2], fill_value=0, rescale=True)
	zflat=mesh.vertices[:,2]-substrate_interpolator(mesh.vertices[:,:2])
	mesh=trimesh.Trimesh(vertices=numpy.stack((mesh.vertices[:,0],mesh.vertices[:,1],zflat),axis=1),faces=mesh.faces)
	
	beads=mesh.slice_plane(numpy.array([0,0,0.25]),numpy.array([0,0,1]))
	
	test=beads.split(only_watertight=False)
	threshold=4000
	xdir_correction=numpy.array([0.,0.,0.])
	for i in range(len(test)):
		if len(test[i].vertices)>threshold:
			p,xdir,ydir,zdir=planeFit(numpy.array([1,1,0])*test[i].vertices)
			xdir_correction+=xdir
	xdir_correction/=numpy.linalg.norm(xdir_correction)
	ydir_correction=numpy.cross(numpy.array([0,0,1]),xdir_correction)
	
	newx=numpy.sum((mesh.vertices)*xdir_correction,axis=1)
	newy=numpy.sum((mesh.vertices)*ydir_correction,axis=1)
	newz=numpy.sum((mesh.vertices)*numpy.array([0,0,1]),axis=1)
	newverts=numpy.stack((newx,newy,newz),axis=1)
	mesh=trimesh.Trimesh(vertices=newverts,faces=mesh.faces)

	return mesh

def beadsplit(mesh):
	"""Splits up a plate into individual bead meshes"""
	beads={}
	beads=mesh.slice_plane(numpy.array([0,0,0.2]),numpy.array([0,0,1]))
	objects=beads.split(only_watertight=False)
	threshold=2000
	keepers={}
	Y_tol=3
	for i in range(len(objects)):
		if len(objects[i].vertices)>threshold:
			ymean=numpy.average(objects[i].vertices[:,1])
			if len(keepers)>0:
				keys=list(keepers.keys())
				ydists=numpy.abs(numpy.array(keys)-ymean)
				closest=numpy.argmin(ydists)
				closest_key=keys[closest]
				if ydists[closest]<Y_tol:
					print(f'concatenating {ymean:3.4f}')
					keepers[closest_key]=trimesh.util.concatenate((keepers[closest_key],objects[i]))
				else:
					print(f'adding {ymean:3.4f}')
					keepers[ymean]=objects[i]
			else:
				print("first entry")
				keepers[ymean]=objects[i]
	return keepers
	
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
	a.export(F[:-4]+"_aligned.stl") # export the aligned mesh
	results["beads"]=beadsplit(a)
	results["paths"]=measure(results["beads"],F)
	results["stats"]=analyze(results["paths"],F)
	return results

def measure(beads,F):
	paths={}
	for B in beads:
		bead=beads[B]
		cut_interval=0.1
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d',aspect='auto')
#		ax.set(xlim=(-50, 50), ylim=(-3, 3),zlim=(0,2))
		xmin=numpy.min(bead.vertices[:,0])
		xmax=numpy.max(bead.vertices[:,0])
		sections=[(xval,bead.section(plane_origin=numpy.array([xval,0,0]),plane_normal=numpy.array([-1,0,0]))) for xval in numpy.linspace(xmin,xmax,int((xmax-xmin)/cut_interval))] 
		pnts=[]
		for S in sections:
			if S[1]:
				ax.plot(S[1].vertices[:,0],S[1].vertices[:,1],S[1].vertices[:,2],'g.')
				weights=S[1].vertices[:,2]
				area=numpy.sum(weights)
				moments=numpy.array([numpy.array([S[1].vertices[i,0],S[1].vertices[i,1],S[1].vertices[i,2]/2])*weights[i] for i in range(len(weights))])
				centroid=numpy.sum(moments,axis=0)/area
				pnts.append(centroid)
		pnts=numpy.array(pnts)
		#print(xmax)

		if xmax > 30:
			left = numpy.where(pnts[:,0]>numpy.min(pnts[:,0])+12)[0][0]
			right = numpy.where(pnts[:,0]>numpy.max(pnts[:,0])-12)[0][0]
			pnts=pnts[left:right,:]
		
		paths[B]=pnts
		ax.plot(pnts[:,0],pnts[:,1],pnts[:,2],'r-')
		plt.savefig(F[:-4]+f'_Y{B:3.4f}_plot.pdf')
		
	return paths
	

def analyze(paths,F):
	results={}
	for P in paths:
		path = paths[P]
		beadcenter=numpy.average(path,axis=0)
		yerror=[abs(pnt[1]-beadcenter[1]) for pnt in path]
		zerror=[abs(pnt[2]-beadcenter[2]) for pnt in path]
		results[P]={}
		results[P]["XY error range"]=[numpy.min(yerror), numpy.max(yerror)]
		results[P]["XY error average"]=numpy.mean(yerror)
		results[P]["XY error stdev"]=numpy.std(yerror)
		results[P]["Z error range"]=[numpy.min(zerror), numpy.max(zerror)]
		results[P]["Z error average"]=numpy.mean(zerror)
		results[P]["Z error stdev"]=numpy.std(zerror)
	return results

if __name__=="__main__":
	results=allthethings(sys.argv[1])
	f=open(sys.argv[1][:-4]+"_data.txt",'w')
	del results["beads"]
	f.write(json.dumps(results,cls=NumpyEncoder))
	f.close()

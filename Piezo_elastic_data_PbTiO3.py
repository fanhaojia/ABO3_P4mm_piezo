#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
modifed by jiafanhao
based on ELATE wrote by François-Xavier Coudert at CNRS / Chimie ParisTech.
https://github.com/coudertlab/elate
"""
import numpy as np
import json
from ase.db import connect
import matplotlib as mpl
##mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import optimize

def dirVec1(theta, phi):
    return np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])

def dirVec2(theta, phi, chi):
    return np.array([np.cos(theta)*np.cos(phi)*np.cos(chi) - np.sin(phi)*np.sin(chi),
             np.cos(theta)*np.sin(phi)*np.cos(chi) + np.cos(phi)*np.sin(chi),
             - np.sin(theta)*np.cos(chi)])

def print_tensor_voigt(tensor):
    print('\n'.join([''.join(['{:^12.3f}'.format(item) for item in row])  for row in tensor]))

def print_tensor_3rd_rank(tensor):
    print('\n'.join(['\n'.join( [' '.join(['{:^12.3f}'.format(item) for item in row]) for row in block]) for block in tensor]))

# Functions to minimize/maximize
def minimize(func, dim):
    if dim == 2:
        r = ((0, np.pi), (0, 2*np.pi))
        n = 20
    elif dim == 3:
        r = ((0, np.pi), (0, 2*np.pi), (0, 2*np.pi))
        n = 20
    # TODO -- try basin hopping or annealing
    return optimize.brute(func, r, Ns = n, full_output = True, finish = optimize.fmin)[0:2]
def maximize(func, dim):
    res = minimize(lambda x: -func(x), dim)
    return (res[0], -res[1])

class Elastic:
    """An elastic tensor, along with methods to access it"""
    def __init__(self, s):
        """Initialize the elastic tensor from a string"""
        
        if s is None:
            raise ValueError("no matrix was provided")
        # Argument can be a 6-line string, a list of list, or a string representation of the list of list
        try:
            if type(json.loads(s)) == list: s = json.loads(s)
        except:
            pass
        if type(s) == str:
            # Remove braces and pipes
            s = s.replace("|", " ").replace("(", " ").replace(")", " ")
            # Remove empty lines
            lines = [line for line in s.split('\n') if line.strip()]
            if len(lines) != 6:
                raise ValueError("should have six rows")
            # Convert to float
            try:
                mat = [list(map(float, line.split())) for line in lines]
            except:
                raise ValueError("not all entries are numbers")
        elif type(s) == list:
            # If we already have a list, simply use it
            mat = s
        elif type(s) == np.ndarray:
            mat = s
        else:
            print (s)
            raise ValueError("invalid argument as matrix")
        # Make it into a square matrix
        mat = np.array(mat)
        if mat.shape != (6,6):
            # Is it upper triangular?
            if list(map(len, mat)) == [6,5,4,3,2,1]:
                mat = [ [0]*i + mat[i] for i in range(6) ]
                mat = np.array(mat)
        # Is it lower triangular?phi
        if list(map(len, mat)) == [1,2,3,4,5,6]:
            mat = [ mat[i] + [0]*(5-i) for i in range(6) ]
            mat = np.array(mat)
        if mat.shape != (6,6):
            raise ValueError("should be a square matrix")
        # Check that is is symmetric, or make it symmetric
        if np.linalg.norm(np.tril(mat, -1)) == 0:
            mat = mat + np.triu(mat, 1).transpose()
        if np.linalg.norm(np.triu(mat, 1)) == 0:
            mat = mat + np.tril(mat, -1).transpose()
        if np.linalg.norm(mat - mat.transpose()) > 1e-3:
            raise ValueError("should be symmetric, or triangular")
        elif np.linalg.norm(mat - mat.transpose()) > 0:
            mat = 0.5 * (mat + mat.transpose())
        # Store it
        self.CVoigt = mat
        # Put it in a more useful representation
        try:
            self.SVoigt = np.linalg.inv(self.CVoigt)
        except:
            raise ValueError("matrix is singular")
        VoigtMat = [[0, 5, 4], [5, 1, 3], [4, 3, 2]]
        def SVoigtCoeff(p,q): return 1. / ((1+p//3)*(1+q//3))
        self.Smat = [[[[ SVoigtCoeff(VoigtMat[i][j], VoigtMat[k][l]) * self.SVoigt[VoigtMat[i][j]][VoigtMat[k][l]]
                         for i in range(3) ] for j in range(3) ] for k in range(3) ] for l in range(3) ]
        return
    
    def Young(self, theta, phi):
        a = dirVec1(theta, phi)
        r = sum([a[i]*a[j]*a[k]*a[l] * self.Smat[i][j][k][l]
                  for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return 1/r

    def LC(self, theta, phi):
        a = dirVec1(theta, phi)
        r = sum([ a[i]*a[j] * self.Smat[i][j][k][k]
                  for i in range(3) for j in range(3) for k in range(3) ])
        return 1000 * r

    def bulk_modulus(self, theta, phi):
        a = dirVec1(theta, phi)
        r = sum([ a[i]*a[j] * self.Smat[i][j][k][k]
                  for i in range(3) for j in range(3) for k in range(3) ])
        '''
        if r < 0.001:
            r = 0.001
        '''
        return 1/r

    def shear(self, theta, phi, chi):
        a = dirVec1(theta, phi)
        b = dirVec2(theta, phi, chi)
        r = sum([ a[i]*b[j]*a[k]*b[l] * self.Smat[i][j][k][l]
                  for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return 1/(4*r)
    '''
    def shear3D_new2(self, theta, phi):
        npoints=180
        chi   = np.linspace(0, 2*np.pi, npoints)
        sh=[]
        for k in range (npoints):
            sh.append(self.shear(theta, phi, chi[k]))
        return max(sh), min(sh)
    '''
    def shear3D_new2(self, theta, phi):
        npoints=72
        chi   = np.linspace(0, 2*np.pi, npoints)
        sh=[]
        for k in range (npoints):
            sh.append(self.shear(theta, phi, chi[k]))
        sh=np.array(sh)
        bh=abs(sh)
        p_max_ind=np.unravel_index(np.argmax(bh, axis=None), bh.shape)
        p_min_ind=np.unravel_index(np.argmin(bh, axis=None), bh.shape)
        p_max=round(sh[p_max_ind],3)
        p_min=round(sh[p_min_ind],3)
        return p_max, p_min

    def Poisson(self, theta, phi, chi):
        a = dirVec1(theta, phi)
        b = dirVec2(theta, phi, chi)
        r1 = sum([ a[i]*a[j]*b[k]*b[l] * self.Smat[i][j][k][l]
                   for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        r2 = sum([ a[i]*a[j]*a[k]*a[l] * self.Smat[i][j][k][l]
                   for i in range(3) for j in range(3) for k in range(3) for l in range(3) ])
        return -r1/r2    
    
    def poisson3D_new2(self, theta, phi):
        npoints=72
        chi   = np.linspace(0, 2*np.pi, npoints)
        sh=[]
        for k in range (npoints):
            sh.append(self.Poisson(theta, phi, chi[k]))
        sh=np.array(sh)
        bh=abs(sh)
        p_max_ind=np.unravel_index(np.argmax(bh, axis=None), bh.shape)
        p_min_ind=np.unravel_index(np.argmin(bh, axis=None), bh.shape)
        p_max=round(sh[p_max_ind],3)
        p_min=round(sh[p_min_ind],3)
        return p_max, p_min
    
    def averages(self):
        A = (self.CVoigt[0][0] + self.CVoigt[1][1] + self.CVoigt[2][2]) / 3
        B = (self.CVoigt[1][2] + self.CVoigt[0][2] + self.CVoigt[0][1]) / 3
        C = (self.CVoigt[3][3] + self.CVoigt[4][4] + self.CVoigt[5][5]) / 3
        a = (self.SVoigt[0][0] + self.SVoigt[1][1] + self.SVoigt[2][2]) / 3
        b = (self.SVoigt[1][2] + self.SVoigt[0][2] + self.SVoigt[0][1]) / 3
        c = (self.SVoigt[3][3] + self.SVoigt[4][4] + self.SVoigt[5][5]) / 3

        KV = (A + 2*B) / 3
        GV = (A - B + 3*C) / 5

        KR = 1 / (3*a + 6*b)
        GR = 5 / (4*a - 4*b + 3*c)

        KH = (KV + KR) / 2
        GH = (GV + GR) / 2

        return [ [KV, 1/(1/(3*GV) + 1/(9*KV)), GV, (1 - 3*GV/(3*KV+GV))/2],
                 [KR, 1/(1/(3*GR) + 1/(9*KR)), GR, (1 - 3*GR/(3*KR+GR))/2],
                 [KH, 1/(1/(3*GH) + 1/(9*KH)), GH, (1 - 3*GH/(3*KH+GH))/2] ]

    def shear2D(self, theta, phi):
        ftol = 0.001
        xtol = 0.01
        def func1(z): return self.shear([theta, phi, z])
        r1 = optimize.minimize(func1, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.shear([theta, phi, z])
        r2 = optimize.minimize(func2, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun))

    def shear3D(self, theta, phi, guess1 = np.pi/2.0, guess2 = np.pi/2.0):
        tol = 0.005
        def func1(z): return self.shear([theta, phi, z])
        r1 = optimize.minimize(func1, guess1, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.shear([theta, phi, z])
        r2 = optimize.minimize(func2, guess2, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        return (float(r1.fun), -float(r2.fun), float(r1.x), float(r2.x))
    
    def Poisson2D(self, theta, phi):
        ftol = 0.001
        xtol = 0.01
        def func1(z): return self.Poisson([theta, phi, z])
        r1 = optimize.minimize(func1, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.Poisson([theta, phi, z])
        r2 = optimize.minimize(func2, np.pi/2.0, args=(), method = 'Powell', options={"xtol":xtol, "ftol":ftol})#, bounds=[(0.0,np.pi)])
        return (min(0,float(r1.fun)), max(0,float(r1.fun)), -float(r2.fun))

    def poisson3D(self, theta, phi, guess1 = np.pi/2.0, guess2 = np.pi/2.0):
        tol = 0.005
        def func1(z): return self.Poisson([theta, phi, z])
        r1 = optimize.minimize(func1, guess1, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        def func2(z): return -self.Poisson([theta, phi, z])
        r2 = optimize.minimize(func2, guess2, args=(), method = 'COBYLA', options={"tol":tol})#, bounds=[(0.0,np.pi)])
        return (min(0,float(r1.fun)), max(0,float(r1.fun)), -float(r2.fun), float(r1.x), float(r2.x))



class Piezoelectric:
    def __init__(self, d_ij):
        if d_ij is None:
            raise ValueError("no matrix was provided")
        if d_ij.shape != (3,6):
            raise ValueError("should be a 3*6 matrix")
        self.d_ij=d_ij    
        VoigtMat = [[0, 5, 4], [5, 1, 3], [4, 3, 2]]
        def VoigtCoeff_3rd(p):
            if p <3:
                r=1.0
            else:
                r=0.5
            return r
        mat0=[]
        for i in range(3):
            a=[]
            for j in range(3):
                b=[]
                for k in range(3):
                    p=VoigtMat[j][k]
                    b.append(VoigtCoeff_3rd(p)*self.d_ij[i][p])
                a.append(b)
            mat0.append(a)
        self.mat=np.array(mat0)

        return
    
    def piezo_3d_surface(self, theta, phi):
        a = dirVec1(theta, phi)
        r = sum([a[i]*a[j]*a[k]*self.mat[i][j][k] \
                 for i in range(3) for j in range(3) for k in range(3)])
        return r
    
    def piezo_lateral(self, theta, phi, chi):
        a = dirVec1(theta, phi)
        b = dirVec2(theta, phi, chi)
        r = sum([a[i]*b[j]*b[k]*self.mat[i][j][k] \
                 for i in range(3) for j in range(3) for k in range(3)])
        return r
        
    def piezo_3d_surface2(self, theta, phi):
        npoints=72
        chi   = np.linspace(0, 2*np.pi, npoints)
        sh=[]
        for k in range (npoints):
            sh.append(self.piezo_lateral(theta, phi, chi[k]))
        
        sh=np.array(sh)
        bh=abs(sh)
        p_max_ind=np.unravel_index(np.argmax(bh, axis=None), bh.shape)
        p_min_ind=np.unravel_index(np.argmin(bh, axis=None), bh.shape)
        p_max=round(sh[p_max_ind],3)
        p_min=round(sh[p_min_ind],3)
        return p_max, p_min


def write3DPlotData(dataX, dataY, dataZ, fname="XYZ"):
    fc=open(fname,'w')
    fc=open(fname,'a+')
    n=len(dataX);m=len(dataX[0])
    for i in range(n):
        for j in range(m):
            kline = '%13.9f%13.9f%13.9f\n' % (dataX[i][j], dataY[i][j], dataZ[i][j])
            fc.writelines(kline)
    fc.close()
    
def write3DPlotData2(dataX, dataY, dataZ, dataP, fname="XYZ"):
    fc=open(fname,'w')
    fc=open(fname,'a+')
    n=len(dataX);m=len(dataX[0])
    for i in range(n):
        for j in range(m):
            kline = '%13.9f%13.9f%13.9f%13.9f\n' % (dataX[i][j], dataY[i][j], dataZ[i][j], dataP[i][j])
            fc.writelines(kline)
    fc.close()
    
def make3DPlot(func, npoints=100, vmin=None, vmax=None, cmp='viridis', pic_name='pic.png'):
    theta = np.linspace(0, np.pi, npoints)
    phi   = np.linspace(0, 2*np.pi, 2*npoints)
    
    dataX = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataY = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataZ = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataP = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    
    dataXi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataYi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataZi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    #dataPi = [[1.0 for i in range(len(phi))] for j in range(len(theta))]
    
    for i in range(len(theta)):
        for j in range(len(phi)):
            p = func(theta[i], phi[j])

            x = abs(p)*np.sin(theta[i]) * np.cos(phi[j])
            y = abs(p)*np.sin(theta[i]) * np.sin(phi[j])
            z = abs(p)*np.cos(theta[i])
            
            xi = np.sin(theta[i]) * np.cos(phi[j])
            yi = np.sin(theta[i]) * np.sin(phi[j])
            zi = np.cos(theta[i])
            
            dataX[i][j] = x
            dataY[i][j] = y
            dataZ[i][j] = z
            dataP[i][j] = p

            dataXi[i][j] = xi
            dataYi[i][j] = yi
            dataZi[i][j] = zi
            
    write3DPlotData(dataX, dataY, dataZ, fname='d_XYZ')
    write3DPlotData2(dataXi, dataYi, dataZi, dataP, fname='dd_xyz')
    
    
    p_max_ind=np.unravel_index(np.argmax(dataP, axis=None), dataP.shape)
    p_min_ind=np.unravel_index(np.argmin(dataP, axis=None), dataP.shape)
    p_max=round(dataP[p_max_ind[0]][p_max_ind[1]],3)
    p_min=round(dataP[p_min_ind[0]][p_min_ind[1]],3)
    print (" ")
    #print (pic_name, p_max_ind, p_min_ind)
    v_max=dirVec1(theta[p_max_ind[0]], phi[p_max_ind[1]])
    print ("max:", p_max, ' direction: ', '(%.2f, %.2f, %.2f)'%(v_max[0], v_max[1], v_max[2]))
    v_min=dirVec1(theta[p_min_ind[0]], phi[p_min_ind[1]])
    print ("min:", p_min, ' direction: ', '(%.2f, %.2f, %.2f)'%(v_min[0], v_min[1], v_min[2]))
    print ("Anisotropy", round(p_max/p_min, 3))
    
    
    ax = plt.subplot(111, projection='3d')

    if vmin==None: 
        vmin=p_min
    if vmax==None:
        vmax=p_max
    #ax.quiver(0, 0, 0, v_max[0], v_max[1], v_max[2], color='k', length=400, arrow_length_ratio=0.08)
    #ax.quiver(0, 0, 150, v_min[0], v_min[1], 150-v_min[2], color='k', length=1, arrow_length_ratio=0.08)
    
    Sc=ax.scatter(dataX, dataY, dataZ, c=dataP, cmap=cmp, vmin=vmin, vmax=vmax)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    

    ax.set_xlim(-230, 230)
    ax.set_ylim(-230, 230)
    ax.set_zlim(-120, 120) 
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-180, -90, 0, 90, 180])
    ax.set_zticks([-100, -50, 0, 50, 100])
    ax.set_xticklabels(['-180', '-90', '0', '90', '180'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_yticklabels(['-180', '-90', '0', '90', '180'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_zticklabels(['-100', '-50', '0', '50', '100'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'}) 

    cbar=plt.colorbar(Sc,orientation="vertical", fraction=0.05, pad=0.15, shrink=0.6)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.tick_params(labelsize=8)
    
    
    ax.view_init(elev=20, azim=300)
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()

def make3DPlot_B(func, npoints=100, vmin=None, vmax=None, cmp='viridis', pic_name='pic.png'):
    theta = np.linspace(0, np.pi, npoints)
    phi   = np.linspace(0, 2*np.pi, 2*npoints)
    
    dataX = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataY = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataZ = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataP = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    
    dataXi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataYi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataZi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    #dataPi = [[1.0 for i in range(len(phi))] for j in range(len(theta))]
    
    for i in range(len(theta)):
        for j in range(len(phi)):
            p = func(theta[i], phi[j])

            x = abs(p)*np.sin(theta[i]) * np.cos(phi[j])
            y = abs(p)*np.sin(theta[i]) * np.sin(phi[j])
            z = abs(p)*np.cos(theta[i])
            
            xi = np.sin(theta[i]) * np.cos(phi[j])
            yi = np.sin(theta[i]) * np.sin(phi[j])
            zi = np.cos(theta[i])
            
            dataX[i][j] = x
            dataY[i][j] = y
            dataZ[i][j] = z
            dataP[i][j] = p

            dataXi[i][j] = xi
            dataYi[i][j] = yi
            dataZi[i][j] = zi
            
    write3DPlotData(dataX, dataY, dataZ, fname='d_XYZ')
    write3DPlotData2(dataXi, dataYi, dataZi, dataP, fname='dd_xyz')
    
    
    p_max_ind=np.unravel_index(np.argmax(dataP, axis=None), dataP.shape)
    p_min_ind=np.unravel_index(np.argmin(dataP, axis=None), dataP.shape)
    p_max=round(dataP[p_max_ind[0]][p_max_ind[1]],3)
    p_min=round(dataP[p_min_ind[0]][p_min_ind[1]],3)
    print (" ")
    #print (pic_name, p_max_ind, p_min_ind)
    v_max=dirVec1(theta[p_max_ind[0]], phi[p_max_ind[1]])
    print ("max:", p_max, ' direction: ', '(%.2f, %.2f, %.2f)'%(v_max[0], v_max[1], v_max[2]))
    v_min=dirVec1(theta[p_min_ind[0]], phi[p_min_ind[1]])
    print ("min:", p_min, ' direction: ', '(%.2f, %.2f, %.2f)'%(v_min[0], v_min[1], v_min[2]))
    print ("Anisotropy", round(p_max/p_min, 3))
    
    
    ax = plt.subplot(111, projection='3d')
    '''
    #ax.set_xlabel('$\it^{x}$')
    #ax.set_ylabel('$\it^{y}$')
    #ax.set_zlabel('$\it^{z}$', rotation=90, ha='right')
    
    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    ax.set_zlim(-400, 300) 
    #ax.set_zlim(1.1*vmin,1.1*vmax)
    '''
    #bulk modulus
    #ax.quiver(0, 0, 0, -v_max[0], -v_max[1], v_max[2], color='k', length=500, arrow_length_ratio=0.08)
    #ax.quiver(0, 0, 0, v_min[0], v_min[1], v_min[2], color='k', length=170, arrow_length_ratio=0.08)
    
    #ax.set_xlim(-450, 450)
    #ax.set_ylim(-450, 450)
    #ax.set_zlim(-610, 610) 
    #ax.set_xticks([-800, -400, 0, 400, 800])
    #ax.set_yticks([-800, -400, 0, 400, 800])
    #ax.set_zticks([-300, -150, 0, 150, 300])
    #ax.set_xticklabels([], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    #ax.set_yticklabels([], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    #ax.set_zticklabels([], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'}) 
    

    #piezo
    #ax.set_xlim(-500, 500)
    #ax.set_ylim(-500, 500)
    #ax.set_ylim(-500, 500)
    #ax.set_xticks([-50, -25, 0, 25, 50])
    #ax.set_yticks([-50, -25, 0, 25, 50])
    #ax.set_zticks([-600, -300,0,300, 600])
    #ax.set_xticklabels([ '-50', '-25', '0', '25', '50'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    #ax.set_yticklabels([ '-50', '-25', '0', '25', '50'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    #x.set_zticklabels([ '-600, -300', '0', '300', '600'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'}) 

    if vmin==None: 
        vmin=p_min
    if vmax==None:
        vmax=p_max

    Sc=ax.scatter(dataX, dataY, dataZ, c=dataP, cmap=cmp, vmin=vmin, vmax=vmax)
    
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    cbar=plt.colorbar(Sc,orientation="vertical", fraction=0.05, pad=0.15, shrink=0.6)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.tick_params(labelsize=8)
    
    ax.view_init(elev=20, azim=300)
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()
    
def make3DPlot_d33(func, npoints=100, vmin=None, vmax=None, cmp='viridis', pic_name='pic.png'):
    theta = np.linspace(0, np.pi, npoints)
    phi   = np.linspace(0, 2*np.pi, 2*npoints)
    
    dataX = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataY = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataZ = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataP = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    
    dataXi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataYi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataZi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    #dataPi = [[1.0 for i in range(len(phi))] for j in range(len(theta))]
    
    for i in range(len(theta)):
        for j in range(len(phi)):
            p = func(theta[i], phi[j])
            x = abs(p)*np.sin(theta[i]) * np.cos(phi[j])
            y = abs(p)*np.sin(theta[i]) * np.sin(phi[j])
            z = abs(p)*np.cos(theta[i])
            xi = np.sin(theta[i]) * np.cos(phi[j])
            yi = np.sin(theta[i]) * np.sin(phi[j])
            zi = np.cos(theta[i])
            
            dataX[i][j] = x
            dataY[i][j] = y
            dataZ[i][j] = z
            dataP[i][j] = p

            dataXi[i][j] = xi
            dataYi[i][j] = yi
            dataZi[i][j] = zi
            
    write3DPlotData(dataX, dataY, dataZ, fname='d_XYZ')
    write3DPlotData2(dataXi, dataYi, dataZi, dataP, fname='dd_xyz')
    
    p_max_ind=np.unravel_index(np.argmax(np.abs(dataP), axis=None), dataP.shape)########################importnat of the abs()
    p_min_ind=np.unravel_index(np.argmin(np.abs(dataP), axis=None), dataP.shape)
    p_max=round(dataP[p_max_ind[0]][p_max_ind[1]],3)
    p_min=round(dataP[p_min_ind[0]][p_min_ind[1]],3)
    print (" ")
    #print (pic_name, p_max_ind, p_min_ind)
    v_max=dirVec1(theta[p_max_ind[0]], phi[p_max_ind[1]])
    print ("max:", p_max, ' direction: ', '(%.2f, %.2f, %.2f)'%(v_max[0], v_max[1], v_max[2]))
    v_min=dirVec1(theta[p_min_ind[0]], phi[p_min_ind[1]])
    print ("min:", p_min, ' direction: ', '(%.2f, %.2f, %.2f)'%(v_min[0], v_min[1], v_min[2]))
    print ("Anisotropy", round(p_max/p_min, 3))
    
    
    ax = plt.subplot(111, projection='3d')
    #piezo

    ax.set_xlim(-115, 115)
    ax.set_ylim(-115, 115)
    ax.set_zlim(-250, 250)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_yticks([-100, -50, 0, 50, 100])
    ax.set_zticks([-200, -100, 0, 100, 200])
    ax.set_xticklabels([ '-100', '-50', '0', '50', '100'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_yticklabels([ '-100', '-50', '0', '50', '100'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_zticklabels([ '-200', '-100', '0', '100', '200'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'}) 

    if vmin==None: 
        vmin=-p_max
    if vmax==None:
        vmax=p_max
    Sc=ax.scatter(dataX, dataY, dataZ, c=dataP, cmap=cmp, vmin=vmin, vmax=vmax)
    
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    
    cbar=plt.colorbar(Sc,orientation="vertical", fraction=0.05, pad=0.15, shrink=0.6)
    cbar.set_ticks([-vmax, vmax])
    cbar.ax.tick_params(labelsize=8)
    
    ax.view_init(elev=12, azim=300)
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()



def make3DPlot_e33(func, npoints=100, vmin=None, vmax=None, cmp='viridis', pic_name='pic.png'):
    theta = np.linspace(0, np.pi, npoints)
    phi   = np.linspace(0, 2*np.pi, 2*npoints)
    
    dataX = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataY = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataZ = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataP = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    
    dataXi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataYi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataZi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    #dataPi = [[1.0 for i in range(len(phi))] for j in range(len(theta))]
    
    for i in range(len(theta)):
        for j in range(len(phi)):
            p = func(theta[i], phi[j])

            x = abs(p)*np.sin(theta[i]) * np.cos(phi[j])
            y = abs(p)*np.sin(theta[i]) * np.sin(phi[j])
            z = abs(p)*np.cos(theta[i])
            
            xi = np.sin(theta[i]) * np.cos(phi[j])
            yi = np.sin(theta[i]) * np.sin(phi[j])
            zi = np.cos(theta[i])
            
            dataX[i][j] = x
            dataY[i][j] = y
            dataZ[i][j] = z
            dataP[i][j] = abs(p)

            dataXi[i][j] = xi
            dataYi[i][j] = yi
            dataZi[i][j] = zi
            
    write3DPlotData(dataX, dataY, dataZ, fname='d_XYZ')
    write3DPlotData2(dataXi, dataYi, dataZi, dataP, fname='dd_xyz')
    
    
    p_max_ind=np.unravel_index(np.argmax(dataP, axis=None), dataP.shape)
    p_min_ind=np.unravel_index(np.argmin(dataP, axis=None), dataP.shape)
    p_max=round(dataP[p_max_ind[0]][p_max_ind[1]],3)
    p_min=round(dataP[p_min_ind[0]][p_min_ind[1]],3)
    print (" ")
    #print (pic_name, p_max_ind, p_min_ind)
    v_max=dirVec1(theta[p_max_ind[0]], phi[p_max_ind[1]])
    print ("max:", p_max, ' direction: ', '(%.2f, %.2f, %.2f)'%(v_max[0], v_max[1], v_max[2]))
    v_min=dirVec1(theta[p_min_ind[0]], phi[p_min_ind[1]])
    print ("min:", p_min, ' direction: ', '(%.2f, %.2f, %.2f)'%(v_min[0], v_min[1], v_min[2]))
    print ("Anisotropy", round(p_max/p_min, 3))
    
    
    ax = plt.subplot(111, projection='3d')
    '''
    #ax.set_xlabel('$\it^{x}$')
    #ax.set_ylabel('$\it^{y}$')
    #ax.set_zlabel('$\it^{z}$', rotation=90, ha='right')
    
    ax.set_xlim(-400, 400)
    ax.set_ylim(-400, 400)
    ax.set_zlim(-400, 300) 
    #ax.set_zlim(1.1*vmin,1.1*vmax)
    #bulk modulus
    ax.set_xticks([-600, -300, 0, 300, 600])
    ax.set_yticks([-600, -300, 0, 300, 600])
    #ax.set_zticks([-600, -300,0,300, 600])
    ax.set_xticklabels([ '-600', '-300', '0', '300', '600'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_yticklabels([ '-600', '-300', '0', '300', '600'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    #ax.set_zticklabels([ '-600, -300', '0', '300', '600'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})    
    '''
    #piezo
    ax.set_xlim(-3.5, 3.5)
    ax.set_ylim(-3.5, 3.5)
    ax.set_xticks([-3, -1.5, 0, 1.5, 3])
    ax.set_yticks([-3, -1.5, 0, 1.5, 3])
    #ax.set_zticks([-600, -300,0,300, 600])
    ax.set_xticklabels([ '-3', '-1.5', '0', '1.5', '3'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_yticklabels([ '-3', '-1.5', '0', '1.5', '3'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    #ax.set_zticklabels([ '-600, -300', '0', '300', '600'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'}) 
    
    if vmin==None: 
        vmin=p_min
    if vmax==None:
        vmax=p_max

    Sc=ax.scatter(dataX, dataY, dataZ, c=dataP, cmap=cmp, vmin=vmin, vmax=vmax)
    
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    cbar=plt.colorbar(Sc,orientation="vertical", fraction=0.05, pad=0.15, shrink=0.6)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.tick_params(labelsize=8)
    
    ax.view_init(elev=20, azim=300)
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()
    
    
def make3DPlot2(func, npoints=100, vmin=None, vmax=None,  cmp='plasma', pic_name='pic.png'):
    
    theta = np.linspace(0, np.pi, npoints)
    phi   = np.linspace(0, 2*np.pi, 2*npoints)
    #theta, phi = np.meshgrid(theta, phi)
    
    dataX = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataY = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataZ = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataXi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataYi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataZi = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataP_min = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    dataP_max = np.array([[0.0 for i in range(len(phi))] for j in range(len(theta))])
    
    for i in range(len(theta)):
        for j in range(len(phi)):
            pmax, pmin = func(theta[i], phi[j])
            x = abs(pmin)*np.sin(theta[i]) * np.cos(phi[j])
            y = abs(pmin)*np.sin(theta[i]) * np.sin(phi[j])
            z = abs(pmin)*np.cos(theta[i])
            xi = abs(pmax)*np.sin(theta[i]) * np.cos(phi[j])
            yi = abs(pmax)*np.sin(theta[i]) * np.sin(phi[j])
            zi = abs(pmax)*np.cos(theta[i])
            dataX[i][j] = x
            dataY[i][j] = y
            dataZ[i][j] = z
            dataXi[i][j] = xi
            dataYi[i][j] = yi
            dataZi[i][j] = zi
            dataP_min[i][j] = pmin
            dataP_max[i][j] = pmax
    write3DPlotData(dataX, dataY, dataZ, fname='d_XYZ')
    write3DPlotData(dataXi, dataYi, dataZi, fname='d_XYZi')

    p_max_ind=np.unravel_index(np.argmax(dataP_max, axis=None), dataP_max.shape)
    p_min_ind=np.unravel_index(np.argmin(dataP_min, axis=None), dataP_min.shape)
    p_max=round(dataP_max[p_max_ind[0]][p_max_ind[1]],3)
    
    p_max_ind2=np.unravel_index(np.argmax(dataP_min, axis=None), dataP_min.shape)
    p_max2=round(dataP_min[p_max_ind2[0]][p_max_ind2[1]],3)
    p_min=round(dataP_min[p_min_ind[0]][p_min_ind[1]],3)
    print (" ")
    #print (pic_name, p_max_ind, p_min_ind)
    v_max=dirVec1(theta[p_max_ind[0]], phi[p_max_ind[1]])
    print ("max:", p_max, ' direction: ', '(%.2f, %.2f, %.2f)'%(v_max[0], v_max[1], v_max[2]))
    v_min=dirVec1(theta[p_min_ind[0]], phi[p_min_ind[1]])
    print ("min:", p_min, ' direction: ', '(%.2f, %.2f, %.2f)'%(v_min[0], v_min[1], v_min[2]))
    print ("Anisotropy", round(p_max/p_min, 3))        
        
    ax = plt.subplot(111, projection='3d')

    ax.set_xlim(-45, 45)
    ax.set_ylim(-45, 45)
    ax.set_zlim(-45, 45)
    ax.set_xticks([-40, -20, 0, 20, 40])
    ax.set_yticks([-40, -20, 0, 20, 40])
    ax.set_zticks([-40, -20, 0, 20, 40])
    ax.set_xticklabels([ '-40', '-20', '0', '20', '40'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_yticklabels([ '-40', '-20', '0', '20', '40'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'})
    ax.set_zticklabels([ '-40', '-20', '0', '20', '40'], fontdict={'fontsize':10,'family': 'Times New Roman' ,'fontweight': 'normal'}) 

    if vmin==None: 
        vmin=p_min
    if vmax==None:
        vmax=p_max2
    Sc=ax.scatter(dataX, dataY, dataZ, c=dataP_min, cmap=cmp, vmin=vmin, vmax=vmax)
    Sc2=ax.plot_surface(dataXi, dataYi, dataZi, alpha=0.2)
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.5, "linestyle": '--', 'color':'grey'})
    cbar=plt.colorbar(Sc,orientation="vertical", fraction=0.05, pad=0.15, shrink=0.6)
    cbar.set_ticks([vmin, vmax])
    cbar.ax.tick_params(labelsize=8)
    
    ax.view_init(elev=12, azim=300)
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()

    
def ela_normal_lattice(func, theta=[np.pi/2, np.pi/2, 0], phi=[0, np.pi/2, 0]):
    d={'0': 'a', '1': 'b', '2': 'c'}
    p= func(theta, phi)
    for i in range(3):
        print (d[str(i)], '  X  ')
        print ('    ', round(p[i],1))

def makePolarPlot_all_S(func, theta=[np.pi/2, np.pi/2, 0], phi=[0, np.pi/2, 0], npoints=100,pic_name='shear.png'):
    d={'0': 'a', '1': 'b', '2': 'c'}
    chi=np.linspace(0, 2*np.pi, npoints)
    xxx=[]; yyy=[]
    for i in range(3):
        xx=[]; yy=[]; 
        for j in range(npoints):
            xx.append(theta[i])
            yy.append(phi[i])
        p= func(xx, yy, chi)
        print (d[str(i)], '--> max  ', '  min  ', '  A  ')
        print ('    ', round(max(p),1), '  ', round(min(p),1), ' ', round(max(p)/min(p), 3))
        x = p * np.cos(chi)
        y = p * np.sin(chi)
        xxx.append(x)
        yyy.append(y)

    p1, =plt.plot(xxx[0], yyy[0], lw=2, color='green' )
    p2, =plt.plot(xxx[1], yyy[1], lw=2, color='red' )
    p3, =plt.plot(xxx[2], yyy[2], lw=2, color='black' )
    plt.legend([p1,p2,p3,], ['{100}', '{010}', '{001}', ], loc='upper right', fontsize=14, frameon=False)
    
    plt.xlim(-205, 205)
    plt.ylim(-205, 205)
    plt.tick_params(axis='both',direction='in')
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()

def makePolarPlot_all_P(func, theta=[np.pi/2, np.pi/2, 0], phi=[0, np.pi/2, 0], npoints=100,pic_name='shear.png'):
    d={'0': 'a', '1': 'b', '2': 'c'}
    chi=np.linspace(0, 2*np.pi, npoints)
    xxx=[]; yyy=[]
    for i in range(3):
        xx=[]; yy=[]; 
        for j in range(npoints):
            xx.append(theta[i])
            yy.append(phi[i])
        p= func(xx, yy, chi)
        print (d[str(i)], '--> max  ', '  min  ', '  A  ')
        print ('    ', round(max(p),3), '  ', round(min(p),3), ' ', round(max(p)/min(p), 3))
        x = p * np.cos(chi)
        y = p * np.sin(chi)
        xxx.append(x)
        yyy.append(y)

    p1, =plt.plot(xxx[0], yyy[0], lw=2, color='green' )
    p2, =plt.plot(xxx[1], yyy[1], lw=2, color='red' )
    p3, =plt.plot(xxx[2], yyy[2], lw=2, color='black' )
    plt.legend([p1,p2,p3,], ['{100}', '{010}', '{001}', ], loc='upper right', fontsize=14, frameon=False)
    
    #plt.xlim(-205, 205)
    #plt.ylim(-205, 205)
    plt.tick_params(axis='both',direction='in')
    fig=plt.gcf()
    fig.subplots_adjust(top=0.88,bottom=0.12,left=0.16,right=0.9,hspace=0.2,wspace=0.2)
    plt.savefig(pic_name,dpi=600)
    plt.show()
##########################################main#############################################################
if __name__ == "__main__":
    database= './ABO3_P4mm_piezo.db'
    db=connect(database)
    
    i=0
    for row in db.select(ABO3_formula='PbTiO3', Space_group_index=99, DFPT='True'):
        i=i+1
        if i>1:
            raise RuntimeError('There are more than one system you wanted in the database!')
        system=row.ABO3_formula
        lattice=row.cell
        c_a_ratios=lattice[2][2]/lattice[0][0]
        
        data=row.data
        gaps=data['Band Gap']
        Cij=data['Elastic Stiffness Tensor Cij']  #Gpa
        Sij=data['Elastic Compliance Tensor Sij']*1000 #1/Tpa
        print ("##############################Elasticity##################################")
        print_tensor_voigt(Cij)
        
        eij=data['Piezoelectric Stress Tensor eij'] #C/m2
        dij=data['Piezoelectric Strain Tensor dij'] #pC/N
        print ("##############################Piezoelectricity##################################")
        print ("Piezoelectric Stress Tensor eij\n")
        print_tensor_voigt(eij)
        print ("Piezoelectric Strain Tensor dij\n")
        print_tensor_voigt(dij)
        print ("###############################################################################")
        
        dielectric_tensor=data['Dielectric Tensor']
        BEC_tensor=data['Born Effective Charge Tensor']
        phonon_frequencie=data['Phonon Frequency']
        
        print (Cij)
        elas = Elastic(Cij)
        make3DPlot(lambda x, y: elas.Young(x, y), npoints=100, cmp='viridis', pic_name="Youngs_modulus_PTO.png")
        #make3DPlot_B(lambda x, y: elas.LC(x, y), npoints=100, cmp='viridis', pic_name="LC_PTO.png")
        #make3DPlot_B(lambda x, y: elas.bulk_modulus(x, y), npoints=100, cmp='viridis', pic_name="bulk_modulus_PTO.png")
        #make3DPlot2(lambda x, y: elas.shear3D_new2(x, y),  npoints=60, cmp='viridis', pic_name="Shear_modulus_3D_BTO.png")
        #ake3DPlot2(lambda x, y: elas.poisson3D_new2(x, y),  npoints=60, cmp='viridis', pic_name="Poisson_ratio_3D_BTO.png")
    
        print ('Young')
        #ela_normal_lattice(elas.Young)
        print ('Bulk_modulus')
        #ela_normal_lattice(elas.bulk_modulus)
        print ('Shear')
        #makePolarPlot_all_S(elas.shear, npoints=100, pic_name="Shear_modulus_a.png")
        print ('Poisson')
        #makePolarPlot_all_P(elas.Poisson, npoints=100, pic_name="Poisson_ratio_a.png")
    
        #piezo = Piezoelectric(eij)
        #make3DPlot_e33(lambda x, y: piezo.piezo_3d_surface(x, y), npoints=100, cmp='twilight_shifted', pic_name="e33_PTO.png")
        #make3DPlot2(lambda x, y: piezo.piezo_3d_surface2(x, y),  npoints=60, cmp='twilight_shifted', pic_name="e31_BTO.png")
        
        piezo = Piezoelectric(dij)
        make3DPlot_d33(lambda x, y: piezo.piezo_3d_surface(x, y), npoints=100, cmp='twilight_shifted_r', pic_name="d33_PTO.png")
        make3DPlot2(lambda x, y: piezo.piezo_3d_surface2(x, y),  npoints=300, cmp='twilight_shifted_r', pic_name="d31_PTO.png")




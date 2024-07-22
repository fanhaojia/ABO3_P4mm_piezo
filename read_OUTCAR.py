#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 19:53:59 2022

@author: jiafanhao
"""
import os
import numpy as np
#from ase.io.vasp import read_vasp

__all__ = [
    'get_INCARInfo', 'get_bandInfo', 'get_elastic_tensor', \
    'get_piezoelectric_tensor', 'get_dielectric_tensor', 'get_born_effective_charge_tensor'
]

" Read OUTCAR to get INCAR "
def get_INCARInfo(fname='OUTCAR'):
    INCAR={};ZVAL=[];POTCAR=[]
    outcar = [line for line in open(fname) if line.strip()]
    for ii, line in enumerate(outcar):
        #Startparameter for this run:
        if 'PREC   =' in line:
            PREC=line.split()[2]
            INCAR.update({"PREC": PREC})
        if 'dimension x,y,z NGX =' in line:
            NGX=int(line.split()[4])
            NGY=int(line.split()[7])
            NGZ=int(line.split()[10])
            INCAR.update({"NGX": NGX})
            INCAR.update({"NGY": NGY})
            INCAR.update({"NGZ": NGZ})
        if 'ISPIN  =' in line:
            ISPIN=int(line.split()[2])
            INCAR.update({"ISPIN": ISPIN})
        if 'LNONCOLLINEAR =' in line:
            LNONCOLLINEAR=line.split()[2]
            INCAR.update({"LNONCOLLINEAR": LNONCOLLINEAR}) 
        #Exchange correlation treatment:
        if 'GGA type' in line:
            GGA=line.split()[2]
            INCAR.update({"GGA": GGA})
        #Electronic Relaxation:
        if 'ENCUT  =' in line:
            ENCUT=float(line.split()[2])
            INCAR.update({"ENCUT": ENCUT})  
        #Ionic relaxation:
        if 'IBRION =' in line:
            IBRION =int(line.split()[2])
            INCAR.update({"IBRION": IBRION})
        if 'NSW    =' in line:
            NSW =int(line.split()[2])
            INCAR.update({"NSW": NSW})
        if 'NBLOCK    =' in line:
            NBLOCK =int(line.split()[2])
            INCAR.update({"NBLOCK": NBLOCK})
        #Linear response parameters:
        if 'LEPSILON=' in line:
            LEPSILON=line.split()[1]
            INCAR.update({"LEPSILON": LEPSILON})
        #Write flags:
        if 'LVTOT        =' in line:
            LVTOT=line.split()[2]
            INCAR.update({"LVTOT": LVTOT})
        #Basic info:
        if 'NIONS =' in line:
            NIONS = int(line.split()[-1])
            INCAR.update({"NIONS": NIONS})
        if 'NELECT =' in line:
            NELECT = float(line.split()[2])
            INCAR.update({"NELECT": NELECT})
        if 'NBANDS=' in line:
            NBANDS= int(line.split()[-1])
            NKPTS=int(line.split()[3])
            INCAR.update({"NBANDS": NBANDS})
            INCAR.update({"NKPTS": NKPTS})
        if 'mass and valenz' in line:
            ZVAL.append(float(line.split()[5]))
        if 'TITEL  =' in line:
            POTCAR.append(line.split()[3])
    INCAR.update({"ZVAL": ZVAL})
    INCAR.update({"POTCAR": POTCAR})         
    return  INCAR

def split_line(line):
    out=[]
    wlist=line.split(' ')
    for word in wlist:
        if len(word)>0 and word != ' ' :
            if word[-1]== '\n':
                if len(word[:-1])>0 and word[:-1]!='':
                    out.append(word[:-1])
            else:
                out.append(word)
    return out 

"Read OUTCAR and KPOINTS to get BAND info"
def get_bandInfo(fname='OUTCAR'):
    'This part is copied from pyband: https://github.com/QijingZheng/pyband'
    outcar = [line for line in open( fname) if line.strip()]
    for ii, line in enumerate(outcar):
        if 'NKPTS =' in line:
            nkpts = int(line.split()[3])
            nband = int(line.split()[-1])
        if 'ISPIN  =' in line:
            ispin = int(line.split()[2])
        if "k-points in reciprocal lattice and weights" in line:
            Lvkpts = ii + 1
        if 'reciprocal lattice vectors' in line:
            ibasis = ii + 1
        if 'E-fermi' in line:
            Efermi = float(line.split()[2])
            LineEfermi = ii + 3
            # break
    B = np.array([line.split()[-3:] for line in outcar[ibasis:ibasis+3]], dtype=float)
    # k-points vectors and weights
    tmp = np.array([line.split() for line in outcar[Lvkpts:Lvkpts+nkpts]], dtype=float)
    vkpts = tmp[:,:3]
    wkpts = tmp[:,-1]
    # for ispin = 2, there are two extra lines "spin component..."
    N = (nband + 2) * nkpts * ispin + (ispin - 1) * 2
    bands = []
    # vkpts = []
    for line in outcar[LineEfermi:LineEfermi + N-1]:
        if 'spin component' in line or 'band No.' in line:
            continue
        if 'k-point' in line:
            # vkpts += [line.split()[3:]]
            continue
        #print (line)
        bands.append(float(line.split()[1]))

    bands = np.array(bands, dtype=float).reshape((ispin, nkpts, nband))

    if os.path.isfile('KPOINTS'):
        kp = open('KPOINTS').readlines()

    if os.path.isfile('KPOINTS') and kp[2][0].upper() == 'L':
        Nk_in_seg = int(kp[1].split()[0])
        Nseg = nkpts // Nk_in_seg
        vkpt_diff = np.zeros_like(vkpts, dtype=float)
        
        for ii in range(Nseg):
            start = ii * Nk_in_seg
            end = (ii + 1) * Nk_in_seg
            vkpt_diff[start:end, :] = vkpts[start:end,:] - vkpts[start,:]

        kpt_path = np.linalg.norm(np.dot(vkpt_diff, B), axis=1)
        # kpt_path = np.sqrt(np.sum(np.dot(vkpt_diff, B)**2, axis=1))
        for ii in range(1, Nseg):
            start = ii * Nk_in_seg
            end = (ii + 1) * Nk_in_seg
            kpt_path[start:end] += kpt_path[start-1]

        # kpt_path /= kpt_path[-1]
        kpt_bounds =  np.concatenate((kpt_path[0::Nk_in_seg], [kpt_path[-1],]))
    else:
        # get band path
        vkpt_diff = np.diff(vkpts, axis=0)
        kpt_path = np.zeros(nkpts, dtype=float)
        kpt_path[1:] = np.cumsum(np.linalg.norm(np.dot(vkpt_diff, B), axis=1))
        # kpt_path /= kpt_path[-1]

        # get boundaries of band path
        xx = np.diff(kpt_path)
        kpt_bounds = np.concatenate(([0.0,], kpt_path[1:][np.isclose(xx, 0.0)], [kpt_path[-1],]))

    return kpt_path, bands, Efermi, kpt_bounds

def read_total_energy(fname='OUTCAR'):
    #incar=get_INCARInfo()
    outcar = [line for line in open( fname) if line.strip()]
    for ii, line in enumerate(outcar):
        if 'free  energy   TOTEN' in line:
            total_energy = float(line.split()[4])
            # break
    return total_energy
    
################################## VASP tensor##########################
def reformat_vasp_tensor_4th(tensor):
    '''
    INPUT: in vasp, it outputs tensor in a favor of "XX YY ZZ XY YZ ZX"
    OUTPUT: in Voigt_notation, it's xx, yy, zz, yz, xz, xy
    '''
    tensor=tensor[[0,1,2,4,5,3],:]
    tensor=tensor[:,[0,1,2,4,5,3]]
    return tensor

################################## Read OUTCAR to get elastic tensor##########################
def get_elastic_tensor(fname='OUTCAR'):
    '''
    in  Gpa
    elastic_tensor: total elastic_tensor
    elastic_tensor_s: ELASTIC MODULI CONTR FROM cell deformation (static)
    elastic_tensor_i: ELASTIC MODULI CONTR FROM IONIC RELAXATION
    '''
    incar=get_INCARInfo()
    if incar['LEPSILON']=='F':
        raise RuntimeError('Need LEPSILON=T to get elastic trnsor')
    outcar = [line for line in open( fname) if line.strip()]
    for ii, line in enumerate(outcar):
        if 'SYMMETRIZED ELASTIC MODULI (kBar)' in line:
            Lela_s=ii+3
        if "ELASTIC MODULI CONTR FROM IONIC RELAXATION (kBar)" in line:
            Lela_i=ii+3
        if 'TOTAL ELASTIC MODULI (kBar)' in line:
            Lela_t=ii+3
    elastic_tensor_s = []
    elastic_tensor_i = []
    elastic_tensor = []
    
    for line in outcar[Lela_s:Lela_s+6]:
        lst= split_line(line)
        elastic_tensor_s.append([float(i) for i in lst[1:]])
    for line in outcar[Lela_i:Lela_i+6]:
         lst= split_line(line)
         elastic_tensor_i.append([float(i) for i in lst[1:]])
    for line in outcar[Lela_t:Lela_t+6]:
        lst= split_line(line)
        elastic_tensor.append([float(i) for i in lst[1:]])         
    elastic_tensor= reformat_vasp_tensor_4th(np.array(elastic_tensor)/10)
    elastic_tensor_s= reformat_vasp_tensor_4th(np.array(elastic_tensor_s)/10)
    elastic_tensor_i= reformat_vasp_tensor_4th(np.array(elastic_tensor_i)/10)
    
    return  elastic_tensor,   elastic_tensor_s ,  elastic_tensor_i  #in Gpa

################################## Read OUTCAR to get piezoelectric tensor##########################
def get_piezoelectric_tensor(fname='OUTCAR'):
    '''
    in  C/m^2
    piezoelectric_tensor: total piezoelectric_tensor
    piezoelectric_tensor_s: PIEZOELECTRIC MODULI CONTR FROM cell deformation (static)
    piezoelectric_tensor_i: PIEZOELECTRIC MODULI CONTR FROM IONIC RELAXATION   
    '''
    incar=get_INCARInfo()
    if incar['LEPSILON']=='F':
        raise RuntimeError('Need LEPSILON=T to get piezoelectric trnsor')
    outcar = [line for line in open(fname) if line.strip()]
    for ii, line in enumerate(outcar):
        if 'PIEZOELECTRIC TENSOR (including local field effects)  for field in x, y, z        (C/m^2)' in line:
            Lpiez_s=ii+3
        if "PIEZOELECTRIC TENSOR IONIC CONTR  for field in x, y, z        (C/m^2)" in line:
            Lpiez_i=ii+3
    piezoelectric_tensor_s = []
    piezoelectric_tensor_i = []
    for line in outcar[Lpiez_s:Lpiez_s+3]:
        lst= split_line(line)
        piezoelectric_tensor_s.append([float(i) for i in lst[1:]])
    for line in outcar[Lpiez_i:Lpiez_i+3]:
         lst= split_line(line)
         piezoelectric_tensor_i.append([float(i) for i in lst[1:]])     
    piezoelectric_tensor_s=np.array(piezoelectric_tensor_s)[:,[0,1,2,4,5,3]]
    piezoelectric_tensor_i=np.array(piezoelectric_tensor_i)[:,[0,1,2,4,5,3]]
    piezoelectric_tensor=piezoelectric_tensor_s+piezoelectric_tensor_i
    
    if abs(np.amax(piezoelectric_tensor))<abs(np.amin(piezoelectric_tensor)):
        piezoelectric_tensor=-1.0*piezoelectric_tensor
        piezoelectric_tensor_s=-1.0*piezoelectric_tensor_s
        piezoelectric_tensor_i=-1.0*piezoelectric_tensor_i
    
    return  piezoelectric_tensor,  piezoelectric_tensor_s,  piezoelectric_tensor_i #C/m^2

################################## Read OUTCAR to get dielectric tensor##########################
def get_dielectric_tensor(fname='OUTCAR'):
    
    '''
    in  dimensionless
    dielectric_tensor: total dielectric_tensor
    dielectric_tensor_s: MACROSCOPIC STATIC DIELECTRIC TENSOR (electronic)
    dielectric_tensor_i: MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION  
    '''
    incar=get_INCARInfo()
    if incar['LEPSILON']=='F':
        raise RuntimeError('Need LEPSILON=T to get dielectric trnsor')
    outcar = [line for line in open(fname) if line.strip()]
    for ii, line in enumerate(outcar):
        if 'MACROSCOPIC STATIC DIELECTRIC TENSOR (including local field effects in DFT)' in line:
            Ldiele_s=ii+2
        if "MACROSCOPIC STATIC DIELECTRIC TENSOR IONIC CONTRIBUTION" in line:
            Ldiele_i=ii+2
    dielectric_tensor_s = []
    dielectric_tensor_i = []
    line =outcar[Ldiele_s]
    lst= split_line(line)
    eps=[float(i) for i in lst[0:3]]
    if eps[0]>10000:
        dielectric_tensor=dielectric_tensor_s=dielectric_tensor_i=np.array([[10000.0, 0,0], [0, 10000.0, 0], [0,0,10000.0]])
    else:
        for line in outcar[Ldiele_s:Ldiele_s+3]:
            lst= split_line(line)
            dielectric_tensor_s.append([float(i) for i in lst[0:3]])
        for line in outcar[Ldiele_i:Ldiele_i+3]:
            lst= split_line(line)
            dielectric_tensor_i.append([float(i) for i in lst[0:3]])
        dielectric_tensor=np.array(dielectric_tensor_s)+np.array(dielectric_tensor_i)
    return  dielectric_tensor, np.array(dielectric_tensor_s),  np.array(dielectric_tensor_i) 

################################## Read OUTCAR to get BORN EFFECTIVE CHARGES tensor##########################
def get_born_effective_charge_tensor( fname='OUTCAR', NIONS=1):
    '''
    (in e, cummulative output)

    '''
    incar=get_INCARInfo()
    if incar['LEPSILON']=='F':
        raise RuntimeError('Need LEPSILON=T to get born_effective_charge_tensor')
    outcar = [line for line in open(fname) if line.strip()]
    for ii, line in enumerate(outcar):
        if 'BORN EFFECTIVE CHARGES (including local field effects)' in line:
            Lbec=ii+2
    born_effective_charge=[]
    Lend=Lbec+NIONS*4;i=0;tensor=[]
    for line in outcar[Lbec:Lend]:
        i=i+1
        if i%4 != 1:
            lst= split_line(line)
            tensor.append([float(i) for i in lst[1:4]])
            if i%4==0:
                born_effective_charge.append(tensor)
                tensor=[]
    born_effective_charge=np.array(born_effective_charge)
    return  born_effective_charge
    
################################## Read OUTCAR to get phonon frequency##########################
def read_frequency(fname='OUTCAR', NIONS=1):
    incar=get_INCARInfo()
    if incar['LEPSILON']=='F':
        raise RuntimeError('Need LEPSILON=T to get DFPT phonon frequency')
    outcar = [line for line in open(fname) if line.strip()]
    for ii, line in enumerate(outcar):
        if 'Eigenvectors and eigenvalues of the dynamical matrix' in line:
            Lfreq=ii+2
    frequency=[]
    Lend=Lfreq+NIONS*3*(NIONS+2);
    for line in outcar[Lfreq:Lend]:
        if "meV" in line:
            lst= split_line(line)
            if "f/i=" in line:
                frequency.append(round(-1.0*float(lst[-8]), 4))
            else:
                frequency.append(round(float(lst[-8]), 4))
    return np.array(frequency) #Thz

################################## Read OUTCAR to get the magnetization of ions ##########################
def read_magnetization(fname='OUTCAR', NIONS=1):
    incar=get_INCARInfo()
    if incar['ISPIN']!=2:
        raise RuntimeError('Need ISPIN=2 to the magnetization of ions')
    outcar = [line for line in open(fname) if line.strip()]
    for ii, line in enumerate(outcar):
        if 'magnetization (x)' in line:
            Lmag=ii+3
    magnetization=[]
    Lend=Lmag+NIONS;
    for line in outcar[Lmag:Lend]:
        lst= split_line(line)
        magnetization.append(round(float(lst[-1]),4))
    line=outcar[Lend+1]
    lst= split_line(line)
    total_magnetization=round(float(lst[-1]),4)
    return total_magnetization, np.array(magnetization) #miuB

def print_tensor(tensor):
    print('\n'.join([''.join(['{:^12.3f}'.format(item) for item in row])  for row in tensor]))

''' 
if __name__ == "__main__":
    total_energy=read_total_energy(fname='OUTCAR')

    INCAR=get_INCARInfo(fname = 'OUTCAR')
    if INCAR['ISPIN']==2:
        total_magnetization,magnetization=read_magnetization(fname='OUTCAR', NIONS=INCAR['NIONS'])
        
    if INCAR['LEPSILON']=='T':
        elastic_tensor,elastic_tensor_s,elastic_tensor_i=get_elastic_tensor(fname = 'OUTCAR')
        piezoelectric_tensor, piezoelectric_tensor_s, piezoelectric_tensor_i=get_piezoelectric_tensor(fname = 'OUTCAR')
        dielectric_tensor, dielectric_tensor_s, dielectric_tensor_i=get_dielectric_tensor(fname = 'OUTCAR')
        born_effective_charge=get_born_effective_charge_tensor(fname = 'OUTCAR', NIONS=INCAR['NIONS'])
        frequency=read_frequency(fname='OUTCAR', NIONS=INCAR['NIONS'])
        
        print_tensor(piezoelectric_tensor)
        print('\n')
        print_tensor(piezoelectric_tensor_s)
        print('\n')
        print_tensor(piezoelectric_tensor_i)
'''
        

    
    
    
    
    
    
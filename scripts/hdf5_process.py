import argparse
import yaml
import os

import numpy as np
from scipy import io
import matplotlib.pyplot as plt
import h5py
import cupy as cp
import cupy.fft as fft
import cupyx.scipy.ndimage as ndimage
import cupyx.scipy.signal as signal
import tables

# Credit for this class and the ePIE implementation in it: Matt Seaberg
class Ptycho:
    def __init__(self, datafile, metadata={}, plots=None, probe=None, mask=None, atten_area=None, use_probe=False):
        # check file type
        if datafile[-2:] == 'h5':
            # load in data, etc
            dat = tables.open_file(datafile).root
            # ipm2 is used for filtering and normalizing the data
            ipm2 = dat.ipm2.sum.read()
            # diffraction images (intensity, not amplitude)
            ims = dat.jungfrau1M.image_img#.ROI_0_area
            #ims = dat.jungfrau1M.ROI_0_area

            # piezo stage positions are in microns
            pi_x = dat.lmc.ch03.read()
            pi_y = dat.lmc.ch04.read()
            pi_z = dat.lmc.ch05.read()

            # Find indices where any of pi_x, pi_y, or pi_z is NaN
            #nan_indices = cp.isnan(pi_x) | cp.isnan(pi_y) | cp.isnan(pi_z)
            
            # vertical coordinate is always pi_z
            ycoords = -pi_z*1e-6

            if 'angle' in metadata.keys():
                angle = metadata['angle']
            else:
                angle = 0

            # horizontal coordinate may be a combination of pi_x and pi_y
            xcoords = (np.cos(np.radians(angle)) * pi_x + np.sin(np.radians(angle)) * pi_y) * 1e-6

            low_thresh = metadata['low_thresh']
            high_thresh = metadata['high_thresh']
            center_x = metadata['center_x']
            center_y = metadata['center_y']
            width = metadata['width']
            pD = metadata['pD']
            self.z = metadata['z']
            self.lambda0 = metadata['lambda0']
            im_thresh = metadata['im_thresh']

            N = int(center_y+width/2) - int(center_y-width/2)

            ipm_mask = np.logical_and.reduce((ipm2>low_thresh,ipm2<high_thresh,np.logical_not(np.isnan(xcoords)),
                                          np.logical_not(np.isnan(ycoords))))
#            ipm_mask = cp.asarray(np.logical_and.reduce((ipm2>low_thresh,ipm2<high_thresh,np.logical_not(np.isnan(xcoords)),
#                                          np.logical_not(np.isnan(ycoords)))) & (~nan_indices))
            diffraction = cp.zeros((N,N,int(cp.sum(ipm_mask))))
            for num, i in enumerate(ipm_mask.nonzero()[0]):
                diffraction[:,:,num] = cp.asarray(np.array(ims[i,int(center_y-width/2):int(center_y+width/2),
                                           int(center_x-width/2):int(center_x+width/2)]))
            # diffraction = cp.asarray(np.array(ims[mask][:,int(center_y-width/2):int(center_y+width/2),
            #                               int(center_x-width/2):int(center_x+width/2)]))

            if atten_area is not None and 'atten' in metadata.keys():
                multiplier = cp.ones((N,N))
                multiplier[atten_area.astype(bool)] = 1/metadata['atten']
                
                diffraction *= cp.reshape(multiplier,(N,N,1))
            
            diffraction[diffraction<im_thresh] = 0
            ipm2 = cp.asarray(ipm2)
            self.diffraction = cp.sqrt(diffraction / cp.reshape(ipm2[ipm_mask],(1,1,int(cp.sum(ipm_mask)))))
            xcoords = cp.asarray(xcoords[ipm_mask])
            ycoords = cp.asarray(ycoords[ipm_mask])
            
            
            print(xcoords)
            self.bin = 1
            self.probeGuess = probe

        elif datafile[-3:] == 'mat':
            data = io.loadmat(datafile)
            self.diffraction = cp.asarray(data['diffraction'])
            xcoords = cp.asarray(data['xGrid'])
            ycoords = cp.asarray(data['yGrid'])
            self.z = cp.asarray(data['z']).reshape(1)
            self.lambda0 = cp.asarray(data['lambdas']).reshape(1)
            self.bin = cp.asarray(data['bin']).reshape(1)
            self.probeGuess = probe
            if self.probeGuess is None and use_probe==True:
                self.probeGuess = cp.asarray(data['probes'])
            if data['pD'] is None:
                pD = 13.5e-6
            else:
                pD = data['pD']
        else:
            print('incorrect data type')
            return None

        self.pD = pD*self.bin

        # get image sizes
        self.N, self.M, self.P = cp.shape(self.diffraction)

        xcoords = xcoords.reshape(self.P)
        ycoords = ycoords.reshape(self.P)

        # generic coordinates
        x1 = cp.linspace(-self.M/2,self.M/2-1,self.M)
        y1 = cp.linspace(-self.N/2,self.N/2-1,self.N)
        self.x1, self.y1 = cp.meshgrid(x1,y1)

        # set up coordinates
        xD = cp.linspace(-self.M/2.,self.M/2.-1, self.M)*self.pD
        yD = cp.linspace(-self.N/2.,self.N/2.-1, self.N)*self.pD
        self.xD, self.yD = cp.meshgrid(xD,yD)

        # spatial frequencies at detector
        dfx = 1./(self.N*self.pD)
        fD = cp.linspace(-self.N/2.,self.N/2.-1, self.N)*dfx
        self.fDx, self.fDy = cp.meshgrid(fD,fD)

        # sample plane coordinates
        self.xS = self.fDx*self.lambda0*self.z
        self.yS = self.fDy*self.lambda0*self.z

        if self.probeGuess is None:
            self.probeGuess = cp.exp(-(self.xS**2+self.yS**2)/(cp.max(self.xS)/2)**2)*cp.exp(1j*2*np.pi/self.lambda0*200*(self.xS**2+self.yS**2))

        diff_sums = cp.sum(self.diffraction**2,axis=(0,1))

        # normalize probe guess to diffraction pattern
        self.probeGuess = self.probeGuess / cp.sqrt(
            cp.sum(cp.abs(self.probeGuess) ** 2) / cp.mean(diff_sums))

        self.initial_probe = cp.copy(self.probeGuess)

        # pixel size at sample
        self.dxS = dfx*self.lambda0*self.z

        # calculate scan positions in pixel units
        xcoords = xcoords/self.dxS
        ycoords = ycoords/self.dxS
        
        if 'xcoords' in metadata.keys():
            xcoords = cp.asarray(metadata['xcoords'])
        if 'ycoords' in metadata.keys():
            ycoords = cp.asarray(metadata['ycoords'])
            

        # center scan positions
        self.xcoords = xcoords - cp.min(xcoords)+self.N/2.
        self.ycoords = ycoords - cp.min(ycoords)+self.N/2.

        self.xMin = cp.zeros_like(self.xcoords)
        self.yMin = cp.zeros_like(self.xcoords)
        self.xMax = cp.zeros_like(self.xcoords)
        self.yMax = cp.zeros_like(self.xcoords)

        # set up object guess based on probe positions/probe
        self.generateObject()

        self.objectGuess = cp.ones(self.objectGuess.shape,dtype=complex)

        # initial scan positions
        self.xcoords1 = cp.copy(self.xcoords)
        self.ycoords1 = cp.copy(self.ycoords)

        # initialize arrays for position correction correlations
        self.shiftXOld = cp.zeros(self.P)
        self.shiftYOld = cp.zeros(self.P)
        self.shiftXNew = cp.zeros(self.P)
        self.shiftYNew = cp.zeros(self.P)

        # initialize alpha and beta (ePIE feedback parameters)
        self.alphaStart = 1.5
        self.beta = 1.5
        self.alpha = self.alphaStart

        # initialize position correction parameters
        self.gamma = 200.0*0
        self.scale = 1000.0

        # flag for probe retrieval
        self.flag = 1

        # flag for object retrieval
        self.flag2 = 1

        # initialize object guess
        self.objectGuess1 = cp.copy(self.objectGuess)

        if mask is None:
            self.mask = cp.ones((self.M,self.N), dtype=bool)
        else:
            self.mask = mask.astype(bool)

        self.inverseMask = cp.invert(self.mask)

        self.psi = cp.zeros_like(self.probeGuess)

    def reset_object(self):
        self.objectGuess = cp.ones(self.objectGuess.shape, dtype=complex)

    def reset_probe(self):
        self.probeGuess = cp.copy(self.initial_probe)

    def ptychography_step(self,index):
        amplitude = self.diffraction[:,:,index]

        # corner of subgrid to remove from object
        xMin = self.xcoords[index] - self.M/2.
        yMin = self.ycoords[index] - self.N/2.

        # get integer component
        xMinInt = int(cp.round(xMin))
        yMinInt = int(cp.round(yMin))

        # get subpixel component
        xSub = xMin - xMinInt
        ySub = yMin - yMinInt

        # take subobject from full object
        subObject = self.objectGuess[yMinInt:yMinInt+self.N,xMinInt:xMinInt + self.M]

        # define subpixel linear phase shift
        shiftPhase = cp.exp(-1j*2*np.pi*(xSub*self.x1/self.M + ySub*self.y1/self.N))

        # subpixel shift probe
        probeSubShift = Ptycho.infft(Ptycho.nfft(self.probeGuess)*shiftPhase)

        # multiply shifted probe with subobject to define psi
        subPsi = subObject*probeSubShift

        # FFT of psi
        fPsi = Ptycho.nfft(subPsi)

        # get error based on measured data
        errorOut = cp.sum(cp.abs(cp.abs(fPsi[self.mask])-amplitude[self.mask])**2)

        # apply modulus constraint
        fPsiTemp = amplitude*cp.exp(1j*cp.angle(fPsi))

        fPsiTemp[self.inverseMask] = fPsi[self.inverseMask]

        fPsi = cp.copy(fPsiTemp)

        # go back to object space (psi prime)
        subPsiPrime = Ptycho.infft(fPsi)

        # get maximum of probe
        probeDivide = cp.max(np.abs(probeSubShift))

        # update object
        subObjectNew = subObject + self.alpha * cp.conj(probeSubShift)/cp.abs(probeDivide)**2*(subPsiPrime-subPsi)

        objectGuess = cp.copy(self.objectGuess)

        # replace object section with updated object
        objectGuess[yMinInt:yMinInt+self.N,xMinInt:xMinInt+self.M] = subObjectNew

        # get maximum of object
        objectDivide = cp.max(cp.abs(subObject))

        # update probe
        probeOutShifted = probeSubShift + self.beta*cp.conj(subObject)/cp.abs(objectDivide)**2*(subPsiPrime-subPsi)

        # shift updated probe back
        probeGuess = Ptycho.infft(Ptycho.nfft(probeOutShifted)*cp.conj(shiftPhase))

        # calculate next guess for coordinates
        # find probe threshold mask at 10%
        mask = np.abs(probeSubShift) > 0.1*probeDivide

        # mask off object guesses
        subObject = subObject*mask
        subObjectNew = subObjectNew*mask

        # check if we're supposed to update position
        if self.gamma != 0:
            # do subpixel cross-correlation between new and old guesses
            shift = Ptycho.subPixelCorr(subObjectNew,subObject,self.scale)

            # get shift in pixels
            shift = self.gamma*shift

            # update current positions
            self.xcoords[index] = self.xcoords[index] - shift[0]
            self.ycoords[index] = self.ycoords[index] - shift[1]

        # check if we're supposed to update the probe
        if self.flag:
            self.probeGuess = probeGuess

        # check if we're supposed to update the object
        if self.flag2:
            self.objectGuess = objectGuess

        return errorOut

    def adjust_pc(self):
        # check scale

        if (cp.max(cp.abs(self.shiftXNew)) <= 2*self.gamma/self.scale
                and cp.max(cp.abs(self.shiftYNew)) <= 2*self.gamma/self.scale):
            self.scale *= 10
        elif (cp.mean(cp.abs(self.shiftXNew)) >= 5*self.gamma/self.scale or
            cp.mean(cp.abs(self.shiftYNew)) >= 5*self.gamma/self.scale):
            self.scale /= 10

        # check gamma
        shiftXCorr = signal.correlate(self.shiftXNew,self.shiftXOld,mode='valid')
        shiftYCorr = signal.correlate(self.shiftYNew,self.shiftYOld,mode='valid')

        aveCorr = (shiftXCorr + shiftYCorr)/2

        # adjust gamma depending on the sign of the correlation
        if aveCorr <0:
            self.gamma *= 0.9
        elif aveCorr > 0.3:
            self.gamma *= 1.1

        return aveCorr

    def fullePIEstep(self, indexOrder):

        errorOut = 0
        oldx = cp.copy(self.xcoords)
        oldy = cp.copy(self.ycoords)
        self.shiftXOld = cp.copy(self.shiftXNew)
        self.shiftYOld = cp.copy(self.shiftYNew)

        for i in range(self.P):

            errorTemp = self.ptychography_step(indexOrder[i])

            errorOut += errorTemp

        self.shiftXNew = self.xcoords-oldx
        self.shiftYNew = self.ycoords-oldy

        if self.gamma>0:
            ave_corr = self.adjust_pc()
        else:
            ave_corr = 0

        metadata = {
            'error': errorOut,
            'ave_corr': ave_corr
        }

        return metadata

    def run_iterations(self, recipe, status=False):

        num_iterations = recipe['N']
        turn_on_pc = recipe['pc_on']
        turn_off_pc = recipe['pc_off']
        gammaStart = recipe['gamma']
        alpha = recipe['alpha']
        beta = recipe['beta']
        turn_on_probe = recipe['probe_on']
        turn_off_probe = recipe['probe_off']
        reset_object = recipe['reset_object']
        if reset_object is None:
            reset_object = [-1]

        if turn_off_probe <= turn_on_probe:
            turn_off_probe = -1
        if turn_off_pc <= turn_on_pc:
            turn_off_pc = -1

        total_error = cp.zeros(num_iterations)
        corr_list = cp.zeros(num_iterations)
        scale_list = cp.zeros(num_iterations)
        gamma_list = cp.zeros(num_iterations)

        self.gamma = 0
        self.alpha = alpha
        self.beta = beta
        self.flag = 0

        res = {}

        for i in range(num_iterations):
            
            if status and np.mod(i,10) == 0:
                print(i)

            if i==turn_on_probe:
                self.flag = 1
            if i==turn_on_pc:
                self.gamma = gammaStart
            if i==turn_off_probe:
                self.flag = 0
            if i==turn_off_pc:
                self.gamma = 0

            if i in reset_object:
                self.reset_object()

            sorter = cp.random.random(self.P)
            ix = cp.argsort(sorter)

            metadata = self.fullePIEstep(ix)
            total_error[i] = metadata['error']
            corr_list[i] = metadata['ave_corr']
            scale_list[i] = self.scale
            gamma_list[i] = self.gamma

        res = {
            'total_error': total_error,
            'corr': corr_list,
            'scale': scale_list,
            'gamma': gamma_list
        }

        return res

    def generatePsi(self):

        for i in range(self.P):
            self.psi[:,:,i] = self.objectGuess[self.yMin[i]:self.yMax[i],self.xMin[i]:self.xMax[i]]*self.probeGuess

    def generateObject(self):

        # wiggle room for object (used for position correction).
        # just set to 5 pixels since that is what I've always used
        pixels = 20

        # get coordinate range
        xRange = cp.max(self.xcoords)-cp.min(self.xcoords) + self.M + 4*pixels
        yRange = cp.max(self.ycoords)-cp.min(self.ycoords) + self.N + 4*pixels

        # get final probe coordinates
        self.xcoords = self.xcoords - cp.min(self.xcoords) + self.M/2. + 2*pixels
        self.ycoords = self.ycoords - cp.min(self.ycoords) + self.N/2. + 2*pixels

        # initialize object
        self.objectGuess = cp.zeros((int(yRange),int(xRange)))

        self.N2, self.M2 = cp.shape(self.objectGuess)

        # make probe mask
        mask = Ptycho.new_support(self.probeGuess,0.1,1)
        #
        # populate object
        for i in range(self.P):

            self.xMin[i] = int(cp.round(self.xcoords[i]-self.M/2.))
            self.xMax[i] = int(cp.round(self.xcoords[i]+self.M/2.))
            self.yMin[i] = int(cp.round(self.ycoords[i]-self.N/2.))
            self.yMax[i] = int(cp.round(self.ycoords[i]+self.N/2.))
            self.objectGuess[int(self.yMin[i]):int(self.yMax[i]),int(self.xMin[i]):int(self.xMax[i])] = 1

    @staticmethod
    def new_support(imIn,threshold,width):

        supp = cp.zeros(imIn.shape)

        peak = cp.max(np.abs(imIn))

        supp[cp.abs(imIn)>peak*threshold] = 1

        # h = np.zeros((int(supp.size/2),int(supp.size/2)))
        N,M = cp.shape(supp)
        N = int(4*width)
        width = int(width/2)
        h = cp.zeros((N,N))
        h[int(N/2-width):int(N/2+width),int(N/2-width):int(N/2+width)] = 1

        supp = ndimage.convolve(supp,h)
        supp[supp>0.9] = 1

        return supp

    @staticmethod
    def nfft(a):
        """
        Class method for 2D FFT with zero frequency at center
        :param a: (N,M) ndarray
            array to be Fourier transformed
        :return: (N,M) ndarray
            Fourier transformed array of same shape as a
        """

        return fft.fftshift(fft.fft2(fft.ifftshift(a),norm='ortho'))

    @staticmethod
    def infft(a):
        """
        Class method for 2D IFFT with zero frequency at center
        :param a: (N,M) ndarray
            array to be inverse Fourier transformed
        :return: (N,M) ndarray
            Array after inverse Fourier transform, same shape as a
        """

        return fft.fftshift(fft.ifft2(fft.ifftshift(a),norm='ortho'))

    @staticmethod
    def subPixelCorr(im1,im2,scale):

        N,M = im1.shape

        # fourier transform images
        F = Ptycho.nfft(im1)
        G = Ptycho.nfft(im2)

        # start setting up cross-correlation
        FG = F*np.conj(G)

        # set up spatial and frequency coordinates
        x = cp.linspace(-10./scale,10./scale,21).reshape((1,21))
        y = cp.linspace(-10./scale,10./scale,21).reshape((21,1))
        u = cp.linspace(-.5+1./M*.5,.5-1./M*.5,M).reshape((M,1))
        v = cp.linspace(-.5+1./N*.5,.5-1./N*.5,N).reshape((1,N))

        # perform the inverse fourier transform to obtain zoomed-in correlation
        corr2 = cp.dot(cp.exp(1j*2*np.pi*cp.dot(y,v)), cp.dot(FG, cp.exp(1j*2*np.pi*cp.dot(u,x))))

        # find the peak of the correlation
        xmax2 = cp.argmax(cp.max(cp.abs(corr2),axis=0))
        ymax2 = cp.argmax(cp.max(cp.abs(corr2),axis=1))

        shift = cp.array([x[0,xmax2],y[ymax2,0]])

        return shift

def main():
    parser = argparse.ArgumentParser(description='xpp ptychography data loading and probe reconstruction script')
    
    # Required arguments
    parser.add_argument('experiment', type=str, help='Experiment name (e.g. xppd00120)')
    parser.add_argument('run', type=int, help='Run number (e.g. 1084)')
    
    # Optional arguments  
    parser.add_argument('-m', '--mode', type=str, choices=['load_only', 'load_recon'], default='load_only',
                        help='Script mode: "load_only" (default) or "load_recon"')
    parser.add_argument('-w', '--width', type=int, choices=[64, 128], default=64, 
                        help='Width parameter (default: 64)')
    parser.add_argument('-o', '--output_prefix', type=str, default='ptycho_output',
                        help='Output file prefix (default: "ptycho_output")')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='Path to YAML config file for metadata')
    
    # Metadata arguments (used if no config file provided)
    parser.add_argument('--low_thresh', type=int, default=40000, help='Low threshold (default: 40000)')
    parser.add_argument('--high_thresh', type=int, default=50000, help='High threshold (default: 50000)')  
    parser.add_argument('--center_x', type=int, default=150, help='Center X (default: 150)')
    parser.add_argument('--center_y', type=int, default=150, help='Center Y (default: 150)')
    parser.add_argument('--im_thresh', type=int, default=20, help='Image threshold (default: 20)')
    parser.add_argument('--z', type=float, default=4.147, help='Z value (default: 4.147)')
    parser.add_argument('--lambda0', type=float, default=1239.8/8889*1e-9, help='Lambda0 value (default: 1239.8/8889*1e-9)')
    parser.add_argument('--pD', type=float, default=75e-6, help='pD value (default: 75e-6)')  
    parser.add_argument('--angle', type=int, default=180, help='Angle (default: 180)')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Construct HDF5 file path  
    input_file_prefix = '/cds/data/psdm/xpp'
    fName = f'{args.experiment}_Run{args.run}.h5'
    filename = os.path.join(input_file_prefix, args.experiment, 'hdf5', 'smalldata', fName)
    
    # Load metadata from YAML file or command line arguments
    if args.config:
        with open(args.config, 'r') as f:
            metadata = yaml.safe_load(f)
    else:
        metadata = {
            'low_thresh': args.low_thresh,
            'high_thresh': args.high_thresh,
            'center_x': args.center_x, 
            'center_y': args.center_y,
            'width': args.width,
            'im_thresh': args.im_thresh,
            'z': args.z,
            'lambda0': args.lambda0,
            'pD': args.pD,
            'angle': args.angle
        }
    
    # Initialize Ptycho object  
    obj = Ptycho(filename, metadata=metadata)

    # Scale coordinates if width is 64
    if args.width == 64:
        obj.xcoords *= 0.5
        obj.xcoords1 *= 0.5  
        obj.ycoords *= 0.5
        obj.ycoords1 *= 0.5
    
    output_file = f'{args.output_prefix}_w{args.width}.npz'
    
    if args.mode == 'load_only':
        # Save specified arrays to NPZ file
        np.savez(output_file, 
                 diffraction=cp.asnumpy(obj.diffraction),
                 xcoords=cp.asnumpy(obj.xcoords),
                 ycoords=cp.asnumpy(obj.ycoords),
                 xcoords_start=cp.asnumpy(obj.xcoords1),
                 ycoords_start=cp.asnumpy(obj.ycoords1))
        
    elif args.mode == 'load_recon':
        # Run reconstruction with specified recipe
        recipe = {
            'N': args.iterations,
            'pc_on': args.pc_on,  
            'pc_off': args.pc_off,
            'probe_on': args.probe_on,
            'probe_off': args.probe_off,  
            'gamma': args.gamma,
            'alpha': args.alpha,
            'beta': args.beta,
            'reset_object': args.reset_object
        }
        
        res = obj.run_iterations(recipe, status=True)
        
        # Save specified arrays and reconstruction results to NPZ file  
        np.savez(output_file,
                 diffraction=cp.asnumpy(obj.diffraction),
                 probeGuess=cp.asnumpy(obj.probeGuess), 
                 objectGuess=cp.asnumpy(obj.objectGuess),
                 xcoords=cp.asnumpy(obj.xcoords),
                 ycoords=cp.asnumpy(obj.ycoords),
                 xcoords_start=cp.asnumpy(obj.xcoords1),
                 ycoords_start=cp.asnumpy(obj.ycoords1))
                 
    print(f'Output saved to {output_file}')
        
if __name__ == '__main__':
    main()

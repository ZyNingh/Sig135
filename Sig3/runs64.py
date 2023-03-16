import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from matplotlib import pyplot as PLT
from bilevelmri.experiment_setup import learn, compute_statistics
from bilevelmri.linear_ops.gradients import Grad
from bilevelmri.linear_ops.wavelets import Wavelet
from bilevelmri.functionals import Smoothed1Norm
from bilevelmri.loss_functions import least_squares
from bilevelmri.penalty_functions import l1_disc_penalty
from bilevelmri.parametrisations import alpha_parametrisation, free_parametrisation
import torch
import numpy as np
torch.set_default_dtype(torch.float64)
import SimpleITK as sitk
import skimage.io as io
import imageio
from scipy import fftpack
from multiprocessing import cpu_count
from pytorch_wavelets import DWTInverse, DWTForward
import sys
import requests


cpu_num = cpu_count() 
os.environ ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

SigOpt = int(sys.argv[1]) # 0 - 5

a = torch.zeros([6,1,64,64])
dwt = DWTForward(J=3, wave='db4', mode='per')
idwt = DWTInverse(wave='db4', mode='per')
rel, rell = dwt(a)
indsig = 0
#
difloc = [4]
rel[indsig,0,3,difloc] += 6
indsig += 1
#
difloc = [16]
rell[0][indsig,0,0,10,difloc] += 6
indsig += 1
#
difloc = [16]
rell[0][indsig,0,1,20,difloc] += 6
indsig += 1
#
difloc = [16]
rell[0][indsig,0,2,30,difloc] += 6
indsig += 1
#
difloc = [8]
rell[1][indsig,0,0,7,difloc] += 6
indsig += 1
#
difloc = [4]
rell[2][indsig,0,2,3,difloc] += 6

#

rec = idwt((rel, rell))

for BETACO in [0.00001,0.0001,0.001,0.01,0.1]:

    x = torch.zeros(1,64,64,2)

    inp = rec[SigOpt,0]
    MavVl = float(torch.max(inp))
    rawImage = inp / MavVl
    x[0,:,:,0] = rawImage

    y = torch.fft(x, signal_ndim=2, normalized=True) + 0.03 * torch.randn_like(x)
    data = {'x': x, 'y': y}
    n1, n2 = x.shape[1:3]

    params = {
        'model': {
            'n1': n1,
            'n2': n2
        },
        'alg_params': {
            'll_sol': {
                'maxit': 1000,
                'tol': 1e-10
            },
            'lin_sys': {
                'maxit': 1000,
                'tol': 1e-6
            },
            'LBFGSB': {
                'maxit': 2500,
                'pgtol': 4e-8
            }
        }
    }

    A = Wavelet()
    reg_func = Smoothed1Norm(gamma=1e-2)


    def penalty(p):
        return l1_disc_penalty(p[:-2], beta=(BETACO, BETACO))

    # tune alpha on full sampling pattern to get initialisation
    tuned_alpha = learn(data, 1e-3, [(0, np.inf)], alpha_parametrisation, A,
                        reg_func, least_squares, lambda p: torch.zeros_like(p),
                        params)

    p_init = np.ones(n1 * n2 + 2)
    p_init[-1] = 1e-2
    p_init[-2] = tuned_alpha['p']
    p_bounds = [(0., 1.) for _ in range(n1 * n2)]
    p_bounds.append((0, np.inf))
    p_bounds.append((1e-2, 1e-2))
    # learn sampling pattern
    result = learn(data, p_init, p_bounds, free_parametrisation, A, reg_func,
                least_squares, penalty, params)

    stats = compute_statistics(data, result['p'], A, reg_func, free_parametrisation, params)

    imageio.imwrite("Sampling" + "_beta" + str(BETACO) + ".png",fftpack.fftshift(result['p'][:-2].reshape(n1, n2)))


    imageio.imwrite("RawB_sig" + str(SigOpt) + ".png",torch.sqrt(torch.sum(data['x'][0, :, :, :]**2, dim=2)))
    imageio.imwrite("RecB_sig" + str(SigOpt) + ".png",torch.sqrt(torch.sum(stats['recons'][0, :, :, :]**2, dim=2)))

    print(str(result))
    print('\n')
    print(str(stats))

    fou = open('res.txt','a')
    fou.write("###ITERATION" + " beta" + str(BETACO) )
    fou.write(str(result))
    fou.write('\n')
    fou.write(str(stats))
    
    op = open(".//beta"+str( BETACO)+'//res.txt','a')
    op.write(str(BETACO) + '\n' + str(100 * np.mean(result['p']>0)))
    op.write('\n' + str(result))
    op.write('\n'+ str(stats))
    op.close()


    TOKEN  = "6071613273:AAEDCA5RLBtshqbCSSalxPF1KCgeEBgsfLs"
    rate = 100 * np.mean(result['p']>0)
    chat_id = "1189489886"
    message = "Sig " + str(SigOpt) + "with beta " + str(BETACO) + "\n" + "result" + str(result) + "\nrate" + str(rate)

    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"

    print(requests.get(url).json())

message = "FINISHED, check!"

url = f"https://api.telegram.org/bot{TOKEN}/sendMessage?chat_id={chat_id}&text={message}"

print(requests.get(url).json())

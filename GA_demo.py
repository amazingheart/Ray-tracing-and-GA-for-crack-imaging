import numpy as np
from scipy.signal import hilbert, detrend
from geneticalgorithm import geneticalgorithm as ga
import matplotlib.pyplot as plt
import time


def denoise(x):
    x = detrend(x)
    fourier = np.fft.rfft(x, n=5000)
    frequencies = np.fft.rfftfreq(5000, delta_t)
    fourier[frequencies > 1e5] = 0
    denoised = np.fft.irfft(fourier)
    return denoised[:len(x)]


def cal_indices(x):
    x_ = np.int_(x * (N-1))
    len_x = abs(x_[2] - x_[0]) + 1
    len_y = abs(x_[3] - x_[1]) + 1
    length = max(len_x, len_y)
    if length == 1:
        loc = (x_[:2])[np.newaxis, :]
    else:
        loc_x = np.int_(np.arange(length) * (x_[2] - x_[0]) / (length - 1)) + x_[0]
        loc_y = np.int_(np.arange(length) * (x_[3] - x_[1]) / (length - 1)) + x_[1]
        loc = np.stack((loc_x, loc_y)).T
    indices = loc[:, 0] + loc[:, 1] * N
    return indices


def norm_env(x):
    env = abs(hilbert(x))
    env /= np.linalg.norm(env)
    return env


def f(x):  # x = [x1, y1, x2, y2]
    s = np.sum(signal[:, cal_indices(x)], axis=-1)
    result = norm_env(s)
    error = np.linalg.norm(result - res)

    return error


def img_err(x, x_):  # x = [x1, y1, x2, y2]
    return np.linalg.norm(x[:2]-x_[:2])+np.linalg.norm(x[2:]-x_[2:])


def corr_coef(x, y):
    x -= np.mean(x)
    y -= np.mean(y)
    return x.dot(y) / np.linalg.norm(x) / np.linalg.norm(y)


##################################################################
# 'sim1' for the vertical crack in the center with a length of 50 mm
# 'rough_p3' for the 'sim1' crack with a roughness of 0.3 mm added
# 'rough_p4' for the 'sim1' crack with a roughness of 0.4 mm added
# 'rough_p5' for the 'sim1' crack with a roughness of 0.5 mm added
crack_name = 'sim1'
##################################################################

delta_t = 5e-7
signal = np.load('Dictionary_sim.npy')
M = int(signal.shape[0]/7)

N = int(np.sqrt(signal.shape[1]))
bl = np.genfromtxt('sim_bl.csv', delimiter=',')[::10, 1:].T
crk = np.genfromtxt('%s.csv' % crack_name, delimiter=',')[::10, 1:].T

res = norm_env(np.ravel(crk - bl))

varbound = np.array([[0, 1]] * 4)

truth = np.array([50, 50-25, 50, 50+25])/100

pred = np.zeros(4)
min_err = np.inf

for i in range(7):
    start = time.time()

    model = ga(function=f, dimension=4, variable_type='real', variable_boundaries=varbound)

    model.run()

    convergence = model.report
    solution = model.output_dict

    pred_ = np.array(solution['variable']).ravel()

    end = time.time()
    print('Computing time %s s' % (end - start))

    err = img_err(pred_, truth)
    if err < min_err:
        pred = pred_
        min_err = err

np.save('Pred_%s.npy' % crack_name, pred)

pred = np.load('Pred_%s.npy' % crack_name)

print('Imaging error: ', img_err(truth, pred))

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
px = 1 / plt.rcParams['figure.dpi']
plt.rcParams['figure.figsize'] = [550 * px, 550 * px]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 7
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

plt.plot(truth[[0, 2]], truth[[1, 3]], 'k', label='Truth')
plt.plot(pred[[0, 2]], pred[[1, 3]], 'r', label='Prediction')
plt.xlim(0, 1)
plt.xticks([0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100])
plt.xlabel('x [mm]')
plt.ylim(0, 1)
plt.yticks([0, 0.25, 0.5, 0.75, 1], [0, 25, 50, 75, 100], rotation=90)
plt.ylabel('y [mm]', rotation=90)
plt.legend()
plt.tight_layout()
plt.show()

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
px = 1 / plt.rcParams['figure.dpi']
plt.rcParams['figure.figsize'] = [550 * px, 1241 * px]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 7
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['legend.fontsize'] = 'medium'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

raw = np.sum(signal[:, cal_indices(pred)], axis=-1)
res_ = norm_env(raw)
subfigures = np.arange(711, 718)
sensors = [2, 4, 6, 1, 3, 5, 7]
for i in range(7):
    plt.subplot(subfigures[sensors[i]-1])
    plt.plot(np.arange(M)*delta_t*1e6, res[M*i:M*(i+1)], 'k', label='Measurement')
    plt.plot(np.arange(M)*delta_t*1e6, res_[M*i:M*(i+1)], 'r', label='Model')

    pcc = corr_coef(res[M*i:M*(i+1)], res_[M*i:M*(i+1)])

    if sensors[i] == 1:
        plt.legend()
    if i != 6:
        plt.xticks([], [])
    else:
        plt.xlabel(r'Time [$\mu$s]')
    plt.ylabel('Sensor %s' % sensors[i])
    plt.ylim(0, 0.06)
plt.tight_layout()
plt.savefig('%s.png' % crack_name)
plt.close()

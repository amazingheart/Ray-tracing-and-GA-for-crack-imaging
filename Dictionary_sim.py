import numpy as np
from scipy.optimize import fsolve
from scipy.signal import detrend
from sklearn.metrics import pairwise_distances
import cmath


class DispersionCurve:
    def __init__(self, f=2e5):
        # T = 293.15 K
        E = 6.9138e10
        nu = 0.33125
        rho = 2700.08
        lmbd = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))

        self.f = f
        self.d = plate_geometry[2]  # Plate thickness
        self.c_l = np.sqrt((lmbd + 2 * mu) / rho)  # Bulk longitudinal velocity
        self.c_t = np.sqrt(mu / rho)  # Bulk transverse velocity

    def fun(self, x):
        k = 2 * cmath.pi * self.f / x
        p = cmath.sqrt((2 * cmath.pi * self.f) ** 2 / self.c_l ** 2 - k ** 2)
        q = cmath.sqrt((2 * cmath.pi * self.f) ** 2 / self.c_t ** 2 - k ** 2)
        return (cmath.tan(q * self.d / 2) * q + (k ** 2 - q ** 2) ** 2 * cmath.tan(p * self.d / 2) / (4 * k ** 2 * p)).real


def velocities(frequencies):
    c = np.zeros(len(frequencies), float)
    for i in range(len(c)):
        tmp = DispersionCurve(f=frequencies[i])
        c[i] = fsolve(tmp.fun, np.array(2000.))
    return c


def excitation(fc=2e5, cycles=5, fs=1e7, samples=2e3):
    n = int(cycles * fs / fc) + 1
    times = np.arange(n) / fs

    raw_signal = np.sin(2 * np.pi * fc * times)
    hann_window = np.hanning(n)
    hann_signal = hann_window * raw_signal

    signal = np.append(hann_signal, np.zeros(int(samples - n)))

    fft = np.fft.rfft(signal)

    return fft


def denoise(x):
    x = detrend(x)
    fourier = np.fft.rfft(x, n=5000)
    frequencies = np.fft.rfftfreq(5000, delta_t)
    fourier[frequencies > 1e5] = 0
    denoised = np.fft.irfft(fourier)
    return denoised[:len(x)]


def pixel_locations_0():
    n = int(plate_geometry[0] / 3 / delta_x) + 1
    pixels = np.zeros((n, n), float)
    pixels += np.arange(n) * delta_x + plate_geometry[0] / 3
    locations = np.concatenate((pixels[:, :, np.newaxis], pixels.T[:, :, np.newaxis]), axis=2)
    return locations.reshape((-1, 2))


def pixel_locations_1(locations_0):
    mirror1 = locations_0 @ np.array([[-1, 0], [0, 1]])
    mirror2 = locations_0 @ np.array([[-1, 0], [0, 1]]) + np.array([2 * plate_geometry[0], 0])
    mirror3 = locations_0 @ np.array([[1, 0], [0, -1]])
    mirror4 = locations_0 @ np.array([[1, 0], [0, -1]]) + np.array([0, 2 * plate_geometry[0]])

    locations = np.stack((mirror1, mirror2, mirror3, mirror4), axis=0)

    return locations


def pixel_locations_2(locations_1):
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 0], [1, 2], [1, 3], [2, 3], [3, 2]])
    N = edges.shape[0]
    locations = np.zeros((N, locations_1.shape[1], 2), float)
    for i in range(N):
        if edges[i, 1] == 0:
            locations[i] = locations_1[edges[i, 0]] @ np.array([[-1, 0], [0, 1]])
        elif edges[i, 1] == 1:
            locations[i] = locations_1[edges[i, 0]] @ np.array([[-1, 0], [0, 1]]) + np.array([2 * plate_geometry[0], 0])
        elif edges[i, 1] == 2:
            locations[i] = locations_1[edges[i, 0]] @ np.array([[1, 0], [0, -1]])
        elif edges[i, 1] == 3:
            locations[i] = locations_1[edges[i, 0]] @ np.array([[1, 0], [0, -1]]) + np.array([0, 2 * plate_geometry[0]])

    return locations


def wave_propagation(d, p, a):
    f_domain = np.exp(-1j * p) * source[low_band:high_band]
    f_domain_full = np.zeros(len(freqs), complex)
    f_domain_full[low_band:high_band] = f_domain
    t_domain = np.fft.irfft(f_domain_full)
    t_domain *= np.array(a)

    tof = d / c_g
    tof_ = int(tof / delta_t)
    t_domain[:tof_] = 0

    return t_domain


def cal_res():
    locations_0 = pixel_locations_0()
    locations_1 = pixel_locations_1(locations_0)
    locations_2 = pixel_locations_2(locations_1)

    d0 = pairwise_distances(sensor_locations, locations_0)
    d1 = pairwise_distances(sensor_locations, locations_1.reshape((-1, 2))).reshape(
        (-1, locations_1.shape[0], locations_0.shape[0]))
    d2 = pairwise_distances(sensor_locations, locations_2.reshape((-1, 2))).reshape(
        (-1, locations_2.shape[0], locations_0.shape[0]))
    d0 -= (delta_d + crack_thick/2)
    d1 -= (delta_d + crack_thick/2)
    d2 -= (delta_d + crack_thick/2)

    signal = np.zeros((d0.shape[1], (d0.shape[0]-1)*length), float)
    # Pixels
    for i in range(d0.shape[1]):
        # Sensors
        for k in range(d0.shape[0]-1):
            # Actuator-Pixel, order 0
            if d0[0, i] < d_max:
                # Pixel-Sensor, order 0
                d = d0[k+1, i] + d0[0, i]
                if d < d_max:
                    phase = wavenumbers * d + np.pi
                    attenuation = 1/np.sqrt(d0[0, i] * d0[k+1, i])
                    signal[i, k*length:(k+1)*length] += wave_propagation(d, phase, attenuation)
                # Pixel-Sensor, order 1
                for m in range(d1.shape[1]):
                    d = d0[0, i] + d1[k+1, m, i]
                    if d < d_max:
                        phase = wavenumbers * d
                        attenuation = 1/np.sqrt(d0[0, i] * d1[k+1, m, i])
                        signal[i, k * length:(k + 1) * length] += wave_propagation(d, phase, attenuation)
                # Pixel-Sensor, order 2
                for m in range(d2.shape[1]):
                    d = d0[0, i] + d2[k+1, m, i]
                    if d < d_max:
                        phase = wavenumbers * d + np.pi
                        attenuation = 1/np.sqrt(d0[0, i] * d2[k+1, m, i])
                        signal[i, k * length:(k + 1) * length] += wave_propagation(d, phase, attenuation)
            # Actuator-Pixel, order 1
            for j in range(d1.shape[1]):
                if d1[0, j, i] < d_max:
                    # Pixel-Sensor, order 0
                    d = d1[0, j, i] + d0[k+1, i]
                    if d < d_max:
                        phase = wavenumbers * d
                        attenuation = 1/np.sqrt(d1[0, j, i] * d0[k+1, i])
                        signal[i, k * length:(k + 1) * length] += wave_propagation(d, phase, attenuation)
                    # Pixel-Sensor, order 1
                    for m in range(d1.shape[1]):
                        d = d1[0, j, i] + d1[k+1, m, i]
                        if d < d_max:
                            phase = wavenumbers * d + np.pi
                            attenuation = 1/np.sqrt(d1[0, j, i] * d1[k+1, m, i])
                            signal[i, k * length:(k + 1) * length] += wave_propagation(d, phase, attenuation)
                    # Pixel-Sensor, order 2
                    for m in range(d2.shape[1]):
                        d = d1[0, j, i] + d2[k+1, m, i]
                        if d < d_max:
                            phase = wavenumbers * d
                            attenuation = 1/np.sqrt(d1[0, j, i] * d2[k+1, m, i])
                            signal[i, k * length:(k + 1) * length] += wave_propagation(d, phase, attenuation)
            # Actuator-Pixel, order 2
            for j in range(d2.shape[1]):
                if d2[0, j, i] < d_max:
                    # Pixel-Sensor, order 0
                    d = d2[0, j, i] + d0[k+1, i]
                    if d < d_max:
                        phase = wavenumbers * d + np.pi
                        attenuation = 1/np.sqrt(d2[0, j, i] * d0[k+1, i])
                        signal[i, k * length:(k + 1) * length] += wave_propagation(d, phase, attenuation)
                    # Pixel-Sensor, order 1
                    for m in range(d1.shape[1]):
                        d = d2[0, j, i] + d1[k+1, m, i]
                        if d < d_max:
                            phase = wavenumbers * d
                            attenuation = 1/np.sqrt(d2[0, j, i] * d1[k+1, m, i])
                            signal[i, k * length:(k + 1) * length] += wave_propagation(d, phase, attenuation)
                    # Pixel-Sensor, order 2
                    for m in range(d2.shape[1]):
                        d = d2[0, j, i] + d2[k+1, m, i]
                        if d < d_max:
                            phase = wavenumbers * d + np.pi
                            attenuation = 1/np.sqrt(d2[0, j, i] + d2[k+1, m, i])
                            signal[i, k * length:(k + 1) * length] += wave_propagation(d, phase, attenuation)

    return signal


if __name__ == '__main__':
    plate_geometry = [0.3, 0.3, 1e-3]
    crack_thick = 2e-4
    delta_d = 0.5e-2
    delta_x = 1e-3
    delta_t = 5e-7
    delta_f = 2e3
    length = int(1 / (delta_t * delta_f))
    freqs = np.arange(length / 2 + 1) * delta_f
    central_freq = 5e4
    c_g = 1321.796
    low_band = int((central_freq - 2e4) / delta_f)
    high_band = int((central_freq + 2e4) / delta_f)
    d_max = length * delta_t * c_g

    source = -excitation(fc=central_freq, fs=int(1/delta_t), samples=length)
    v = velocities(freqs[low_band:high_band])
    wavenumbers = 2 * np.pi * freqs[low_band:high_band] / v

    L = plate_geometry[0]
    sensor_locations = np.array([[L/6, L/3],
                                 [L*2/3, L/6],
                                 [L*5/6, L*2/3],
                                 [L/3, L*5/6],
                                 [L/3, L/6],
                                 [L*5/6, L/3],
                                 [2*L/3, L*5/6],
                                 [L/6, 2*L/3]])

    dictionary = cal_res()

    np.save('Dictionary_sim.npy', dictionary.T)

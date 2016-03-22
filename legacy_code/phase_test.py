import numpy as np
import pylab as plt
import time

def create_uv(stations, frequency, hour_angle, declination):

    baselines = (len(stations)*(len(stations)-1))/2
    uv_points = np.empty((baselines,4))
    wavelength = 3.e8/frequency

    declination = declination*(np.pi/180)

    idx = 0

    for i in range(0,len(stations)-1):
        for j in range(i+1,len(stations)):
            d = (stations[j] - stations[i])/wavelength
            u = d*np.cos(hour_angle)
            v = d*np.sin(hour_angle)*np.sin(declination)
            uv_points[idx,0] = i
            uv_points[idx,1] = j
            uv_points[idx,2] = u
            uv_points[idx,3] = v
            idx += 1

    return uv_points


def create_model_vis(uv_points):

    visibilities = np.ones((np.shape(uv_points)[0],1), dtype=np.complex128)

    return visibilities


def create_obs_vis(uv_points, stations):

    visibilities = np.ones((np.shape(uv_points)[0],1),dtype=np.complex128)

    phase_errors = np.random.random([len(stations),1]).astype(np.complex128)

    for i in range(visibilities.shape[0]):

        ant_a = uv_points[i,0]
        ant_b = uv_points[i,1]

        ant_a_dphi = np.exp(-1j*phase_errors[ant_a])
        ant_b_dphi = np.exp(-1j*phase_errors[ant_b])

        visibilities[i] = ant_a_dphi*visibilities[i]*np.conj(ant_b_dphi)

    return visibilities, phase_errors


def jacobian(obs_vis, model_vis, stations, uv_points, phase_errors):

    J = np.zeros((2*np.shape(obs_vis)[0],len(stations)),dtype=np.complex128)

    for i in range(J.shape[0]/2):
        for j in range(len(stations)):
            ant_a = uv_points[i,0]
            ant_b = uv_points[i,1]

            ant_a_dphi = np.exp(-1j*phase_errors[ant_a])
            ant_b_dphi = np.exp(-1j*phase_errors[ant_b])

            if ant_a == j:
                eq_conj = np.conj(ant_b_dphi)[0]
                ep_deri = -1j*ant_a_dphi[0]
                J[i,j] += model_vis[i,0]*eq_conj*ep_deri

            if ant_b == j:
                ep = ant_a_dphi[0]
                eq_conj_deri = np.conj(-1j*ant_b_dphi)[0]
                J[i,j] += model_vis[i,0]*ep*eq_conj_deri

    J[np.shape(obs_vis)[0]:,:] = np.conj(J[:np.shape(obs_vis)[0],:])

    return J


def residual(obs_vis, model_vis, stations, uv_points, phase_errors):

    r = np.empty(obs_vis.shape, dtype=np.complex128)

    for i in range(r.shape[0]):

        ant_a = uv_points[i,0]
        ant_b = uv_points[i,1]

        ant_a_dphi = np.exp(-1j*phase_errors[ant_a])
        ant_b_dphi = np.exp(-1j*phase_errors[ant_b])

        r[i] = obs_vis[i] - ant_a_dphi*model_vis[i]*np.conj(ant_b_dphi)

    r = np.vstack((r,r.conj()))

    return r

def jhr(obs_vis, model_vis, stations, uv_points, phase_errors):

    r = np.empty(obs_vis.shape, dtype=np.complex128)

    for i in range(r.shape[0]):

        ant_a = uv_points[i,0]
        ant_b = uv_points[i,1]

        ant_a_dphi = np.exp(-1j*phase_errors[ant_a])
        ant_b_dphi = np.exp(-1j*phase_errors[ant_b])

        r[i] = obs_vis[i] - ant_a_dphi*model_vis[i]*ant_b_dphi.conj()

    JHR = np.zeros(len(stations), dtype=np.complex128)

    for k in range(len(stations)):
        for i in range(len(stations)):

            ant_a = k
            ant_b = i

            if ant_a==ant_b:
                continue

            if ant_a<ant_b:
                baseline = recsum(len(stations)-1, ant_a) + ant_b - 1
            elif ant_b<ant_a:
                baseline = recsum(len(stations)-1, ant_b) + ant_a - 1

            ant_a_dphi = np.exp(-1j*phase_errors[ant_a])[0]
            ant_b_dphi = np.exp(-1j*phase_errors[ant_b])[0]

            res = obs_vis[baseline,0]

            if ant_a<ant_b:
                JHR[k] += ant_a_dphi.conj()*model_vis[baseline,0].conj()*\
                    ant_b_dphi*res

            else:
                ant_a_dphi, ant_b_dphi = ant_b_dphi, ant_a_dphi
                JHR[k] += ant_a_dphi*model_vis[baseline,0]*ant_b_dphi.conj()*res.conj()

    return JHR

def recsum(a,b):

    sum = 0
    for i in range(a-b,a):
        sum += i
    return sum

def order_data(obs_vis, model_vis, stations, phase_solutions):

    mod_mat = np.ones([len(stations),len(stations)], dtype=np.complex128) \
            - np.diagflat(np.ones([len(stations),1]))
    vis_mat = np.zeros([len(stations),len(stations)],dtype=np.complex128)

    rownum = 0
    firste = 1

    start = 0
    stop = len(stations) - firste

    for i in range(vis_mat.shape[0]):

        vis_mat[rownum,firste:] = obs_vis[start:stop,0]
        vis_mat[firste:,rownum] = obs_vis[start:stop,0].conj()

        rownum+=1
        firste+=1
        start, stop = stop, stop + len(stations) - firste

    return vis_mat, mod_mat

def get_updates(vis_mat, mod_mat, stations, phase_solutions):

    updates = np.empty([len(stations),1])

    t0 = time.time()
    phase_solutions = phase_solutions.reshape([len(stations)])
    denom = np.sum(abs(mod_mat)**2, 1)

    for i in range(updates.shape[0]):
        gains = np.exp(-1j*phase_solutions)
        numer = (-gains[i].conj()*(vis_mat[i,:].dot(gains*mod_mat[:,i]))).imag
        # denom = mod_mat[:,i].T.conj().dot(mod_mat[:,i])
        # denom = np.sum(abs(mod_mat[:,i])**2)

        updates[i] = numer/denom[i]

    print time.time() - t0

    return updates

def gn_solver(obs_vis, model_vis, stations, uv_points, phase_errors):

    # Need to start with random initial guess seems odd, though may be a
    # property of the weird derivative/the fact that the solution is poorly
    # constrained in the single point source in the center of the field case.
    # Another point of interest is the rotational ambiguity - using perfect
    # gaussians means any pointing error at a given radius will minimise the
    # residual. This is less of an issue when there are multiple sources to
    # constrain the solution.

    # phase_solutions = phase_errors + ((np.random.random([len(stations),1]) - \
    #                                   0.5)/10).astype(np.complex128)
    # phase_solutions = (np.random.random([len(stations),1]) - 0.5).\
    #                     astype(np.complex128)
    phase_solutions = np.zeros([len(stations),1], dtype=np.complex128)

    improvement = np.inf
    accuracy = 1e-6

    r = residual(obs_vis, model_vis, stations, uv_points, phase_solutions)

    i = 1

    vis_mat, mod_mat = order_data(obs_vis,model_vis,stations,phase_solutions)

    while improvement > accuracy:

        # t0 = time.time()
        updates = get_updates(vis_mat, mod_mat,stations,phase_solutions)
        # print time.time() - t0

        t0 = time.time()
        J = jacobian(obs_vis, model_vis, stations, uv_points, phase_solutions)
        H = J.T.conj().dot(J)

        # print J.T.conj().shape
        # print np.vstack((-model_vis,-model_vis.conj())).shape
        # print J.T.conj().dot(np.vstack((-model_vis,-model_vis.conj())))
        # print J.T.conj().dot(np.vstack((-obs_vis,-obs_vis.conj())))

        # plt.imshow(abs(J),interpolation="none")
        # plt.show()
        # plt.imshow(abs(H),interpolation="none")
        # plt.show()
        # We can take the approximate inversion here but need to reduce the
        # step size in order to ensure convergence. Averaging multiple steps
        # works! Need to investigate why. Read the StEFCal paper again.

        H = np.diagflat(np.diagonal(H))

        # print -2*jhr(obs_vis, model_vis, stations, uv_points,
        #              phase_solutions).imag.reshape([len(stations),1])
        # print (J.T.conj()).dot(r).real

        if (i%2)==0:
            phase_solutions += np.linalg.pinv(H).dot((J.T.conj()).dot(r))/2
        else:
            phase_solutions += np.linalg.pinv(H).dot((J.T.conj()).dot(r))

        print "Normal time = ", time.time()-t0

        print np.allclose(np.linalg.pinv(H).dot((J.T.conj()).dot(r)).real,
                          updates)

        r = residual(obs_vis, model_vis, stations, uv_points, phase_solutions)

        print "GN NORM = ", np.linalg.norm(r)

        i += 1

        if i>10:
            print phase_errors
            # print np.exp(-1j*phase_solutions)
            # print phase_solutions[0] - phase_solutions[1]
            # print phase_errors[0] - phase_errors[1]
            # print abs(np.exp(-1j*phase_errors)), abs(np.exp(
            #     -1j*phase_solutions))

            # for yy in range(phase_errors.shape[0]):
            #     for zz in range(yy+1,phase_errors.shape[0]):
            #         print np.exp(-1j*phase_errors[yy])*np.conj(np.exp(
            #             -1j*phase_errors[
            #             zz]))
            #         print np.exp(-1j*phase_solutions[yy])*np.conj(np.exp(
            #             -1j*phase_solutions[zz]))

            # print phase_errors, phase_solutions

            break

    return J.T.conj().dot(J)

if __name__=="__main__":
    stations = [0,144,288,432,576,720,864,1008,1152,1296,1332,1440,2772,4176]
    frequency = 1.4e9

    uv_points = create_uv(stations,frequency, 0, 90)

    # print voltage_beam(0,0,1.,1.)
    # print voltage_derivative(0,0,1.,1.,"l")
    # print voltage_derivative(0,0,1.,1.,"m")

    model_vis = create_model_vis(uv_points)
    obs_vis, phase_errors = create_obs_vis(uv_points, stations)

    # phase_errors = phase_errors.reshape([-1,1])
    # J = jacobian(obs_vis,model_vis,stations, uv_points, pointing_errors)
    # plt.imshow(J,interpolation='none')
    # plt.show()

    # JHJ = lm_solver(obs_vis,model_vis,stations, uv_points, pointing_errors)
    JHJ = gn_solver(obs_vis,model_vis,stations, uv_points, phase_errors)

import numpy as np
from sympy.unify.core import unify_var
import pylab as plt
import time
from samba.dcerpc.samr import UserWorkStationsInformation


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


# def create_model_vis(uv_points):
#
#     visibilities = np.ones((np.shape(uv_points)[0],1))
#
#     return visibilities


def create_model_vis(sources, uv_points):

    point_sources = sources

    visibilities = np.zeros([uv_points.shape[0],point_sources.shape[0]],dtype=np.complex128)

    for i in range(point_sources.shape[0]):
        A_i = point_sources[i,0]
        l_i = point_sources[i,1]
        m_i = point_sources[i,2]
        visibilities[:,i] += A_i*np.exp(-2*np.pi*1j*
                                  (uv_points[:,2]*l_i+uv_points[:,3]*m_i))

    return visibilities


# def create_obs_vis(uv_points, stations):
# 
#     visibilities = np.ones((np.shape(uv_points)[0],1))
# 
#     pointing_errors = (np.random.random([2*len(stations),1]) - 0.5)*2
# 
#     offset = len(stations)
# 
#     for i in range(visibilities.shape[0]):
# 
#         ant_a = uv_points[i,0]
#         ant_b = uv_points[i,1]
# 
#         ant_a_dl = pointing_errors[ant_a]
#         ant_a_dm = pointing_errors[ant_a + offset]
# 
#         ant_b_dl = pointing_errors[ant_b]
#         ant_b_dm = pointing_errors[ant_b + offset]
# 
#         visibilities[i] = \
#             voltage_beam(0,0,ant_a_dl,ant_a_dm)*visibilities[i]* \
#             np.conj(voltage_beam(0,0,ant_b_dl,ant_b_dm))
# 
#     return visibilities, pointing_errors

def create_obs_vis(model_vis, sources, uv_points, stations, pointing_errors,
                   noise=0):

    visibilities = model_vis.copy()

    offset = len(stations)

    for i in range(visibilities.shape[0]):

        ant_a = uv_points[i,0]
        ant_b = uv_points[i,1]

        ant_a_dl = pointing_errors[ant_a]
        ant_a_dm = pointing_errors[ant_a + offset]

        ant_b_dl = pointing_errors[ant_b]
        ant_b_dm = pointing_errors[ant_b + offset]

        for j in range(sources.shape[0]):

            source_l = sources[j,1]
            source_m = sources[j,2]

            ant_a_effect = voltage_beam(source_l,source_m,ant_a_dl,ant_a_dm)
            ant_b_effect = np.conj(voltage_beam(source_l,source_m,ant_b_dl,ant_b_dm))

            visibilities[i,j] = ant_a_effect*visibilities[i,j]*ant_b_effect +\
                                noise[i]

    visibilities = np.sum(visibilities,1)

    return visibilities


def voltage_beam(l,m,delta_l=0,delta_m=0):

    sigma_l = 1
    sigma_m = 1

    beam_gain = np.exp(-1*(((l+delta_l)**2)/(2*(sigma_l**2)) +
                           ((m+delta_m)**2)/(2*(sigma_m**2))))

    return beam_gain.astype(np.complex128)[0]

def voltage_derivative(l,m,delta_l,delta_m,dimension):

    sigma_l = 1
    sigma_m = 1

    if dimension=="l":
        beam_derivative = (-1*(l+delta_l)/(sigma_l**2))*\
                          np.exp(-1*(((l+delta_l)**2)/(2*sigma_l**2) +
                                     ((m+delta_m)**2)/(2*sigma_m**2)))
    elif dimension=="m":
        beam_derivative = (-1*(m+delta_m)/(sigma_m**2))*\
                          np.exp(-1*(((l+delta_l)**2)/(2*sigma_l**2) +
                                     ((m+delta_m)**2)/(2*sigma_m**2)))

    return beam_derivative.astype(np.complex128)[0]


def jacobian(obs_vis, model_vis, stations, uv_points, pointing_errors,
             sources):

    J = np.zeros((2*np.shape(obs_vis)[0],2*len(stations)), dtype=np.complex128)

    offset = len(stations)

    for i in range(J.shape[0]/2):
        for j in range(len(stations)):
            ant_a = uv_points[i,0]
            ant_b = uv_points[i,1]

            ant_a_dl = pointing_errors[ant_a]
            ant_a_dm = pointing_errors[ant_a + offset]
            ant_b_dl = pointing_errors[ant_b]
            ant_b_dm = pointing_errors[ant_b + offset]

            if ant_a == j:
                for k in range(sources.shape[0]):
                    source_l = sources[k,1]
                    source_m = sources[k,2]

                    eq_conj = np.conj(voltage_beam(source_l,source_m,ant_b_dl,
                                                   ant_b_dm))
                    ep_deri = voltage_derivative(source_l,source_m,ant_a_dl,
                                                 ant_a_dm,"l")

                    J[i,j] += model_vis[i,k]*eq_conj*ep_deri

            if ant_b == j:
                for k in range(sources.shape[0]):
                    source_l = sources[k,1]
                    source_m = sources[k,2]

                    ep = voltage_beam(source_l,source_m,ant_a_dl,ant_a_dm)
                    eq_conj_deri = np.conj(voltage_derivative(source_l,
                                            source_m,ant_b_dl,ant_b_dm,"l"))
                    J[i,j] += model_vis[i,k]*ep*eq_conj_deri

    for i in range(J.shape[0]/2):
        for j in range(len(stations)):
            ant_a = uv_points[i,0]
            ant_b = uv_points[i,1]

            ant_a_dl = pointing_errors[ant_a]
            ant_a_dm = pointing_errors[ant_a + offset]
            ant_b_dl = pointing_errors[ant_b]
            ant_b_dm = pointing_errors[ant_b + offset]

            if ant_a == j:
                for k in range(sources.shape[0]):
                    source_l = sources[k,1]
                    source_m = sources[k,2]

                    eq_conj = np.conj(voltage_beam(source_l,source_m,ant_b_dl,ant_b_dm))
                    ep_deri = voltage_derivative(source_l,source_m,ant_a_dl,ant_a_dm,"m")
                    J[i,j+offset] += model_vis[i,k]*eq_conj*ep_deri

            if ant_b == j:
                for k in range(sources.shape[0]):
                    source_l = sources[k,1]
                    source_m = sources[k,2]

                    ep = voltage_beam(source_l,source_m,ant_a_dl,ant_a_dm)
                    eq_conj_deri = np.conj(voltage_derivative(source_l,source_m,
                                            ant_b_dl,ant_b_dm,"m"))
                    J[i,j+offset] += model_vis[i,k]*ep*eq_conj_deri

    J[np.shape(obs_vis)[0]:,:] = np.conj(J[:np.shape(obs_vis)[0],:])

    return J

    # plt.imshow(J.T.conj().dot(J), interpolation="none")
    # plt.colorbar()
    # plt.show()

def residual(obs_vis, model_vis, stations, uv_points, pointing_solutions,
             sources, noise):

    r = obs_vis - create_obs_vis(model_vis, sources, uv_points, stations,
                                 pointing_solutions,
                                 np.zeros([obs_vis.shape[0]],dtype=np
                                 .complex128))

    r = np.hstack((r,r.conj()))

    return r

def gn_solver(obs_vis, model_vis, stations, uv_points, pointing_errors,
              sources, noise):

    # Need to start with random initial guess seems odd, though may be a
    # property of the weird derivative/the fact that the solution is poorly
    # constrained in the single point source in the center of the field case.
    # Another point of interest is the rotational ambiguity - using perfect
    # gaussians means any pointing error at a given radius will minimise the
    # residual. This is less of an issue when there are multiple sources to
    # constrain the solution.

    pointing_solutions = (np.random.random([2*len(stations),1]) - 0.5)
    # pointing_solutions = pointing_solutions.astype(np.complex128)
    # pointing_solutions = np.ones([2*len(stations),1])

    improvement = np.inf
    accuracy = 1e-6

    r = residual(obs_vis, model_vis, stations, uv_points, pointing_solutions,
                 sources, noise)

    i = 1

    while improvement > accuracy:

        J = jacobian(obs_vis, model_vis, stations, uv_points,
                     pointing_solutions, sources)
        H = J.T.conj().dot(J)

        print np.sum(abs(H[0,1:14])+abs(H[0,15:])), abs(H[0,0])+abs(H[0,14])

        # plt.imshow(abs(H), interpolation="None")
        # plt.show()

        # We can take the approximate inversion here but need to reduce the
        # step size in order to ensure convergence. Averaging multiple steps
        # works! Need to investigate why. Read the StEFCal paper again.

        # NB! Convergence occurs using only major diagonal and averaging step.
        # NB! Convergence occurs using major and sub-diagonals without
        # averaging.

        # midpoint = int(0.5*H.shape[0])

        # Hn = np.diagflat(np.diagonal(H))
        # Hn[:midpoint,midpoint:] = np.diagflat(np.diagonal(H, midpoint))
        # Hn[midpoint:,:midpoint] = np.diagflat(np.diagonal(H,-midpoint))
        # H = Hn

        if (i%2)==0:
            pointing_solutions[:,0] += np.linalg.pinv(H).dot((J.T.conj()).dot(
                r))/2
        else:
            pointing_solutions[:,0] += np.linalg.pinv(H).dot((J.T.conj()).dot(
                r))

        r = residual(obs_vis, model_vis, stations, uv_points,
                     pointing_solutions, sources, noise)

        # print np.hstack((np.linalg.norm(pointing_solutions), pointing_errors))

        print "GN NORM = ", np.linalg.norm(abs(r))

        i += 1

        if i>10:
            # print np.hstack((pointing_solutions[0:14]**2+
            #                  pointing_solutions[14:]**2,
            #                  pointing_errors[0:14]**2+
            #                  pointing_errors[14:]**2))
            print np.hstack((pointing_solutions, pointing_errors))
            break

if __name__=="__main__":
    stations = [0,144,288,432,576,720,864,1008,1152,1296,1332,1440,2772,4176]
    frequency = 1.4e9

    uv_points = create_uv(stations,frequency, 0, 90)

    sources = np.array([[1,0,0],[1,0.8,0],[1,0.22,1],[1,-0.01,1],[1,1,1]])
    pointing_errors = (np.random.random([2*len(stations),1]) - 0.5)

    model_vis = create_model_vis(sources, uv_points)

    noise = ((np.random.random(model_vis.shape[0])-0.5)/100+1j*
             (np.random.random(model_vis.shape[0])-0.5)/100)

    noise[:] = 0

    obs_vis = create_obs_vis(model_vis, sources, uv_points, stations,
                             pointing_errors, noise)

    gn_solver(obs_vis,model_vis,stations, uv_points, pointing_errors,
                    sources, noise)

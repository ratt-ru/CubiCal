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

    visibilities = np.ones((np.shape(uv_points)[0],1))

    return visibilities


def create_obs_vis(uv_points, stations):

    visibilities = np.ones((np.shape(uv_points)[0],1))

    pointing_errors = (np.random.random([2*len(stations),1]) - 0.5)*2

    offset = len(stations)

    for i in range(visibilities.shape[0]):

        ant_a = uv_points[i,0]
        ant_b = uv_points[i,1]

        ant_a_dl = pointing_errors[ant_a]
        ant_a_dm = pointing_errors[ant_a + offset]

        ant_b_dl = pointing_errors[ant_b]
        ant_b_dm = pointing_errors[ant_b + offset]

        visibilities[i] = \
            voltage_beam(0,0,ant_a_dl,ant_a_dm)*visibilities[i]* \
            np.conj(voltage_beam(0,0,ant_b_dl,ant_b_dm))

    return visibilities, pointing_errors


def voltage_beam(l,m,delta_l=0,delta_m=0):

    sigma_l = 1
    sigma_m = 1

    beam_gain = np.exp(-1*(((l+delta_l)**2)/(2*(sigma_l**2)) +
                           ((m+delta_m)**2)/(2*(sigma_m**2))))

    return beam_gain

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

    return beam_derivative


def jacobian(obs_vis, model_vis, stations, uv_points, pointing_errors):

    J = np.zeros((2*np.shape(obs_vis)[0],2*len(stations)))

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
                eq_conj = np.conj(voltage_beam(0,0,ant_b_dl,ant_b_dm))
                ep_deri = voltage_derivative(0,0,ant_a_dl,ant_a_dm,"l")
                J[i,j] += model_vis[i,0]*eq_conj*ep_deri

            if ant_b == j:
                ep = voltage_beam(0,0,ant_a_dl,ant_a_dm)
                eq_conj_deri = np.conj(voltage_derivative(0,0,ant_b_dl,ant_b_dm,"l"))
                J[i,j] += model_vis[i,0]*ep*eq_conj_deri

    for i in range(J.shape[0]/2):
        for j in range(len(stations)):
            ant_a = uv_points[i,0]
            ant_b = uv_points[i,1]

            ant_a_dl = pointing_errors[ant_a]
            ant_a_dm = pointing_errors[ant_a + offset]
            ant_b_dl = pointing_errors[ant_b]
            ant_b_dm = pointing_errors[ant_b + offset]

            if ant_a == j:
                eq_conj = np.conj(voltage_beam(0,0,ant_b_dl,ant_b_dm))
                ep_deri = voltage_derivative(0,0,ant_a_dl,ant_a_dm,"m")
                J[i,j+offset] += model_vis[i,0]*eq_conj*ep_deri

            if ant_b == j:
                ep = voltage_beam(0,0,ant_a_dl,ant_a_dm)
                eq_conj_deri = np.conj(voltage_derivative(0,0,ant_b_dl,
                                                          ant_b_dm,"m"))
                J[i,j+offset] += model_vis[i,0]*ep*eq_conj_deri

    J[np.shape(obs_vis)[0]:,:] = np.conj(J[:np.shape(obs_vis)[0],:])

    return J

    # plt.imshow(J.T.conj().dot(J), interpolation="none")
    # plt.colorbar()
    # plt.show()

def residual(obs_vis, model_vis, stations, uv_points, pointing_errors):

    r = np.empty(obs_vis.shape)

    offset = len(stations)

    for i in range(r.shape[0]):

        ant_a = uv_points[i,0]
        ant_b = uv_points[i,1]

        ant_a_dl = pointing_errors[ant_a]
        ant_a_dm = pointing_errors[ant_a + offset]

        ant_b_dl = pointing_errors[ant_b]
        ant_b_dm = pointing_errors[ant_b + offset]

        r[i] = obs_vis[i] -\
            voltage_beam(0,0,ant_a_dl,ant_a_dm)*model_vis[i]* \
            np.conj(voltage_beam(0,0,ant_b_dl,ant_b_dm))

    r = np.vstack((r,r.conj()))

    return r

def gn_solver(obs_vis, model_vis, stations, uv_points, pointing_errors):

    # Need to start with random initial guess seems odd, though may be a
    # property of the weird derivative/the fact that the solution is poorly
    # constrained in the single point source in the center of the field case.
    # Another point of interest is the rotational ambiguity - using perfect
    # gaussians means any pointing error at a given radius will minimise the
    # residual. This is less of an issue when there are multiple sources to
    # constrain the solution.

    pointing_solutions = (np.random.random([2*len(stations),1]) - 0.5)*2
    # pointing_solutions = np.ones([2*len(stations),1])

    improvement = np.inf
    accuracy = 1e-6

    r = residual(obs_vis, model_vis, stations, uv_points, pointing_solutions)

    i = 1

    while improvement > accuracy:

        t0 = time.time()
        J = jacobian(obs_vis, model_vis, stations, uv_points, pointing_solutions)
        H = J.T.conj().dot(J)
        print "TIME:", time.time() - t0

        print np.sum(abs(H[0,1:])), abs(H[0,0])

        # We can take the approximate inversion here but need to reduce the
        # step size in order to ensure convergence. Averaging multiple steps
        # works! Need to investigate why. Read the StEFCal paper again.

        midpoint = int(0.5*H.shape[0])


        # Hn = np.diagflat(np.diagonal(H))
        # Hn[:midpoint,midpoint:] = np.diagflat(np.diagonal(H, midpoint))
        # Hn[midpoint:,:midpoint] = np.diagflat(np.diagonal(H,-midpoint))
        # H = Hn

        t0 = time.time()
        temp1, temp2, temp3, temp4 = build_H(obs_vis, model_vis, stations,
                                        uv_points,
                       pointing_solutions)
        print "TIME:", time.time() - t0

        print np.allclose(temp1[:14],np.diag(H[:14,:14]).reshape(14,1))
        print np.allclose(temp2[:14],np.diag(H[:14,14:]).reshape(14,1))
        print np.allclose(temp3[:14],np.diag(H[14:,:14]).reshape(14,1))
        print np.allclose(temp4[:14],np.diag(H[14:,14:]).reshape(14,1))

        if (i%2)==0:
            pointing_solutions += np.linalg.pinv(H).dot((J.T.conj()).dot(r))/2
        else:
            pointing_solutions += np.linalg.pinv(H).dot((J.T.conj()).dot(r))

        # print pointing_solutions
        # print pointing_errors


        # plt.imshow(J.T.conj(),interpolation='none')
        # plt.show()

        t0 = time.time()
        r = residual(obs_vis, model_vis, stations, uv_points, pointing_solutions)
        print time.time() - t0

        print "GN NORM = ", np.linalg.norm(r)

        # raw_input("Press Enter to continue...")

        i += 1

        if i>20:
            # print pointing_solutions[0:14]**2+pointing_solutions[14:]**2
            # print pointing_errors[0:14]**2+pointing_errors[14:]**2
            # print np.allclose(pointing_solutions[0:14]**2+pointing_solutions[
            #                                               14:]**2,pointing_errors[0:14]**2+pointing_errors[14:]**2)
            # print pointing_solutions
            # print pointing_errors
            break

    return J.T.conj().dot(J)


def build_H(obs_vis, model_vis, stations, uv_points, pointing_errors):

    A = np.zeros([len(stations),1])
    B = np.zeros([len(stations),1])
    C = np.zeros([len(stations),1])
    D = np.zeros([len(stations),1])

    offset = len(stations)

    for i in range(uv_points.shape[0]):
        ant_a = uv_points[i,0]
        ant_b = uv_points[i,1]

        ant_a_dl = pointing_errors[ant_a]
        ant_a_dm = pointing_errors[ant_a + offset]
        ant_b_dl = pointing_errors[ant_b]
        ant_b_dm = pointing_errors[ant_b + offset]

        eq_conj = np.conj(voltage_beam(0,0,ant_b_dl,ant_b_dm))
        ep_conj = np.conj(voltage_beam(0,0,ant_a_dl,ant_a_dm))

        dep_dl = voltage_derivative(0,0,ant_a_dl,ant_a_dm,"l")
        dep_dm = voltage_derivative(0,0,ant_a_dl,ant_a_dm,"m")

        deq_dl = voltage_derivative(0,0,ant_b_dl,ant_b_dm,"l")
        deq_dm = voltage_derivative(0,0,ant_b_dl,ant_b_dm,"m")

        A[ant_a] += 2*np.abs(model_vis[i]*eq_conj*dep_dl)**2
        A[ant_b] += 2*np.abs(model_vis[i]*ep_conj*deq_dl)**2

        B[ant_a] += np.abs(model_vis[i]*eq_conj)**2*\
                    (dep_dl.conj()*dep_dm+dep_dl*dep_dm.conj())
        B[ant_b] += np.abs(model_vis[i]*ep_conj)**2*\
                    (deq_dl.conj()*deq_dm+deq_dl*deq_dm.conj())

        C = B.conj()

        D[ant_a] += 2*np.abs(model_vis[i]*eq_conj*dep_dm)**2
        D[ant_b] += 2*np.abs(model_vis[i]*ep_conj*deq_dm)**2

    return A, B, C, D

def lm_solver(obs_vis, model_vis, stations, uv_points, pointing_errors):

    # Need to start with random initial guess seems odd, though may be a
    # property of the weird derivative/the fact that the solution is poorly
    # constrained in the single point source in the center of the field case.
    # Another point of interest is the rotational ambiguity - using perfect
    # gaussians means any pointing error at a given radius will minimise the
    # residual. This is less of an issue when there are multiple sources to
    # constrain the solution.

    pointing_solutions = (np.random.random([2*len(stations),1]) - 0.5)*2
    # pointing_solutions = np.ones([2*len(stations),1])

    improvement = np.inf
    accuracy = 1e-6

    r = residual(obs_vis, model_vis, stations, uv_points, pointing_solutions)

    i = 1

    lamda = 1

    while improvement > accuracy:

        t0 = time.time()
        J = jacobian(obs_vis, model_vis, stations, uv_points, pointing_solutions)
        H = J.T.conj().dot(J)
        print "TIME:", time.time() - t0

        # We can take the approximate inversion here but need to reduce the
        # step size in order to ensure convergence. Averaging multiple steps
        # works! Need to investigate why. Read the StEFCal paper again.

        midpoint = int(0.5*H.shape[0])

        Hn = np.diagflat(np.diagonal(H))
        D = Hn.copy()
        Hn[:midpoint,midpoint:] = np.diagflat(np.diagonal(H, midpoint))
        Hn[midpoint:,:midpoint] = np.diagflat(np.diagonal(H,-midpoint))
        H = Hn

        # t0 = time.time()
        # temp1, temp2, temp3, temp4 = build_H(obs_vis, model_vis, stations,
        #                                 uv_points,
        #                pointing_solutions)
        # print "TIME:", time.time() - t0
        #
        # print np.allclose(temp1[:14],np.diag(H[:14,:14]).reshape(14,1))
        # print np.allclose(temp2[:14],np.diag(H[:14,14:]).reshape(14,1))
        # print np.allclose(temp3[:14],np.diag(H[14:,:14]).reshape(14,1))
        # print np.allclose(temp4[:14],np.diag(H[14:,14:]).reshape(14,1))

        if (i%2)==0:
            pointing_solutions += np.linalg.pinv(H+lamda*D).dot((J.T.conj())
                                    .dot(r))/2
        else:
            pointing_solutions += np.linalg.pinv(H+lamda*D).dot((J.T.conj())
                                    .dot(r))

        # print pointing_solutions
        # print pointing_errors


        # plt.imshow(J.T.conj(),interpolation='none')
        # plt.show()

        r = residual(obs_vis, model_vis, stations, uv_points, pointing_solutions)

        print "LM NORM = ", np.linalg.norm(r)

        i += 1

        if i>20:
            # print pointing_solutions[0:14]**2+pointing_solutions[14:]**2
            # print pointing_errors[0:14]**2+pointing_errors[14:]**2
            # print np.allclose(pointing_solutions[0:14]**2+pointing_solutions[
            #                                               14:]**2,pointing_errors[0:14]**2+pointing_errors[14:]**2)
            # print pointing_solutions
            # print pointing_errors
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
    obs_vis, pointing_errors = create_obs_vis(uv_points, stations)

    # J = jacobian(obs_vis,model_vis,stations, uv_points, pointing_errors)
    # plt.imshow(J,interpolation='none')
    # plt.show()

    # JHJ = lm_solver(obs_vis,model_vis,stations, uv_points, pointing_errors)
    JHJ = gn_solver(obs_vis,model_vis,stations, uv_points, pointing_errors)

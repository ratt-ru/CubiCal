import os, sys, os.path

def logprint(arg):
    print>>sys.stderr,arg


def kernels_test():
    command = os.path.join(os.path.dirname(__file__),"benchmark/kernel_timings.py") 

    for kernel in "phase_only", "f_slope", "t_slope", "tf_plane":
        for interval in 1, 10:
            command = "python {command} cy{kernel} cy{kernel}_omp --reference cy{kernel}_reference " \
                   "--omp 4 --diag --nd 1 --nd 5 --nf 50 --nt 50 --ti {interval} --fi {interval}".format(**locals())
            logprint("Running {}".format(command))
            if os.system(command):
                sys.exit(1)


if __name__ == '__main__':
    kernels_test()

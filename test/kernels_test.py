from test.util import logprint
from test.benchmark.kernel_timings import main


cmd = "cy{kernel} cy{kernel}_omp --reference cy{kernel}_reference " \
      "--omp 4 --diag --nd 1 --nd 5 --nf 50 --nt 50 --ti {interval} --fi {interval}"


def kernels_test():
    for kernel in "phase_only", "f_slope", "t_slope", "tf_plane":
        for interval in 1, 10:
            args = cmd.format(kernel=kernel, interval=interval).split(' ')
            logprint("Running cubical with arguments '{}'".format(' '.join(args)))
            main(args)


if __name__ == '__main__':
    kernels_test()

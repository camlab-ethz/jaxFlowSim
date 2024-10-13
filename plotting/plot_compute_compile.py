import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use("science")

compile_time_fn = "timing_compile.txt"
compute_time_fn = "timing_compute.txt"
compile_time = np.loadtxt(compile_time_fn)
compute_time = np.loadtxt(compute_time_fn)

N_segments_compile_time = np.unique(compile_time[:, 0])
N_segments_compute_time = np.unique(compute_time[:, 0])

print(N_segments_compile_time)
print(N_segments_compute_time)

N_different_segments_compile_time = len(N_segments_compile_time)
N_different_segments_compute_time = len(N_segments_compute_time)
N_samples_compile_time = len(
    np.where(compile_time[:, 0] == N_segments_compile_time[0])[0]
)
N_samples_compute_time = len(
    np.where(compute_time[:, 0] == N_segments_compute_time[0])[0]
)

averages_compile_time = np.empty(N_different_segments_compile_time)
averages_compute_time = np.empty(N_different_segments_compute_time)

for i in range(N_different_segments_compile_time):
    i = int(i)
    averages_compile_time[i] = (
        compile_time[
            np.where(compile_time[:, 0] == N_segments_compile_time[i]), 1
        ].sum()
        / N_samples_compile_time
    )

for i in range(N_different_segments_compute_time):
    i = int(i)
    averages_compute_time[i] = (
        compute_time[
            np.where(compute_time[:, 0] == N_segments_compute_time[i]), 1
        ].sum()
        / N_samples_compute_time
    )

print(averages_compile_time)
print(averages_compute_time)

fig, ax = plt.subplots()
plt.scatter(N_segments_compile_time, averages_compile_time)
plt.scatter(N_segments_compute_time, averages_compute_time)
ax.set_xlabel(r"\#segments")
ax.set_ylabel("t[s]")
plt.title("average of compute vs compile time over 10 runs")
plt.legend(["compile", "compute"], loc="upper left")
plt.savefig("timing.eps")

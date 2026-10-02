import os
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter


WIDTH = 1200
HEIGHT = 800
MAX_ITER = 250


def ryadok(y):
    row = bytearray(WIDTH)

    for x in range(WIDTH):
        real = -2.0 + x * 3.0 / WIDTH
        imag = -1.5 + y * 3.0 / HEIGHT

        c = complex(real, imag)
        z = 0j
        iteration = 0

        while abs(z) <= 2 and iteration < MAX_ITER:
            z = z * z + c
            iteration += 1

        if iteration == MAX_ITER:
            value = 0
        else:
            value = int(255 * iteration / MAX_ITER)

        row[x] = value

    return row


def poslidovno():
    return [ryadok(y) for y in range(HEIGHT)]


def protsesamy(n):
    with ProcessPoolExecutor(max_workers=n) as ex:
        return list(ex.map(ryadok, range(HEIGHT), chunksize=8))

def porozhniy_pul(n):
    with ProcessPoolExecutor(max_workers=n) as ex:
        list(ex.map(abs, range(n)))


def zamir(fn, progoniv=3, *args):
    times = []
    result = None

    for _ in range(progoniv):
        start = perf_counter()
        result = fn(*args)
        end = perf_counter()

        times.append(end - start)

    return min(times), times, result


if __name__ == "__main__":
    t_base, chasy, base = zamir(poslidovno, 3)

    print(
        "послідовно",
        [round(c, 2) for c in chasy],
        "мінімум",
        round(t_base, 2)
    )

    for n in (1, 2, 4, 8, os.cpu_count()):
        t, chasy, r = zamir(protsesamy, 3, n)

        assert r == base, "результат розійшовся з послідовним"

        s = t_base / t

        print(
            "процесів",
            n,
            [round(c, 2) for c in chasy],
            "мінімум",
            round(t, 2),
            "прискорення",
            round(s, 2),
            "ефективність",
            round(s / n * 100),
            "%"
        )

    for n in (2, os.cpu_count()):
        t, chasy, _ = zamir(porozhniy_pul, 3, n)
        print("порожній пул,", n, "процесів:",
        [round(c, 3) for c in chasy],
        "мінімум", round(t, 3), "с")

n = [1, 2, 4, 8]
pryskorennya = [0.91, 1.47, 2.51, 3.00]          # ваші числа

fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
a1.plot(n, pryskorennya, "o-", label="виміряне")
a1.plot(n, n, "--", color="gray", label="ідеал ×N")
a1.set_xlabel("процесів"); a1.set_ylabel("прискорення, разів"); a1.legend(); a1.grid(True)
a2.plot(n, [s / k * 100 for s, k in zip(pryskorennya, n)], "s-", color="tab:orange")
a2.set_xlabel("процесів"); a2.set_ylabel("ефективність, %"); a2.set_ylim(0, 110); a2.grid(True)
fig.tight_layout()
fig.savefig("grafik.png", dpi=110)
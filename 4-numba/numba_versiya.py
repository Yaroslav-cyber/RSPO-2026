
import time
import numpy as np
from numba import njit, prange

WIDTH = 1200
HEIGHT = 800
MAX_ITER = 250


def mandelbrot(width, height, max_iter):
    pixels = bytearray(width * height)

    for y in range(height):
        for x in range(width):
            real = -2.0 + x * 3.0 / width
            imag = -1.5 + y * 3.0 / height

            c = complex(real, imag)
            z = 0j
            iteration = 0

            while abs(z) <= 2 and iteration < max_iter:
                z = z * z + c
                iteration += 1

            if iteration == max_iter:
                value = 0
            else:
                value = int(255 * iteration / max_iter)

            pixels[y * width + x] = value

    return pixels


@njit
def mandelbrot_numba(width, height, max_iter):
    pixels = np.empty(width * height, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            real = -2.0 + x * 3.0 / width
            imag = -1.5 + y * 3.0 / height

            c = complex(real, imag)
            z = 0j
            iteration = 0

            while abs(z) <= 2 and iteration < max_iter:
                z = z * z + c
                iteration += 1

            if iteration == max_iter:
                value = 0
            else:
                value = int(255 * iteration / max_iter)

            pixels[y * width + x] = value

    return pixels


@njit(parallel=True)
def mandelbrot_prange(width, height, max_iter):
    pixels = np.empty(width * height, dtype=np.uint8)

    for y in prange(height):
        for x in range(width):
            real = -2.0 + x * 3.0 / width
            imag = -1.5 + y * 3.0 / height

            c = complex(real, imag)
            z = 0j
            iteration = 0

            while abs(z) <= 2 and iteration < max_iter:
                z = z * z + c
                iteration += 1

            if iteration == max_iter:
                value = 0
            else:
                value = int(255 * iteration / max_iter)

            pixels[y * width + x] = value

    return pixels


def measure(function, runs=3):
    times = []
    result = None

    for _ in range(runs):
        start = time.perf_counter()
        result = function(WIDTH, HEIGHT, MAX_ITER)
        times.append(time.perf_counter() - start)

    return min(times), times, result


if __name__ == "__main__":
    print(f"Розмір: {WIDTH} x {HEIGHT}")
    print(f"Максимальна кількість ітерацій: {MAX_ITER}")
    print()

    base_time, base_runs, base = measure(mandelbrot)
    base = np.frombuffer(base, dtype=np.uint8)

    print("Послідовна версія:")
    print("Прогони:", [round(t, 3) for t in base_runs])
    print(f"Мінімальний час: {base_time:.3f} с")
    print()

    # Перший виклик включає компіляцію Numba.
    start = time.perf_counter()
    compiled_result = mandelbrot_numba(WIDTH, HEIGHT, MAX_ITER)
    compile_time = time.perf_counter() - start

    print(f"Перший виклик @njit (компіляція + виконання): {compile_time:.3f} с")

    njit_time, njit_runs, njit_result = measure(mandelbrot_numba)

    diff = np.flatnonzero(base != njit_result)

    print("Індекси розбіжних пікселів:", diff)
    for i in diff:
        print(
            f"Індекс: {i}, "
            f"x={i % WIDTH}, y={i // WIDTH}, "
            f"еталон={base[i]}, Numba={njit_result[i]}"
        )

    print("@njit:")
    print("Прогони:", [round(t, 4) for t in njit_runs])
    print(f"Мінімальний час: {njit_time:.4f} с")
    print(f"Прискорення: {base_time / njit_time:.2f}x")
    print("Збіг з еталоном:", np.array_equal(base, njit_result))
    print("Кількість розбіжних пікселів:", np.count_nonzero(base != njit_result))
    print("Перші значення еталона:", base[:20])
    print("Перші значення Numba:", njit_result[:20])
    print()

    # Окремо вимірюємо перший виклик паралельної версії.
    start = time.perf_counter()
    parallel_result = mandelbrot_prange(WIDTH, HEIGHT, MAX_ITER)
    parallel_compile_time = time.perf_counter() - start

    print(f"Перший виклик prange (компіляція + виконання): {parallel_compile_time:.3f} с")

    prange_time, prange_runs, prange_result = measure(mandelbrot_prange)

    print("prange:")
    print("Прогони:", [round(t, 4) for t in prange_runs])
    print(f"Мінімальний час: {prange_time:.4f} с")
    print(f"Прискорення відносно послідовної версії: {base_time / prange_time:.2f}x")
    print("Збіг з еталоном:", np.array_equal(base, prange_result))
    print("Кількість розбіжних пікселів:", np.count_nonzero(base != prange_result))
    print("Перші значення еталона:", base[:20])
    print("Перші значення Numba:", prange_result[:20])

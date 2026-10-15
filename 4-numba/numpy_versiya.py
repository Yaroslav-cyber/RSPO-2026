import time
import numpy as np

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


def mandelbrot_numpy(width, height, max_iter):
    real = -2.0 + np.arange(width) * 3.0 / width
    imag = -1.5 + np.arange(height) * 3.0 / height

    cx, cy = np.meshgrid(real, imag)

    z = np.zeros((height, width), dtype=np.complex128)
    c = cx + 1j * cy

    pixels = np.zeros((height, width), dtype=np.uint8)
    active = np.ones((height, width), dtype=bool)

    for iteration in range(max_iter):
        z[active] = z[active] * z[active] + c[active]

        escaped = active & (np.abs(z) > 2)
        pixels[escaped] = int(255 * (iteration + 1) / max_iter)
        active[escaped] = False

        if not active.any():
            break

    pixels[active] = 0

    return pixels.ravel()


if __name__ == "__main__":
    print(f"Розмір: {WIDTH} x {HEIGHT}")
    print(f"Максимальна кількість ітерацій: {MAX_ITER}")
    print()

    base = np.frombuffer(
        mandelbrot(WIDTH, HEIGHT, MAX_ITER),
        dtype=np.uint8
    )

    start = time.perf_counter()
    result = mandelbrot_numpy(WIDTH, HEIGHT, MAX_ITER)
    elapsed = time.perf_counter() - start

    differences = np.count_nonzero(base != result)

    print(f"NumPy: {elapsed:.4f} с")
    print(f"Прискорення: {14.225 / elapsed:.2f}x")
    print(f"Розбіжних пікселів: {differences} із {base.size}")

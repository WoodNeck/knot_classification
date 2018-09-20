import numpy as np
from scipy import ndimage


def perona_malik(img, iterations, delta, kappa):
    # center pixel distances
    dd = np.sqrt(2)

    # 2D finite difference windows
    windows = [
        np.array(
                [[0, 1, 0], [0, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [0, 1, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 1], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [1, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 1], [0, -1, 0], [0, 0, 0]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [0, 0, 1]], np.float64
        ),
        np.array(
                [[0, 0, 0], [0, -1, 0], [1, 0, 0]], np.float64
        ),
        np.array(
                [[1, 0, 0], [0, -1, 0], [0, 0, 0]], np.float64
        ),
    ]

    for r in range(iterations):
        # approximate gradients
        nabla = [ndimage.filters.convolve(img, w) for w in windows]

        # approximate diffusion function
        diff = [1 / (1 + (n / kappa) ** 2) for n in nabla]

        # update image
        terms = [diff[i] * nabla[i] for i in range(4)]
        terms += [(1 / (dd ** 2)) * diff[i] * nabla[i] for i in range(4, 8)]
        img = img + delta * sum(terms)

    # Gradient 계산
    # Kernel for Gradient in x-direction
    Kx = np.array(
        [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
    )
    # Kernel for Gradient in y-direction
    Ky = np.array(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
    )

    # Apply kernels to the image
    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    # return norm of (Ix, Iy)
    gradient = np.hypot(Ix, Iy)

    return img, gradient

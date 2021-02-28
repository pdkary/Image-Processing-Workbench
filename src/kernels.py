import numpy as np
blur_kernel_5 = (
    np.array(
        [
            [1, 2, 4, 2, 1],
            [2, 4, 8, 4, 2],
            [4, 8, 16, 8, 4],
            [2, 4, 8, 4, 2],
            [1, 2, 4, 2, 1],
        ]
    )
    / 100
)

blur_kernel_3 = (
    np.array(
        [
            [1, 2, 1],
            [2, 4, 2],
            [1, 2, 1],
        ]
    )
    / 16
)
laplace_kernel_3 = (
    np.array(
        [
            [1, 1, 1],
            [1, -8, 1],
            [1, 1, 1],
        ]
    )
)
sharpen_kernel_3 = (
    np.array(
        [
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0],
        ]
    )
)
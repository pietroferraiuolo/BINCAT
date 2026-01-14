import xupy.typings as _xt
from skimage.transform import resize


def upsample(img: _xt.Array, s: int = 4, order: str = "cubic") -> _xt.Array:
    """
    Upsample an image by a factor of `s` using skimage's resize function.

    Parameters
    ----------
    img : _xt.Array
        The input image to be upsampled. Can be a cube, with the last dimension
        representing the image id.
    s : int
        The upsampling factor.
    order : str
        The interpolation order to use:
        - 'nearest': Nearest-neighbor interpolation.
        - 'linear': Bilinear interpolation.
        - 'quadratic': Bi-quadratic interpolation.
        - 'cubic': Bicubic interpolation.
        - 'quartic': Bi-quartic interpolation.
        - 'quintic': Bi-quintic interpolation.
    """
    if order not in ["nearest", "linear", "quadratic", "cubic", "quartic", "quintic"]:
        raise ValueError(
            "Invalid order. Must be one of 'nearest', 'linear', 'quadratic', 'cubic', 'quartic', 'quintic'."
        )
    order_map = {
        "nearest": 0,
        "linear": 1,
        "quadratic": 2,
        "cubic": 3,
        "quartic": 4,
        "quintic": 5,
    }
    out_shape = (
        (img.shape[0] * s, img.shape[1] * s)
        if img.ndim == 2
        else (img.shape[0] * s, img.shape[1] * s, img.shape[2])
    )
    y = resize(
        img, out_shape, order=order_map[order], anti_aliasing=False, preserve_range=True
    )
    return y.astype(img.dtype)

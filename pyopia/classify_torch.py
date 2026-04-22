"""
Classifier for PyOPIA using PyTorch TorchScript models.

Loads a TorchScript archive (``.pt``) containing a traced model and a
``metadata.json`` with class labels, input dimensions, normalization
stats, and optional preprocessing config. Compatible with models
exported by ``pyopia-train export-pyopia-torch`` and by
``pyopia.train_torch.save_pytorch_model``.

The TorchScript archive is self-contained, so only extra dependency
is PyTorch (optional dependency).
"""

import hashlib
import json
import logging

import numpy as np
from skimage.exposure import rescale_intensity
from skimage.transform import resize

logger = logging.getLogger()

try:
    import torch
except ImportError:
    info_str = "ERROR: Could not import PyTorch. classify_torch will not work"
    info_str += " until you install torch.\n"
    info_str += "Use: uv sync --extra classification-torch\n"
    raise ImportError(info_str)


def _get_device():
    """Select the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_pytorch_model(
    model, class_labels, img_height, img_width, output_path, preprocess=None
):
    """Save a PyTorch model with metadata as a TorchScript archive.

    The model is expected to accept inputs in the [0, 255] range and
    bake any per-channel normalisation into its own forward pass (see
    notebook export examples).  This mirrors the convention in
    :mod:`pyopia.classify` where the TF model has a Rescaling layer
    as its first op.

    Parameters
    ----------
    model : torch.nn.Module
        Trained PyTorch model.  Must accept ``(B, 3, H, W)`` floats in
        ``[0, 255]`` and produce class logits.
    class_labels : list of str
        Class name strings.
    img_height : int
        Expected input image height.
    img_width : int
        Expected input image width.
    output_path : str
        Path to save the .pt file.
    preprocess : dict or None
        Optional preprocessing config (e.g.
        ``{"type": "percentile", "lower_pct": 1.0, "upper_pct": 99.0}``).
        Stored in metadata so inference applies matching preprocessing.
    """
    model.eval()
    try:
        scripted = torch.jit.script(model)
    except Exception:
        example_input = torch.randn(1, 3, int(img_height), int(img_width)) * 255
        scripted = torch.jit.trace(model, example_input)

    meta = {
        "class_labels": list(class_labels),
        "img_height": int(img_height),
        "img_width": int(img_width),
    }
    if preprocess is not None:
        meta["preprocess"] = preprocess

    torch.jit.save(
        scripted, output_path, _extra_files={"metadata.json": json.dumps(meta)}
    )


class Classify:
    """PyTorch TorchScript classifier for the PyOPIA pipeline.

    Drop-in replacement for :class:`pyopia.classify.Classify`.
    Loads a TorchScript ``.pt`` archive with metadata.

    Supports models from:
    - ``pyopia-train export-pyopia-torch``

    Parameters
    ----------
    model_path : str
        Path to a ``.pt`` TorchScript archive.
    normalize_intensity : bool
        Scale input ROI intensity to [0-1] before classification.
    correct_whitebalance : bool
        Per-channel histogram correction before classification.

    Example
    -------
    .. code-block:: python

        cl = Classify(model_path='model.pt')
        prediction = cl.proc_predict(roi)

    TOML config:

    .. code-block:: toml

        [steps.classifier]
        pipeline_class = 'pyopia.classify_torch.Classify'
        model_path = 'model.pt'
    """

    def __init__(
        self,
        model_path=None,
        normalize_intensity=True,
        correct_whitebalance=False,
    ):
        self.model_path = model_path
        self.correct_whitebalance = correct_whitebalance
        self.normalize_intensity = normalize_intensity
        self.load_model()

    def __call__(self):
        return self

    def load_model(self):
        """Load a TorchScript model and metadata from a .pt archive."""
        model_path = self.model_path

        extra_files = {"metadata.json": ""}
        self.model = torch.jit.load(
            model_path, map_location="cpu", _extra_files=extra_files
        )
        self.model.eval()

        metadata = json.loads(extra_files["metadata.json"])
        self.class_labels = metadata["class_labels"]
        self.img_height = metadata["img_height"]
        self.img_width = metadata["img_width"]
        self.preprocess = metadata.get("preprocess", None)

        self.device = _get_device()
        self.model = self.model.to(self.device)
        logger.info(f"PyTorch classifier loaded on device: {self.device}")

        with open(model_path, "rb") as f:
            digest = hashlib.file_digest(f, "sha256")
        self.model_hash = digest.hexdigest()

        logger.info(self.class_labels)

    def preprocessing(self, img_input):
        """Preprocess an ROI for prediction.

        Parameters
        ----------
        img_input : ndarray
            Particle ROI, float, range 0-1, shape (H, W, 3) or (H, W).

        Returns
        -------
        img_preprocessed : ndarray
            Shape (1, 3, img_height, img_width), float32, ready for the model.
        """
        whitebalanced = img_input.astype(np.float64)

        if self.correct_whitebalance:
            p = 99
            for c in range(3):
                whitebalanced[:, :, c] += (p / 100) - np.percentile(
                    whitebalanced[:, :, c], p
                )
            whitebalanced[whitebalanced > 1] = 1
            whitebalanced[whitebalanced < 0] = 0

        if self.normalize_intensity:
            whitebalanced = rescale_intensity(whitebalanced)

        # Optional percentile preprocessing (from pyopia-train checkpoints).
        # Applied after rescale_intensity, before resize, matching the
        # training pipeline's transform order.
        if self.preprocess is not None:
            preprocess_type = self.preprocess.get("type", "none")
            if preprocess_type == "percentile":
                lower = float(self.preprocess.get("lower_pct", 1.0))
                upper = float(self.preprocess.get("upper_pct", 99.0))
                whitebalanced = self._percentile_normalize(whitebalanced, lower, upper)

        # Convert to [0, 255] floats — matches the convention in
        # pyopia.classify.preprocessing.  Any per-channel normalisation
        # is baked into the model itself (see notebook export).
        img = (whitebalanced * 255).astype(np.float32)

        # Bilinear resize without anti-aliasing — matches the original
        # pyopia.classify (TF) convention and the notebook training
        # pipeline (PIL bilinear).
        img = resize(
            img,
            (self.img_height, self.img_width),
            preserve_range=True,
            anti_aliasing=False,
        ).astype(np.float32)

        # HWC → CHW, add batch dim
        img_preprocessed = np.transpose(img, (2, 0, 1))
        img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
        return img_preprocessed

    @staticmethod
    def _percentile_normalize(img, lower_pct=1.0, upper_pct=99.0):
        """Robust per-image percentile intensity rescaling.

        Matches pyopia-train's ``PercentileNormalize``: computes a single
        low/high range from the per-pixel RGB mean, then applies it
        identically to all channels.
        """
        arr = img.astype(np.float64)
        if arr.ndim == 2:
            intensity = arr
        else:
            intensity = arr.mean(axis=2) if arr.shape[2] > 1 else arr[:, :, 0]
        lo = float(np.percentile(intensity, lower_pct))
        hi = float(np.percentile(intensity, upper_pct))
        if hi <= lo + 1e-6:
            return np.clip(arr, 0.0, 1.0)
        arr = (arr - lo) / (hi - lo)
        return np.clip(arr, 0.0, 1.0)

    def predict(self, img_preprocessed):
        """Run the model on a preprocessed ROI.

        Parameters
        ----------
        img_preprocessed : ndarray
            Shape (1, 3, H, W), float32.

        Returns
        -------
        prediction : ndarray
            Probability distribution, shape (num_classes,).
        """
        with torch.no_grad():
            tensor = torch.from_numpy(img_preprocessed).to(self.device)
            output = self.model(tensor)
            prediction = torch.nn.functional.softmax(output[0], dim=0)
        return prediction.cpu().numpy()

    def proc_predict(self, img_input):
        """Preprocess and classify a particle ROI.

        Parameters
        ----------
        img_input : ndarray
            Particle ROI, float, range 0-1.

        Returns
        -------
        prediction : ndarray
            Probability distribution, shape (num_classes,).
        """
        img_preprocessed = self.preprocessing(img_input)
        prediction = self.predict(img_preprocessed)
        return prediction

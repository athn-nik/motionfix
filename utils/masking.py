import torch


class BaseMask(object):
    @property
    def bool_matrix(self):
        """Return a bool (uint8) matrix with 1s to all places that should be
        kept."""
        raise NotImplementedError()

    @property
    def float_matrix(self):
        """Return the bool matrix as a float to be used as a multiplicative
        mask for non softmax attentions."""
        if not hasattr(self, "_float_matrix"):
            with torch.no_grad():
                self._float_matrix = self.bool_matrix.float()
        return self._float_matrix

    @property
    def lengths(self):
        """If the matrix is of the following form

            1 1 1 0 0 0 0
            1 0 0 0 0 0 0
            1 1 0 0 0 0 0

        then return it as a vector of integers

            3 1 2.
        """
        if not hasattr(self, "_lengths"):
            with torch.no_grad():
                lengths = self.bool_matrix.long().sum(dim=-1)
                # make sure that the mask starts with 1s and continues with 0s
                # this should be changed to something more efficient, however,
                # I chose simplicity over efficiency since the LengthMask class
                # will be used anyway (and the result is cached)
                m = self.bool_matrix.view(-1, self.shape[-1])
                for i, l in enumerate(lengths.view(-1)):
                    if not torch.all(m[i, :l]):
                        raise ValueError("The mask is not a length mask")
                self._lengths = lengths
        return self._lengths

    @property
    def shape(self):
        """Return the shape of the boolean mask."""
        return self.bool_matrix.shape

    @property
    def additive_matrix(self):
        """Return a float matrix to be added to an attention matrix before
        softmax."""
        if not hasattr(self, "_additive_matrix"):
            with torch.no_grad():
                self._additive_matrix = torch.log(self.bool_matrix.float())
        return self._additive_matrix

    @property
    def additive_matrix_finite(self):
        """Same as additive_matrix but with -1e24 instead of infinity."""
        if not hasattr(self, "_additive_matrix_finite"):
            with torch.no_grad():
                self._additive_matrix_finite = (
                    (~self.bool_matrix).float() * (-1e24)
                )
        return self._additive_matrix_finite

    @property
    def all_ones(self):
        """Return true if the mask is all ones."""
        if not hasattr(self, "_all_ones"):
            with torch.no_grad():
                self._all_ones = torch.all(self.bool_matrix)
        return self._all_ones

    @property
    def lower_triangular(self):
        """Return true if the attention is a triangular causal mask."""
        if not hasattr(self, "_lower_triangular"):
            self._lower_triangular = False
            with torch.no_grad():
                try:
                    lengths = self.lengths
                    if len(lengths.shape) == 1:
                        target = torch.arange(
                            1,
                            len(lengths)+1,
                            device=lengths.device
                        )
                        self._lower_triangular = torch.all(lengths == target)
                except ValueError:
                    pass
        return self._lower_triangular

class LengthMask(BaseMask):
    """Provide a BaseMask interface for lengths. Mostly to be used with
    sequences of different lengths.

    Arguments
    ---------
        lengths: The lengths as a PyTorch long tensor
        max_len: The maximum length for the mask (defaults to lengths.max())
        device: The device to be used for creating the masks (defaults to
                lengths.device)
    """
    def __init__(self, lengths, max_len=None, device=None):
        self._device = device or lengths.device
        with torch.no_grad():
            self._lengths = lengths.clone().to(self._device)
        self._max_len = max_len or self._lengths.max()

        self._bool_matrix = None
        self._all_ones = torch.all(self._lengths == self._max_len).item()

    @property
    def bool_matrix(self):
        if self._bool_matrix is None:
            with torch.no_grad():
                indices = torch.arange(self._max_len, device=self._device)
                self._bool_matrix = (
                    indices.view(1, -1) < self._lengths.view(-1, 1)
                )
        return self._bool_matrix


"""Local compatibility wrapper for the archived reaction-diffusion scripts.

The original training scripts imported ``NOs_dict.models.CosNO_II`` from a
shared server-side package.  The implementation below maps that public name to
the model implementation archived in this case folder.
"""

from model import SOL


class CosNO_II(SOL):
    def __init__(
        self,
        in_channels,
        modes,
        width,
        bandwidth,
        out_channels=1,
        dim=1,
        skip=True,
        triL=0,
    ):
        super().__init__(
            T=1,
            in_channels=in_channels,
            modes=modes,
            width=width,
            bandwidth=bandwidth,
            out_channels=out_channels,
            dim=dim,
            skip=skip,
            triL=triL,
        )

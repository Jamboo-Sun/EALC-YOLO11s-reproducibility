from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.tasks import DetectionModel


def copy_graph_meta(dst, src):
    for attr in ("i", "f", "type", "np"):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


class EALCStem(nn.Module):
    """
    Original RGB stem
    +
    illumination-adaptive local-contrast residual.

    Residual branch is zero initialized:
    initial forward == original YOLO stem.
    """

    def __init__(
        self,
        base,
        out_channels,
        tau=0.12,
        sharpness=20.0,
        kernel=15,
    ):
        super().__init__()

        self.base = base

        self.tau = float(tau)
        self.sharpness = float(sharpness)
        self.kernel = int(kernel)

        self.residual = nn.Conv2d(
            3,
            out_channels,
            3,
            2,
            1,
            bias=False,
        )

        nn.init.zeros_(
            self.residual.weight
        )

        # sigmoid ~= 0.20
        self.logit_scale = nn.Parameter(
            torch.tensor(-1.3862944)
        )

        self._gate = None

        copy_graph_meta(
            self,
            base
        )

    def set_gate(self, gate):
        self._gate = gate

    def local_contrast(self, x):

        z = torch.log1p(
            4.0 * x.clamp(0, 1)
        )

        mu = F.avg_pool2d(
            z,
            self.kernel,
            stride=1,
            padding=self.kernel // 2
        )

        c = z - mu

        return torch.clamp(
            c,
            -1.0,
            1.0
        )

    def forward(self, x):

        y = self.base(x)

        c = self.local_contrast(x)

        r = self.residual(c)

        if self._gate is None:
            gate = 1.0
        else:
            gate = self._gate

        scale = torch.sigmoid(
            self.logit_scale
        )

        return y + scale * gate * r


class DarkAdaptiveFeatureContrast(nn.Module):
    """
    Feature-space local-contrast adapter for P3/P4.

    base output y
        -> local high-frequency structure
        -> depthwise spatial extraction
        -> pointwise projection
        -> dark-conditioned residual fusion

    Final pointwise projection is zero initialized.
    """

    def __init__(
        self,
        base,
        channels,
        initial_scale=0.15,
    ):
        super().__init__()

        self.base = base
        self.channels = int(channels)

        self.dw = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=channels,
            bias=False,
        )

        self.pw = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False,
        )

        # Spatial extractor can start identity-like,
        # final projection stays zero so original YOLO
        # output is exactly preserved at initialization.
        nn.init.dirac_(
            self.dw.weight
        )

        nn.init.zeros_(
            self.pw.weight
        )

        logit = torch.log(
            torch.tensor(
                initial_scale
                /
                (1.0 - initial_scale)
            )
        )

        self.logit_scale = nn.Parameter(
            logit
        )

        self._gate = None

        copy_graph_meta(
            self,
            base
        )

    def set_gate(self, gate):
        self._gate = gate

    def feature_contrast(self, y):

        local = F.avg_pool2d(
            y,
            kernel_size=3,
            stride=1,
            padding=1
        )

        c = y - local

        # Per-image/channel normalization prevents
        # activation magnitude alone from dominating.
        denom = (
            c.detach()
            .abs()
            .mean(
                dim=(2, 3),
                keepdim=True
            )
            +
            1e-4
        )

        c = c / denom

        return torch.clamp(
            c,
            -3.0,
            3.0
        )

    def forward(self, x):

        y = self.base(x)

        c = self.feature_contrast(y)

        r = self.pw(
            self.dw(c)
        )

        if self._gate is None:
            gate = 1.0
        else:
            gate = self._gate

        scale = torch.sigmoid(
            self.logit_scale
        )

        return y + scale * gate * r


class EALCV2DetectionModel(DetectionModel):

    TAU = 0.12
    SHARPNESS = 20.0

    def __init__(
        self,
        cfg="yolo11s.yaml",
        ch=3,
        nc=None,
        verbose=True,
    ):
        # DetectionModel.__init__ performs a dummy forward
        # to infer Detect strides. Our overridden _predict_once()
        # is therefore called before super().__init__ finishes.
        # Create this plain flag before that first forward.
        object.__setattr__(
            self,
            "ealc_v2_enabled",
            False
        )

        super().__init__(
            cfg=cfg,
            ch=ch,
            nc=nc,
            verbose=verbose
        )

    def enable_ealc_v2(self):

        if self.ealc_v2_enabled:
            return

        # --------------------------
        # Stem
        # --------------------------
        old0 = self.model[0]

        self.model[0] = EALCStem(
            old0,
            out_channels=
                old0.conv.out_channels,
            tau=self.TAU,
            sharpness=self.SHARPNESS,
            kernel=15,
        )

        # --------------------------
        # P3 feature output
        # YOLO11s layer 16 -> 128 ch
        # --------------------------
        old16 = self.model[16]

        self.model[16] = (
            DarkAdaptiveFeatureContrast(
                old16,
                channels=128,
                initial_scale=0.15,
            )
        )

        # --------------------------
        # P4 feature output
        # YOLO11s layer 19 -> 256 ch
        # --------------------------
        old19 = self.model[19]

        self.model[19] = (
            DarkAdaptiveFeatureContrast(
                old19,
                channels=256,
                initial_scale=0.15,
            )
        )

        self.ealc_v2_enabled = True

    def _illumination_gate(self, x):

        L = x.mean(
            dim=(1, 2, 3),
            keepdim=True
        )

        return torch.sigmoid(
            (
                self.TAU
                -
                L
            )
            *
            self.SHARPNESS
        )

    def _set_ealc_gate(self, x):

        # During DetectionModel construction a dummy forward
        # happens before EALC branches are installed.
        if not getattr(
            self,
            "ealc_v2_enabled",
            False
        ):
            return

        gate = self._illumination_gate(
            x
        )

        self.model[0].set_gate(
            gate
        )

        self.model[16].set_gate(
            gate
        )

        self.model[19].set_gate(
            gate
        )

    def _predict_once(
        self,
        x,
        profile=False,
        embed=None,
    ):
        # x is the normalized original image.
        self._set_ealc_gate(x)

        return super()._predict_once(
            x,
            profile=profile,
            embed=embed,
        )

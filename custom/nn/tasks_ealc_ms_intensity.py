"""Full three-position EALC-MS with conservative fixed per-scale multipliers.

Uses original EALCStem and DarkAdaptiveFeatureContrast computational paths.
Only multiplies the final FEATURE residual at P3 and/or P4; the illumination
sigmoid gate stays identical to original MS. The factor is a fixed ablation
hyperparameter, while each original learnable logit_scale remains trainable.
These classes are pickle-importable because defined at module top-level.
"""
from __future__ import annotations
import torch
from ultralytics.nn.tasks_ealc_v2 import (
    EALCV2DetectionModel, EALCStem, DarkAdaptiveFeatureContrast
)

class ScaledFeatureContrast(DarkAdaptiveFeatureContrast):
    def __init__(self, base, channels, initial_scale=0.15, multiplier=1.0):
        super().__init__(base, channels=channels, initial_scale=initial_scale)
        if not 0 < float(multiplier) <= 1.0:
            raise ValueError(f"invalid multiplier: {multiplier}")
        self.multiplier = float(multiplier)

    def forward(self, x):
        # The next operations are identical to the original parent class
        # except for the last constant feature-residual multiplier.
        y = self.base(x)
        c = self.feature_contrast(y)
        r = self.pw(self.dw(c))
        gate = 1.0 if self._gate is None else self._gate
        scale = torch.sigmoid(self.logit_scale)
        return y + self.multiplier * scale * gate * r

class EALCIntensityModel(EALCV2DetectionModel):
    # Concrete subclasses below fix these values and checkpoint identities.
    P3_MULTIPLIER = 1.0
    P4_MULTIPLIER = 1.0

    def enable_ealc_v2(self):
        if self.ealc_v2_enabled:
            return
        old0 = self.model[0]
        self.model[0] = EALCStem(
            old0, out_channels=old0.conv.out_channels,
            tau=self.TAU, sharpness=self.SHARPNESS, kernel=15
        )
        self.model[16] = ScaledFeatureContrast(
            self.model[16], channels=128, initial_scale=0.15,
            multiplier=self.P3_MULTIPLIER
        )
        self.model[19] = ScaledFeatureContrast(
            self.model[19], channels=256, initial_scale=0.15,
            multiplier=self.P4_MULTIPLIER
        )
        self.ealc_v2_enabled = True
        assert isinstance(self.model[0], EALCStem)
        assert isinstance(self.model[16], DarkAdaptiveFeatureContrast)
        assert isinstance(self.model[19], DarkAdaptiveFeatureContrast)

class EALCMSA(EALCIntensityModel):
    """Candidate A: unchanged Stem/P3; P4 contrast residual x0.5."""
    P3_MULTIPLIER = 1.0
    P4_MULTIPLIER = 0.5

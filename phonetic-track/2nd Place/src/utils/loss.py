import torch
import torch.nn as nn


def _prepare_ctc_inputs(
    log_probs: torch.Tensor,
    output_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # sequence dimension must be first for PyTorch CTCLoss
    log_probs_t = log_probs.transpose(0, 1)

    # Ensure lengths don't exceed the actual tensor size due to rounding
    output_lengths = torch.clamp(output_lengths, max=log_probs_t.shape[0])
    return log_probs_t, output_lengths


class CTCLossBatchReduction(nn.Module):
    def __init__(self, blank_token_id: int):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank_token_id,
            zero_infinity=True,
        )  # zero infinity is crucial: does not allow inf. losses

    def forward(
        self,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        output_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        log_probs_t, output_lengths = _prepare_ctc_inputs(log_probs, output_lengths)
        return self.ctc_loss(log_probs_t, labels, output_lengths, target_lengths)


class CTCLossLengthReduction(nn.Module):
    def __init__(self, blank_token_id: int):
        super().__init__()
        self.ctc_loss = nn.CTCLoss(
            blank=blank_token_id,
            reduction="sum",
            zero_infinity=True,
        )  # zero infinity is crucial: does not allow inf. losses

    def forward(
        self,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        output_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            log_probs:      [B, T_enc, Vocab] — model output.
            labels:         [B, max_target_len] — padded phoneme IDs.
            output_lengths: [B] — per-sample encoder output lengths
                            (already computed by model.get_output_lengths).
            target_lengths: [B] — true unpadded phoneme sequence lengths.
        """
        log_probs_t, output_lengths = _prepare_ctc_inputs(log_probs, output_lengths)

        # Compute CTC loss and normalize by total target length to get average per-token loss.
        loss = self.ctc_loss(log_probs_t, labels, output_lengths, target_lengths)
        normalizer = target_lengths.sum().clamp_min(1).to(loss.dtype) # avoid division by zero, lower bound at 1
        return loss / normalizer


# Backward-compatible alias for older config targets.
CTCLoss = CTCLossLengthReduction


class FocalCTCLoss(nn.Module):
    def __init__(self, blank_token_id: int, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ctc_loss = nn.CTCLoss(
            blank=blank_token_id,
            reduction="none",
            zero_infinity=True,
        )

    def forward(
        self,
        log_probs: torch.Tensor,
        labels: torch.Tensor,
        output_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """
        Applies focal weighting to per-sample CTC losses while keeping the final
        reduction normalized by the total target sequence length.
        """
        log_probs_t, output_lengths = _prepare_ctc_inputs(log_probs, output_lengths)

        per_sample_loss = self.ctc_loss(log_probs_t, labels, output_lengths, target_lengths)
        target_lengths_f = target_lengths.clamp_min(1).to(per_sample_loss.dtype)

        # Use length-normalized per-sample loss to derive the focal weight so the
        # difficulty term is not dominated by longer targets.
        normalized_loss = per_sample_loss / target_lengths_f
        pt = torch.exp(-normalized_loss)
        focal_weight = (1.0 - pt).pow(self.gamma)

        if self.alpha is not None:
            focal_weight = focal_weight * self.alpha

        loss = (focal_weight * per_sample_loss).sum()
        normalizer = target_lengths_f.sum().clamp_min(1.0)
        return loss / normalizer

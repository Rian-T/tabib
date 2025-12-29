"""Linear Chain CRF and BIOUL decoder for sequence labeling.

Ported from nlstruct with simplifications for our use case.
Key features:
- Forward-backward algorithm for marginal probabilities
- Viterbi decoding
- BIOUL transition constraints for NER
"""

import torch
import torch.nn as nn


IMPOSSIBLE = -100000


def masked_flip(x: torch.Tensor, mask: torch.Tensor, dim_x: int = -2) -> torch.Tensor:
    """Flip tensor along a dimension, respecting a mask."""
    flipped_x = torch.zeros_like(x)
    flipped_x[mask] = x.flip(dim_x)[mask.flip(-1)]
    return flipped_x


@torch.jit.script
def logdotexp(log_A: torch.Tensor, log_B: torch.Tensor) -> torch.Tensor:
    """Log-space matrix multiplication: log(exp(A) @ exp(B))."""
    # log_A: 2 * N * M
    # log_B: 2 * M * O
    # out: 2 * N * O
    return (log_A.unsqueeze(-1) + log_B.unsqueeze(-3)).logsumexp(-2)


def multi_dim_triu(x: torch.Tensor, diagonal: int = 0) -> torch.Tensor:
    """Apply upper triangular mask to last two dimensions."""
    return x.masked_fill(
        ~torch.ones(x.shape[-2], x.shape[-1], dtype=torch.bool, device=x.device).triu(diagonal=diagonal),
        0
    )


class LinearChainCRF(nn.Module):
    """Linear Chain CRF for sequence labeling.

    Supports:
    - Forward-backward algorithm for marginal probabilities
    - Viterbi decoding for best path
    - Learnable transition parameters
    - Start/end transition constraints
    """

    def __init__(
        self,
        forbidden_transitions: torch.Tensor,
        start_forbidden_transitions: torch.Tensor | None = None,
        end_forbidden_transitions: torch.Tensor | None = None,
        learnable_transitions: bool = True,
        with_start_end_transitions: bool = True,
    ):
        super().__init__()

        num_tags = forbidden_transitions.shape[0]

        self.register_buffer('forbidden_transitions', forbidden_transitions.bool())

        if start_forbidden_transitions is not None:
            self.register_buffer('start_forbidden_transitions', start_forbidden_transitions.bool())
        else:
            self.register_buffer('start_forbidden_transitions', torch.zeros(num_tags, dtype=torch.bool))

        if end_forbidden_transitions is not None:
            self.register_buffer('end_forbidden_transitions', end_forbidden_transitions.bool())
        else:
            self.register_buffer('end_forbidden_transitions', torch.zeros(num_tags, dtype=torch.bool))

        if learnable_transitions:
            self.transitions = nn.Parameter(torch.zeros_like(forbidden_transitions, dtype=torch.float))
        else:
            self.register_buffer('transitions', torch.zeros_like(forbidden_transitions, dtype=torch.float))

        if learnable_transitions and with_start_end_transitions:
            self.start_transitions = nn.Parameter(torch.zeros(num_tags, dtype=torch.float))
            self.end_transitions = nn.Parameter(torch.zeros(num_tags, dtype=torch.float))
        else:
            self.register_buffer('start_transitions', torch.zeros(num_tags, dtype=torch.float))
            self.register_buffer('end_transitions', torch.zeros(num_tags, dtype=torch.float))

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Viterbi decoding to find best tag sequence.

        Args:
            emissions: (batch, seq_len, num_tags) emission scores
            mask: (batch, seq_len) boolean mask

        Returns:
            (batch, seq_len) best tag sequence
        """
        backtrack = self.propagate(emissions, mask, ring_op_name="max", use_constraints=True, way="forward")[2]
        path = [backtrack[-1][0, :, 0]]

        if len(backtrack) > 1:
            backtrack = torch.stack(backtrack[:-1] + backtrack[-2:-1], 2).squeeze(0)
            backtrack[range(len(mask)), mask.sum(1) - 1] = path[-1].unsqueeze(-1)

            # Backward max path following
            for k in range(backtrack.shape[1] - 2, -1, -1):
                path.insert(0, backtrack[:, k][range(len(path[0])), path[0]])

        path = torch.stack(path, -1).masked_fill(~mask, 0)
        return path

    def forward(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        tags: torch.Tensor,
        use_constraints: bool = True,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Compute NLL loss.

        Args:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len)
            tags: (batch, seq_len) or (batch, n_samples, seq_len) for soft targets
            use_constraints: Whether to apply forbidden transitions
            reduction: 'none', 'sum', 'mean', or 'token_mean'

        Returns:
            NLL loss
        """
        z = self.propagate(emissions, mask, ring_op_name="logsumexp", use_constraints=use_constraints)[0]
        posterior_potential = self.propagate(emissions, mask, tags, ring_op_name="posterior", use_constraints=use_constraints)[0]
        nll = posterior_potential - z

        if reduction == 'none':
            return nll
        if reduction == 'sum':
            return nll.sum()
        if reduction == 'mean':
            return nll.mean()
        if reduction == 'token_mean':
            return nll.sum() / mask.float().sum()
        raise ValueError(f"Unknown reduction: {reduction}")

    def marginal(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute marginal probabilities via forward-backward algorithm.

        Args:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len)

        Returns:
            (batch, seq_len, num_tags) log marginal probabilities
        """
        device = emissions.device
        dtype = emissions.dtype  # For mixed precision compatibility

        # Cast transitions to match emissions dtype
        transitions = self.transitions.to(dtype).masked_fill(self.forbidden_transitions, IMPOSSIBLE)
        start_transitions = self.start_transitions.to(dtype).masked_fill(self.start_forbidden_transitions, IMPOSSIBLE)
        end_transitions = self.end_transitions.to(dtype).masked_fill(self.end_forbidden_transitions, IMPOSSIBLE)

        bi_transitions = torch.stack([transitions, transitions.t()], dim=0)

        # Add start transitions
        emissions = emissions.clone()
        emissions[:, 0] = emissions[:, 0] + start_transitions

        # Add end transitions
        batch_indices = torch.arange(mask.shape[0], device=device)
        seq_lengths = mask.long().sum(1) - 1
        emissions[batch_indices, seq_lengths] = emissions[batch_indices, seq_lengths] + end_transitions

        # Stack forward and backward emissions
        bi_emissions = torch.stack([emissions, masked_flip(emissions, mask, dim_x=1)], 1)
        bi_emissions = bi_emissions.transpose(0, 2)

        out = [bi_emissions[0]]
        for k in range(1, len(bi_emissions)):
            res = logdotexp(out[-1], bi_transitions)
            out.append(res + bi_emissions[k])
        out = torch.stack(out, dim=0).transpose(0, 2)

        forward = out[:, 0]
        backward = masked_flip(out[:, 1], mask, dim_x=1)
        backward_z = backward[:, 0].logsumexp(-1)

        return forward + backward - emissions - backward_z[:, None, None]

    def propagate(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor,
        tags: torch.Tensor | None = None,
        ring_op_name: str = "logsumexp",
        use_constraints: bool = True,
        way: str = "forward",
    ):
        """Forward/backward propagation with different semirings.

        Args:
            emissions: (batch, seq_len, num_tags)
            mask: (batch, seq_len)
            tags: Optional target tags
            ring_op_name: 'logsumexp', 'posterior', or 'max'
            use_constraints: Whether to apply forbidden transitions
            way: 'forward' or 'backward'

        Returns:
            (partition, log_probs, backtrack)
        """
        emissions = emissions.transpose(0, 1)
        mask = mask.transpose(0, 1)

        if tags is not None:
            if len(tags.shape) == 2:
                tags = tags.transpose(0, 1).unsqueeze(1)
            elif len(tags.shape) == 3:
                tags = tags.permute(2, 0, 1)

        backtrack = None

        if ring_op_name == "logsumexp":
            def ring_op(last_potential, trans, loc):
                return (last_potential.unsqueeze(-1) + trans.unsqueeze(0).unsqueeze(0)).logsumexp(2)
        elif ring_op_name == "posterior":
            def ring_op(last_potential, trans, loc):
                return trans[tags[loc]] + last_potential[
                    torch.arange(tags.shape[1]).unsqueeze(1),
                    torch.arange(tags.shape[2]).unsqueeze(0),
                    tags[loc]
                ].unsqueeze(-1)
        elif ring_op_name == "max":
            backtrack = []

            def ring_op(last_potential, trans, loc):
                res, indices = (last_potential.unsqueeze(-1) + trans.unsqueeze(0).unsqueeze(0)).max(2)
                backtrack.append(indices)
                return res
        else:
            raise NotImplementedError(f"Unknown ring_op: {ring_op_name}")

        # Cast transitions to match emissions dtype (for mixed precision training)
        dtype = emissions.dtype
        if use_constraints:
            start_transitions = self.start_transitions.to(dtype).masked_fill(self.start_forbidden_transitions, IMPOSSIBLE)
            transitions = self.transitions.to(dtype).masked_fill(self.forbidden_transitions, IMPOSSIBLE)
            end_transitions = self.end_transitions.to(dtype).masked_fill(self.end_forbidden_transitions, IMPOSSIBLE)
        else:
            start_transitions = self.start_transitions.to(dtype)
            transitions = self.transitions.to(dtype)
            end_transitions = self.end_transitions.to(dtype)

        if way == "backward":
            assert ring_op_name != "max", "Backward max not supported"
            start_transitions, end_transitions = end_transitions, start_transitions
            transitions = transitions.t()
            emissions = masked_flip(emissions.transpose(0, 1), mask.transpose(0, 1), -2).transpose(0, 1)

        log_probs = [(start_transitions + emissions[0]).unsqueeze(0).repeat_interleave(
            tags.shape[1] if tags is not None else 1, dim=0
        )]

        for k in range(1, len(emissions)):
            res = ring_op(log_probs[-1], transitions, k - 1)
            log_probs.append(torch.where(
                mask[k].unsqueeze(-1),
                res + emissions[k],
                log_probs[-1]
            ))

        if ring_op_name == "logsumexp":
            z = ring_op(log_probs[-1], end_transitions.unsqueeze(1), 0)
        else:
            z = ring_op(
                log_probs[-1],
                end_transitions.unsqueeze(1),
                ((mask.sum(0) - 1).unsqueeze(0), torch.arange(log_probs[-1].shape[0]).unsqueeze(1), torch.arange(mask.shape[1]).unsqueeze(0))
            ).squeeze(-1)

        log_probs = torch.cat(log_probs, dim=0)

        if way == "backward":
            log_probs = masked_flip(
                log_probs.transpose(0, 1),
                mask.transpose(0, 1),
                dim_x=-2,
            ).transpose(0, 1)

        return z, log_probs, backtrack


class BIOULDecoder(LinearChainCRF):
    """BIOUL CRF decoder for NER.

    Tag scheme: O, I-label, B-label, L-label, U-label
    - O: Outside any entity
    - B: Beginning of multi-token entity
    - I: Inside multi-token entity
    - L: Last token of multi-token entity
    - U: Unit/single-token entity

    Supports:
    - Nested entities (allow_overlap=True)
    - Adjacent entities of same type (allow_juxtaposition=True)
    """

    def __init__(
        self,
        num_labels: int,
        with_start_end_transitions: bool = True,
        allow_overlap: bool = False,
        allow_juxtaposition: bool = True,
        learnable_transitions: bool = True,
    ):
        O, I, B, L, U = 0, 1, 2, 3, 4

        self.allow_overlap = allow_overlap
        self.num_labels = num_labels
        num_tags = 1 + num_labels * 4  # O + (I, B, L, U) per label

        # Build forbidden transition matrix
        forbidden_transitions = torch.ones(num_tags, num_tags, dtype=torch.bool)
        forbidden_transitions[O, O] = 0  # O to O

        for i in range(num_labels):
            STRIDE = 4 * i
            for j in range(num_labels):
                STRIDE_J = j * 4
                forbidden_transitions[L + STRIDE, B + STRIDE_J] = 0  # L-i to B-j
                forbidden_transitions[L + STRIDE, U + STRIDE_J] = 0  # L-i to U-j
                forbidden_transitions[U + STRIDE, B + STRIDE_J] = 0  # U-i to B-j
                forbidden_transitions[U + STRIDE, U + STRIDE_J] = 0  # U-i to U-j

            forbidden_transitions[O, B + STRIDE] = 0  # O to B-i
            forbidden_transitions[B + STRIDE, I + STRIDE] = 0  # B-i to I-i
            forbidden_transitions[I + STRIDE, I + STRIDE] = 0  # I-i to I-i
            forbidden_transitions[I + STRIDE, L + STRIDE] = 0  # I-i to L-i
            forbidden_transitions[B + STRIDE, L + STRIDE] = 0  # B-i to L-i

            forbidden_transitions[L + STRIDE, O] = 0  # L-i to O
            forbidden_transitions[O, U + STRIDE] = 0  # O to U-i
            forbidden_transitions[U + STRIDE, O] = 0  # U-i to O

            if not allow_juxtaposition:
                forbidden_transitions[L + STRIDE, U + STRIDE] = 1  # L-i to U-i
                forbidden_transitions[U + STRIDE, B + STRIDE] = 1  # U-i to B-i
                forbidden_transitions[U + STRIDE, U + STRIDE] = 1  # U-i to U-i
                forbidden_transitions[L + STRIDE, B + STRIDE] = 1  # L-i to B-i

            if allow_overlap:
                forbidden_transitions[L + STRIDE, I + STRIDE] = 0  # L-i to I-i
                forbidden_transitions[L + STRIDE, L + STRIDE] = 0  # L-i to L-i
                forbidden_transitions[I + STRIDE, B + STRIDE] = 0  # I-i to B-i
                forbidden_transitions[B + STRIDE, B + STRIDE] = 0  # B-i to B-i
                forbidden_transitions[B + STRIDE, U + STRIDE] = 0  # B-i to U-i
                forbidden_transitions[U + STRIDE, L + STRIDE] = 0  # U-i to L-i
                forbidden_transitions[U + STRIDE, I + STRIDE] = 0  # U-i to I-i
                forbidden_transitions[I + STRIDE, U + STRIDE] = 0  # I-i to U-i

        # Start/end constraints
        start_forbidden_transitions = torch.zeros(num_tags, dtype=torch.bool)
        end_forbidden_transitions = torch.zeros(num_tags, dtype=torch.bool)

        if with_start_end_transitions:
            for i in range(num_labels):
                STRIDE = 4 * i
                start_forbidden_transitions[I + STRIDE] = 1  # Can't start with I
                start_forbidden_transitions[L + STRIDE] = 1  # Can't start with L
                end_forbidden_transitions[I + STRIDE] = 1  # Can't end with I
                end_forbidden_transitions[B + STRIDE] = 1  # Can't end with B

        super().__init__(
            forbidden_transitions,
            start_forbidden_transitions,
            end_forbidden_transitions,
            with_start_end_transitions=with_start_end_transitions,
            learnable_transitions=learnable_transitions,
        )

    def tags_to_spans(
        self,
        tags: torch.Tensor,
        mask: torch.Tensor | None = None,
        do_overlap_disambiguation: bool = False,
    ) -> torch.Tensor:
        """Convert BIOUL tags to span predictions.

        Args:
            tags: (batch, seq_len) tag indices
            mask: (batch, seq_len) optional mask

        Returns:
            (batch, seq_len, seq_len) boolean span matrix
        """
        I, B, L, U = 0, 1, 2, 3

        if mask is not None:
            tags = tags.masked_fill(~mask, 0)

        unstrided_tags = ((tags - 1) % 4).masked_fill(tags == 0, -1)
        is_B_or_U = (unstrided_tags == B) | (unstrided_tags == U)
        is_L_or_U = (unstrided_tags == L) | (unstrided_tags == U)

        # Only prevent O tag between two bounds
        cs_no_hole = (tags == 0).long().cumsum(1)
        has_no_hole = (cs_no_hole.unsqueeze(-1) - cs_no_hole.unsqueeze(-2)) == 0

        prediction = multi_dim_triu((is_B_or_U).unsqueeze(-1) & (is_L_or_U).unsqueeze(-2) & has_no_hole)

        # If no overlapping, prevent anything other than I between two bounds
        if not self.allow_overlap:
            begin_cs = (is_B_or_U).cumsum(1)
            end_cs = (is_L_or_U).cumsum(1)
            begin_count = begin_cs.unsqueeze(-1) - begin_cs.unsqueeze(-2)
            end_count = end_cs.unsqueeze(-1) - end_cs.unsqueeze(-2)
            prediction &= ((begin_count + end_count) == 0) | ((begin_count + end_count == -1))

        if mask is not None:
            prediction = prediction & mask.unsqueeze(-1) & mask.unsqueeze(-2)

        return prediction

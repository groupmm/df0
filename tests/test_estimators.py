import pytest
import torch

from df0.dyin import dYIN
from df0.dswipe import dSWIPE
from df0.f0_utils import hz_to_cents

FS = 16000
HOP_SIZE = 160
DUR_SEC = 3
F0 = 400.0
N_HARMONICS = 5
ATOL_CENTS = 20.0
F0_CHIRP_START = 100.0
F0_CHIRP_END = 800.0
ATOL_CENTS_CHIRP = 50.0


@pytest.fixture(scope="module")
def harmonic():
    t = torch.arange(0, DUR_SEC * FS) / FS
    return sum(torch.sin(2 * torch.pi * k * F0 * t) / k for k in range(1, N_HARMONICS + 1))


@pytest.fixture(scope="module")
def chirp():
    t = torch.arange(0, DUR_SEC * FS) / FS
    # logarithmic (exponential) chirp: instantaneous frequency = F0_CHIRP_START * (F0_CHIRP_END/F0_CHIRP_START)^(t/DUR_SEC)
    phase = (
        2
        * torch.pi
        * F0_CHIRP_START
        * DUR_SEC
        / torch.log(torch.tensor(F0_CHIRP_END / F0_CHIRP_START))
        * ((F0_CHIRP_END / F0_CHIRP_START) ** (t / DUR_SEC) - 1)
    )
    return torch.sin(phase)


def chirp_f0_ref(n_frames):
    """Ground truth instantaneous F0 at each analysis frame."""
    t = torch.arange(n_frames) * HOP_SIZE / FS
    return F0_CHIRP_START * (F0_CHIRP_END / F0_CHIRP_START) ** (t / DUR_SEC)


@pytest.fixture(scope="module")
def dyin_model():
    model = dYIN(
        fs=FS,
        frame_size=FS // 10,
        hop_size=HOP_SIZE,
        f0_min=55.0,
        f0_max=1760.0,
        f0_r_cent=10.0,
        f0_selection_strategy="argmax",
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def dswipe_model():
    model = dSWIPE(
        fs=FS,
        hop_size=HOP_SIZE,
        f0_min=55.0,
        f0_max=1760.0,
        f0_r_cent=10.0,
        f0_selection_strategy="argmax",
    )
    model.eval()
    return model


class Test_dYIN:
    def test_output_shapes(self, dyin_model, harmonic):
        with torch.no_grad():
            out = dyin_model(harmonic)
        n_frames = out["f0_hz"].shape[0]
        n_classes = dyin_model.f0_classes_hz.numel()
        assert out["logits"].shape == (n_frames, n_classes)
        assert out["probs"].shape == (n_frames, n_classes)
        assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(n_frames), atol=1e-5)

    def test_f0_accuracy(self, dyin_model, harmonic):
        with torch.no_grad():
            out = dyin_model(harmonic)
        pred_cents = hz_to_cents(out["f0_hz"][1:-1], f_ref=55.0)
        ref_cents = hz_to_cents(torch.tensor(F0), f_ref=55.0).expand_as(pred_cents)
        assert torch.allclose(pred_cents, ref_cents, atol=ATOL_CENTS)

    def test_chirp_accuracy(self, dyin_model, chirp):
        with torch.no_grad():
            out = dyin_model(chirp)
        pred = out["f0_hz"][4:-4]
        ref = chirp_f0_ref(out["f0_hz"].shape[0])[4:-4]
        assert torch.allclose(
            hz_to_cents(pred, f_ref=55.0), hz_to_cents(ref, f_ref=55.0), atol=ATOL_CENTS_CHIRP
        )

    def test_integer_assertions(self):
        with pytest.raises(AssertionError):
            dYIN(fs=16000.0)
        with pytest.raises(AssertionError):
            dYIN(frame_size=1024.0)
        with pytest.raises(AssertionError):
            dYIN(hop_size=160.0)


class Test_dSWIPE:
    def test_output_shapes(self, dswipe_model, harmonic):
        with torch.no_grad():
            out = dswipe_model(harmonic)
        n_frames = out["f0_hz"].shape[0]
        n_classes = dswipe_model.f0_classes_hz.numel()
        assert out["logits"].shape == (n_frames, n_classes)
        assert out["probs"].shape == (n_frames, n_classes)
        assert torch.allclose(out["probs"].sum(dim=-1), torch.ones(n_frames), atol=1e-5)

    def test_f0_accuracy(self, dswipe_model, harmonic):
        with torch.no_grad():
            out = dswipe_model(harmonic)
        pred_cents = hz_to_cents(out["f0_hz"][4:-4], f_ref=55.0)
        ref_cents = hz_to_cents(torch.tensor(F0), f_ref=55.0).expand_as(pred_cents)
        assert torch.allclose(pred_cents, ref_cents, atol=ATOL_CENTS)

    def test_chirp_accuracy(self, dswipe_model, chirp):
        with torch.no_grad():
            out = dswipe_model(chirp)
        pred = out["f0_hz"][4:-4]
        ref = chirp_f0_ref(out["f0_hz"].shape[0])[4:-4]
        assert torch.allclose(
            hz_to_cents(pred, f_ref=55.0), hz_to_cents(ref, f_ref=55.0), atol=ATOL_CENTS_CHIRP
        )

    def test_batch_input(self, dswipe_model, harmonic):
        x_batch = harmonic.unsqueeze(0).expand(2, -1)
        with torch.no_grad():
            out = dswipe_model(x_batch)
        assert out["f0_hz"].shape[0] == 2

    def test_integer_assertions(self):
        with pytest.raises(AssertionError):
            dSWIPE(fs=16000.0)
        with pytest.raises(AssertionError):
            dSWIPE(hop_size=160.0)

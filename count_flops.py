import torch
# from ptflops import get_model_complexity_info
from thop import profile, clever_format

# Import your model and dependencies
from src.models_pack import DCDMUSIC, SubspaceNet, DeepCNN, TransMUSIC
from src.system_model import SystemModel, SystemModelParams
from src.config import device
import time
import os

if device.type == "cpu": # turn off acceleration
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
from src.utils import set_unified_seed
set_unified_seed()

# If your model requires additional arguments in forward (e.g., number_of_sources),
# wrap it in a custom nn.Module as shown below:
class Wrappedmodel(torch.nn.Module):
    def __init__(self, model, number_of_sources):
        super().__init__()
        self.model = model
        self.number_of_sources = number_of_sources

    def forward(self, x):
        return self.model(x, self.number_of_sources)
    
    def __repr__(self):
        return self.model.__repr__()

    def __str__(self):
        return self.model.__str__()
    
    def _get_name(self):
        return self.model._get_name()

# =============================
# Fill in your parameters below
# =============================
N = 64
M = 2
T = 100
snr = 10
field_type = "near"
signal_nature = "non-coherent"
signal_type = "narrowband"
eta = 0.0
bias = 0.0
sv_noise_var = 0.0
doa_range = 60
doa_resolution = 0.5
max_range_ratio_to_limit = 0.5
range_resolution = 1
wavelength = 0.06
tau = 8
system_model_params = (
        SystemModelParams()
        .set_parameter("N", N)
        .set_parameter("M", M)
        .set_parameter("T", T)
        .set_parameter("snr", snr)
        .set_parameter("field_type", field_type)
        .set_parameter("signal_nature", signal_nature)
        .set_parameter("signal_type", signal_type)
        .set_parameter("eta", eta)
        .set_parameter("bias", bias)
        .set_parameter("sv_noise_var", sv_noise_var)
        .set_parameter("doa_range", doa_range)
        .set_parameter("doa_resolution", doa_resolution)
        .set_parameter("max_range_ratio_to_limit", max_range_ratio_to_limit)
        .set_parameter("range_resolution", range_resolution)
        .set_parameter("wavelength", wavelength)
    )
system_model = SystemModel(
    system_model_params
)

dcd_music = DCDMUSIC(
    system_model,
    tau=tau,
    diff_method=("esprit", "music_1d"),  
)

nfssn = SubspaceNet(
    system_model=system_model,
    tau=tau,
    diff_method="music_2D", 
    field_type="near",
    train_loss_type="music_spectrum",
    variant="small",
    norm_layer=True,
    psd_epsilon=1e-6,
    batch_norm=False,
    skip_connection=False,
)

wrapped_dcd = Wrappedmodel(dcd_music, M)
wrapped_nfssn = Wrappedmodel(nfssn, M)
deep_cnn = DeepCNN(system_model=system_model).to(device)
trans_music = TransMUSIC(system_model=system_model).to(device)

# Define the input shape (excluding batch size)
# Example: (2, 128) for 2 sensors, 128 snapshots
input_shape = (N, T)
# Create a dummy input tensor with the correct shape
input_tensor = torch.randn((1,) + input_shape, dtype=torch.complex64).to(device)

def count_flops_and_timing(model, input_tensor):
    model.to(device)
    model.eval()
    print(f"{model._get_name()}")
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)

    macs, params = clever_format([flops, params], '%.3f')

    print('{:<30}  {:<8}'.format('Computational complexity:', macs))
    print('{:<30}  {:<8}'.format('Number of parameters:', params)) 

    # Warm-up (especially important for GPU)
    for _ in range(10):
        _ = model(input_tensor)

    # Timing
    n_runs = 500
    torch.cuda.synchronize() if device.type == "cuda" else None
    start = time.time()
    for _ in range(n_runs):
        _ = model(input_tensor)
    torch.cuda.synchronize() if device.type == "cuda" else None
    end = time.time()

    avg_time = (end - start) / n_runs
    print(f"Average inference time per run: {avg_time * 1000:.3f} ms")

# =============================
# Count FLOPs and timing
# =============================

count_flops_and_timing(wrapped_dcd, input_tensor)
count_flops_and_timing(wrapped_nfssn, input_tensor)
count_flops_and_timing(deep_cnn, input_tensor)
count_flops_and_timing(trans_music, input_tensor)
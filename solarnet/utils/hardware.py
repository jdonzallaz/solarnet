import psutil
import torch
import platform


def gpu_info() -> str:
    """
    Get info about GPU support and available GPUs.

    :return: a string with GPU support info
    """

    if torch.cuda.is_available():
        return f"{torch.cuda.device_count()}x {torch.cuda.get_device_name()}"
    return "Cuda not available"


def machine_summary() -> str:
    """
    Get a summary of the machine OS / hardware

    :return: a string with a machine summary
    """

    uname = platform.uname()
    cores = psutil.cpu_count(logical=False)
    freq = f"{psutil.cpu_freq().max:.2f}Mhz"
    ram = str(round(psutil.virtual_memory().total / (1024.0 ** 3))) + " GB"

    return f"{uname.node} | {uname.system} {uname.version} | {cores} cores @ {freq} | RAM {ram} | {gpu_info()}"

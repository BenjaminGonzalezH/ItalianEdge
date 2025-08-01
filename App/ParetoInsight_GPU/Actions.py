######### Libraries #########
import cupy as cp                                   # Efficient Math Operations (GPU).
import numpy as np                                  # Efficient Math Operations (CPU).

######### Functions #########

"""
This block contains all main functions.
"""

def TransformMathStructure(
        Structure: np.ndarray | cp.ndarray,
        Origin: str = "CPU") -> np.ndarray | cp.ndarray:
    """
    TransformMathStructure (function): Transform cupy/numpy structure into
    numpy/cupy data type.
    
    Parameters:
    - Structure: Math structure (array) to transform.

    Returns:
    - Converted: Structure transformed into np.ndarray.
    """
    try:
        match Origin.upper():

            ######################################### CPU
            case "CPU":
                Converted = cp.asarray(Structure)
            
            ######################################### GPU
            case "GPU":
                Converted = cp.asnumpy(Structure)

    except Exception as e:
        print(f"Unexpected error: {e}")
    else:
        return Converted

def GPU_MemoryPool(
        limit_bytes: int | None) -> None:
    """
    InitializeGPU (function): Reserve a initial memory pool for GPU in order to
    avoid initial bottleneck of cupys functions. 
    
    Parameters:
    - limit_bytes: limit of bytes that you wanna use.

    Returns:
    - None; just initialize memory.
    """
    try:
        mempool = cp.get_default_memory_pool()
        if limit_bytes is not None:
            if isinstance(limit_bytes, int) and limit_bytes > 0:
                mempool.set_limit(size=limit_bytes)
            else:
                raise ValueError("limit_bytes must be a positive integer or None.")
        _ = cp.empty((10,))
        print("Bytes usados actualmente:", mempool.used_bytes())
        print("Bytes reservados en el pool:", mempool.total_bytes())

    except ValueError as ve:
        raise ValueError(ve)
    except Exception as e:
        print(f"Unexpected error: {e}")
    else:
        return mempool

def GPU_freeMemoryPool(
        mempool: cp.cuda.MemoryPool) -> None:
    """
    GPU_freeMemoryPool (function) Free all memory blocks used by CuPy.

    Parameters:
    - mempool: 

    Returns:
    - None; just initialize memory.
    """
    try:
        mempool.free_all_blocks()

    except Exception as e:
        raise RuntimeError(f"Unexpected error: {e}")

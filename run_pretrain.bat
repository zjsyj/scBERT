@echo off
:: scBERT Pretraining Launch Script

echo Updating scBERT package...
pip install -e ".[dev]"
if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Failed to update scBERT package.
    pause
    exit /b %ERRORLEVEL%
)

:: 1. Configuration - Edit these paths as needed
set DATA_PATH="../scRNA_20_mouse_Tabula_Muris/downsampled_scRNA_20_mouse.h5ad"
set MODEL_PATH="../scRNA_20_mouse_Tabula_Muris/pretrained_model"

:: 1. Environment Fix for Windows
set USE_LIBUV=0

:: 2. Configuration
set NUM_GPUS=1

echo Starting pretraining for scBERT...
echo [INFO] USE_LIBUV set to 0 to prevent DistStoreError.

set MASTER_ADDR=127.0.0.1
set MASTER_PORT=29500
set WORLD_SIZE=1
set RANK=0

:: 2. Execution
:: Note: finetune.py has been refactored to pretrain.py as requested
python -m scBERT.pretrain_with_GPU ^
    --data_path %DATA_PATH% ^
    --ckpt_dir %MODEL_PATH% ^
    --gene_num 23434 ^
    --epoch 2 ^
    --model_name scRNA_20_mouse_Tabula_Muris_pretrained_model ^
    --seed 2026

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Pretraining failed with exit code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo [SUCCESS] Pretraining completed.
pause

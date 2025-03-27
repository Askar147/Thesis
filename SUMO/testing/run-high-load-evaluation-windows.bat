@echo off
REM Script to run high-load evaluation of TE-DQN and FF-DQN models for Windows

REM Path to your models and data
SET TRANSFORMER_MODEL=best_model\best_model.pt
SET FF_DQN_MODEL=best_model\best_model.pth
SET CONFIG_FILE=config.json
SET ENERGY_CSV=merged_dag1.csv

REM Output directory (with timestamp)
FOR /F "tokens=1-4 delims=/ " %%A IN ('date /t') DO (SET DATE=%%D%%B%%C)
FOR /F "tokens=1-2 delims=: " %%A IN ('time /t') DO (SET TIME=%%A%%B)
SET OUTPUT_DIR=high_load_results_%DATE%_%TIME%

REM Create output directory
mkdir %OUTPUT_DIR%

REM Set evaluation parameters
SET EPISODES=10
SET LOAD_FACTOR=3.0

echo Starting high-load evaluation with load factor %LOAD_FACTOR%x
echo Outputs will be saved to: %OUTPUT_DIR%

REM Run single load factor evaluation
python vec-high-load-evaluation.py ^
    --transformer_model %TRANSFORMER_MODEL% ^
    --dqn_model %FF_DQN_MODEL% ^
    --config %CONFIG_FILE% ^
    --energy_csv %ENERGY_CSV% ^
    --load_factor %LOAD_FACTOR% ^
    --output_dir %OUTPUT_DIR% ^
    --episodes %EPISODES%

echo.
echo Do you want to run varying load evaluation? (y/n)
set /p run_varying=

if "%run_varying%"=="y" (
    REM Run varying load evaluation
    mkdir %OUTPUT_DIR%\varying_load
    python vec-high-load-evaluation.py ^
        --transformer_model %TRANSFORMER_MODEL% ^
        --dqn_model %FF_DQN_MODEL% ^
        --config %CONFIG_FILE% ^
        --energy_csv %ENERGY_CSV% ^
        --output_dir "%OUTPUT_DIR%\varying_load" ^
        --episodes 5 ^
        --varying_load ^
        --load_factors "1.0,1.5,2.0,3.0,4.0"
    
    echo Varying load evaluation completed.
)

echo All evaluations completed. Results saved to: %OUTPUT_DIR%
pause
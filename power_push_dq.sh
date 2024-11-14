set -e

cd /mnt/public/home/wasp_dpm/yl_dir/software/powerforecast_alg

nohup /mnt/public/home/wasp_dpm/yl_dir/env/miniconda3/condabin/conda run -n powerforecast python power_push_dq.py >> runlog/shandong_push.log 2>&1 </dev/null 
echo "功率推送"

wait

# nohup /home/gpusr/miniconda3/condabin/conda run -n forecast_power_gpu python ecmwf_weather_push.py >> runlog/jiangsu_yancheng_binhaiH2_weather.log 2>&1 </dev/null 
# echo "气象推送"

current_time3=$(date "+%Y-%m-%d %H:%M:%S")
echo "$current_time3 Finished"
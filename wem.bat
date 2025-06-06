@echo off

set config_path=%1
set enable_logs=%2
set enable_real_time_views=%3

python wem_app/wem_main.py %config_path% %enable_logs% %enable_real_time_views%
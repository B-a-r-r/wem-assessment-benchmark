@echo off

set config_path=%1
set enable_logs=%2

python wem_app/wem_main.py %config_path% %enable_logs%
@echo off
if /i "%1" == "clean" goto clean
goto all

:all
python setup.py build_ext --inplace
rd /s /q build

goto exit



:clean
del /f /s /q *.cpp
del /f /s /q *.c
del /f /s /q *.pyd

goto exit

:exit

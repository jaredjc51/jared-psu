:: currentfilepath should be where the file is saved on the local disk.
:: newfilepath should be the location in Box drive
@echo off
set currentfilepath="C:\Users\jjc407\Desktop\SD05Fe2a_200s_pole2.diel"
set newfilepath="C:\Users\jjc407\Box\AFOSR_Randall\Measurements\Impedance\BC10b\b1d"
:loop
xcopy %currentfilepath% %newfilepath% /y /q
timeout /t 600 /nobreak
goto :loop
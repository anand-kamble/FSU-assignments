@echo off
set /p userInput= Git cha msg type kar : 

REM Example command using the user input
echo SomeCommand %userInput%

git add .
git commit -m " %userInput% "
git push origin main
REM Add more commands as needed

pause
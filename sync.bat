@echo off
git status
set /p userInput= Git cha msg type kar : 

echo You entered: %userInput%

REM Example command using the user input
echo Performing a command with the user input: 
echo SomeCommand %userInput%

git add .
git commit -m " %userInput% "
git push origin main
REM Add more commands as needed

pause
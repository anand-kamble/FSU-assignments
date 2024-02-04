@echo off
echo Syncing the repository with the remote repository
echo --------------------------------------------------
set /p userInput= Enter the commit message : 
echo --------------------------------------------------
REM Example command using the user input
echo Adding the commit as : %userInput%

git add .
echo Files added to the staging area
git commit -m " %userInput% "
echo Commit added to the repository
git push origin main
echo Repository synced with the remote repository

REM Add more commands as needed

pause
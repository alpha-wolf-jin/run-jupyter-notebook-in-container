# run-jupyter-notebook-in-container

echo "# run-jupyter-notebook-in-container" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/alpha-wolf-jin/run-jupyter-notebook-in-container.git

git config --global credential.helper 'cache --timeout 72000'

git push -u origin main

git add . ; git commit -a -m "update README" ; git push -u origin main

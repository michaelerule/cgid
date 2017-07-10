#!/usr/bin/env bash

shopt -s extglob
echo running autodoc
cd ./docs
sphinx-apidoc -fe -o . ../
make clean html
cd ../

# removes everything that shouldn't be being tracked as per
# .gitignore. Possibly dangerous. 
cat .gitignore | awk "/^[.\*]/" | sed 's/"/"\\""/g;s/.*/"&"/' |  xargs -E '' -I{} git rm -rf --cached {}
git rm -rf --cached *.pyc
git add . 
git add -u :/
git commit -m "$1"
git push origin master

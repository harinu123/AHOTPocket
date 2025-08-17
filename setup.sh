#!/bin/bash

cd data
git clone https://github.com/Mickdub/gvp.git
cd gvp
git checkout pocket_pred
conda env create -f pocketminer.yml
cp ../pocketminer.py src
cd ..

tar -xvf hotpocket_data.tar.gz
mv hotpocket_data/* .
rm -rf hotpocket_data
cd ..

echo "setup complete!"
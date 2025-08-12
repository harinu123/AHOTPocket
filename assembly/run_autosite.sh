grep ^ATOM $1 > ${2}_truncated.pdb
obabel ${2}_truncated.pdb -O ${2}.pdbqt -xr -p 7.4
autosite -r ${2}.pdbqt --numpysave out.npy

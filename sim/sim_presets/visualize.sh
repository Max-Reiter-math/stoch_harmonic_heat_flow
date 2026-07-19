python -m sim.run -vtx 1 -fsr 0.01 -msr 0.01 -ovw 1 -sid l36 -lam 1 -dh 10 -dt 0.01 -e unstable -m linear_cg -cs 1 -seed 36 || true
python -m sim.run -vtx 1 -fsr 0.01 -msr 0.01 -ovw 1 -sid l60 -lam 1 -dh 10 -dt 0.01 -e unstable -m linear_cg -cs 1 -seed 60 || true
python -m sim.run -vtx 1 -fsr 0.01 -msr 0.01 -ovw 1 -sid l98 -lam 1 -dh 10 -dt 0.01 -e unstable -m linear_cg -cs 1 -seed 98 || true
python -m sim.run -vtx 1 -fsr 0.01 -msr 0.01 -ovw 1 -sid l100 -lam 1 -dh 10 -dt 0.01 -e unstable -m linear_cg -cs 1 -seed 100 || true
python -m sim.run -vtx 1 -fsr 0.01 -msr 0.005 -ovw 1 -sid n36 -lam 1 -dh 10 -dt 0.01 -e unstable -m nonlin_cg -cs 1 -seed 36 || true
python -m sim.run -vtx 1 -fsr 0.01 -msr 0.005 -ovw 1 -sid n60 -lam 1 -dh 10 -dt 0.01 -e unstable -m nonlin_cg -cs 1 -seed 60 || true
python -m sim.run -vtx 1 -fsr 0.01 -msr 0.005 -ovw 1 -sid n98 -lam 1 -dh 10 -dt 0.01 -e unstable -m nonlin_cg -cs 1 -seed 98 || true
python -m sim.run -vtx 1 -fsr 0.01 -msr 0.005 -ovw 1 -sid n100 -lam 1 -dh 10 -dt 0.01 -e unstable -m nonlin_cg -cs 1 -seed 100 || true
python -m sim.run -vtx 1 -fsr 0.01 -msr 0.005 -ovw 1 -sid fp36 -lam 1 -dh 10 -dt 0.01 -e unstable -m fp_coupled -cs 1 -seed 36 || true
python -m sim.run -vtx 1 -fsr 0.01 -msr 0.005 -ovw 1 -sid fp60 -lam 1 -dh 10 -dt 0.01 -e unstable -m fp_coupled -cs 1 -seed 60 || true
python -m sim.run -vtx 1 -fsr 0.01 -msr 0.005 -ovw 1 -sid fp98 -lam 1 -dh 10 -dt 0.01 -e unstable -m fp_coupled -cs 1 -seed 98 || true
python -m sim.run -vtx 1 -fsr 0.01 -msr 0.005 -ovw 1 -sid fp100 -lam 1 -dh 10 -dt 0.01 -e unstable -m fp_coupled -cs 1 -seed 100
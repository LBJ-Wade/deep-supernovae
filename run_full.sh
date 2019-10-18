#!/bin/bash
for i in `seq 1 1`
do
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge full
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge full -nonrep
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge full -nonrep --addrep 50
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge full -nonrep --addrep 100
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge full -nonrep --addrep 200
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM -nohostz --challenge full
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell LSTM --challenge full
	python main.py --hidden 128 128 -augment --test_fraction 0.93143 --cell PLSTM --challenge full --dataset des
	python main.py --hidden 128 128 -augment --test_fraction 0.5 --cell PLSTM --challenge full
	python main.py --hidden 128 128 -augment --test_fraction 0.5 --cell PLSTM -nohostz --challenge full
done
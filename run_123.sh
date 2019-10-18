#!/bin/bash
for i in `seq 1 1`
do
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge 123
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge 123 -nonrep
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge 123 -nonrep --addrep 50
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge 123 -nonrep --addrep 100
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge 123 -nonrep --addrep 200
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM -nohostz --challenge 123
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell LSTM --challenge 123
	python main.py --hidden 128 128 -augment --test_fraction 0.93143 --cell PLSTM --challenge 123 --dataset des
	python main.py --hidden 128 128 -augment --test_fraction 0.5 --cell PLSTM --challenge 123
	python main.py --hidden 128 128 -augment --test_fraction 0.5 --cell PLSTM -nohostz --challenge 123
done
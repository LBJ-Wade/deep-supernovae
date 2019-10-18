#!/bin/bash
for i in `seq 1 1`
do
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge sn1a
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge sn1a -nonrep
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge sn1a -nonrep --addrep 50
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge sn1a -nonrep --addrep 100
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM --challenge sn1a -nonrep --addrep 200
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell PLSTM -nohostz --challenge sn1a
	python main.py --hidden 128 128 -augment --test_fraction 0.94827 --cell LSTM --challenge sn1a
	python main.py --hidden 128 128 -augment --test_fraction 0.93143 --cell PLSTM --challenge sn1a --dataset des
	python main.py --hidden 128 128 -augment --test_fraction 0.5 --cell PLSTM --challenge sn1a
	python main.py --hidden 128 128 -augment --test_fraction 0.5 --cell PLSTM -nohostz --challenge sn1a
done
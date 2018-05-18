Enviroment: Python 3.5.2
Package: scipy + numpy

For gen.py:
    Execute gen.py, and it will produce test_gen.txt(test pattern).
	And do comparsion between equations results with X*W.
	
	#number of components(dic_index) to build filter
	s = 3
	kh = 3
	kw = 3

	# index number of dic
	k = 100

	# number of input channel
	m = 64

	# number of output channel
	n = 1

	# number of input image X
	h = 14
	w = 14

	test_gen.txt order:
	#D;X;W;S;P;C;I;X*W;
	D: 100x64
	X: 64x14x14
	W: 1x64x3x3
	S: 100x14x14
	C: 1x3x3x3
	I: 1x3x3x3
	X*W: 1x14x14 

For gen_error_sim.py:
	Execute gen_error_sim.py.py, and it will calculate the error 
        between the results from original equations and approximate equations.
	Line 30 is truncated bit: it can be 0~16, an integer.




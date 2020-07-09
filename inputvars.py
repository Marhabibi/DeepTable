import sys
import getopt

def inputvarsTrain(av,g):
	try:
		opts,args = getopt.getopt(av,"he:v:l:i:o:",["epoch=","embedding=", "learning_rate=","input_file=","output_dir="])
	except:
		print(g,"-e <epoch> -v <embedding> -l <learning_rate> -i <input_file> -o <output_dir>")
		sys.exit(2)
	if len(opts)<5:
		print(g,"-e <epoch> -v <embedding> -l <learning_rate> -i <input_file> -o <output_dir>")
		sys.exit(2)
	epoch = ""
	lr = ""
	out = ""
	emb = ""
	inp = ""
	for opt,arg in opts:
		if opt == "-h":
			print(g,"-e <epoch> -v <embedding> -l <learning_rate> -i <input_file> -o <output_dir>")
			sys.exit(2)
		elif opt in ("-e","--epoch"):
			epoch = arg
		elif opt in ("-v","--embedding"):
			emb = arg
		elif opt in ("-i","--input_file"):
			inp = arg
		elif opt in ("-o","--output_dir"):
			out = arg
		elif opt in("-l","--learning_rate"):
			lr = arg
	return epoch,out,emb,lr,inp


def inputvarsEval(av,g):
	try:
		opts,args = getopt.getopt(av,"hm:i:o:",["model=","input=","output="])
	except:
		print(g,"-m <model> -i <input> -o <output>")
		sys.exit(2)
	outp = ""
	model = ""
	inp = ""
	if len(opts) < 3:
		print(g,"-m <model> -i <input> -o <output>")
		sys.exit(2)
	for opt,arg in opts:
		if opt == "-h":
			print(g,"-m <model> -i <input> -o <output>")
			sys.exit(2)
		elif opt in ("-o","--output"):
			outp = arg
		elif opt in ("-i","--input"):
			inp = arg
		elif opt in ("-m","--model"):
			model = arg
	print(outp)
	return model,outp,inp
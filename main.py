import argparse
from RecSysBaseLine import RecSysBaseLine
from RecSysAdv import RecSysAdv
from EvaSys import EvaSys

parser = argparse.ArgumentParser(
        description='Evaluate the recommender system with the ratings given.')
parser.add_argument('model', metavar='model', type=str, nargs=1,
                    choices=['BLRS', 'ARS'],
                    help='The recommender system to evaluate ("BLRS"|"ARS")')
parser.add_argument('-f', dest='files', metavar='file', type=str, nargs='+',
                    help='The path(s) of the rating file(s). If multiple \
                            files are specifiled, they will be treated as \
                            split files and be applied directly to k-fold \
                            evaluation manner, where k euqals the number of \
                            files you specifiled.')
parser.add_argument('-k', dest='ks', metavar='K', type=int, nargs='+',
                    help='The value of K(s) of the evaluation matrics P@K, \
                            R@K, etc')

args = parser.parse_args()
model = args.model[0]
files = args.files
ks = args.ks

if model == "BLRS":
    RS = RecSysBaseLine()
else:
    RS = RecSysAdv()

ES = EvaSys()
if len(files) == 1:
    ES.load_total_ratings(files[0])
else:
    ES.load_split_ratings(files)

print()
print("###### Evaluating     ", model)
print("###### Rating File(s) ", files)
print("###### K(s)           ", ks)
ES.evaluate(RS, ks)

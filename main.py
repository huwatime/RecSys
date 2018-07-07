import argparse
from RecSysBaseLine import RecSysBaseLine
from RecSysAdv import RecSysAdv
from EvaSys import EvaSys

parser = argparse.ArgumentParser(
        description='Evaluate the recommender system with the ratings given.')
parser.add_argument('model', metavar='model', type=str,
                    choices=['BLRS', 'ARS'],
                    help='The recommender system to be evaluated \
                            ("BLRS"|"ARS")')
parser.add_argument('-f', dest='files', metavar='file', type=str, nargs='+',
                    required=True,
                    help='The path(s) of the rating file(s). If multiple \
                            files are specifiled, they will be treated as \
                            split files and be applied directly to n-fold \
                            evaluation manner, where n euqals the number of \
                            files you specifiled.')
parser.add_argument('-k', dest='ks', metavar='K', type=int, nargs='+',
                    required=True,
                    help='The value of K(s) of the evaluation matrics P@K, \
                            R@K, etc.')
parser.add_argument('-r', dest='rank', metavar='rank', type=int, default=50,
                    help='The rank for U, V in matrix factorization. Only \
                            applied in ARS.')
parser.add_argument('-n', dest='num_fold', metavar='num_fold', type=int,
                    default=5,
                    help='The number of folds in n-fold evaluation manner. \
                            If multiple rating files are specifiled, this \
                            number will be ingnored.')

args = parser.parse_args()
model = args.model
files = args.files
ks = args.ks
rank = args.rank
num_fold = args.num_fold

if model == "BLRS":
    RS = RecSysBaseLine()
else:
    RS = RecSysAdv()
    RS.set_rank(rank)

ES = EvaSys()
if len(files) == 1:
    ES.load_total_ratings(num_fold, files[0])
else:
    ES.load_split_ratings(files)

print()
print("###### Evaluating     ", model)
print("###### Rating File(s) ", files)
print("###### K(s)           ", ks)
ES.evaluate(RS, ks)

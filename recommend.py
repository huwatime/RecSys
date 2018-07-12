import argparse
from RecSysBaseLine import RecSysBaseLine
from RecSysAdv import RecSysAdv

parser = argparse.ArgumentParser(
        description='Make recommendation by different recommender systems. \
                    You can recommend top-k items to a certain user, or \
                    predict the rating of a certain item to a certain user.')
parser.add_argument('-m', dest='model', metavar='model', type=str,
                    required=True, choices=['BLRS', 'ARS'],
                    help='The recommender system to use ("BLRS"|"ARS")')
parser.add_argument('-f', dest='file', metavar='file', type=str,
                    required=True,
                    help='The path of the rating file.')
parser.add_argument('-r', dest='rank', metavar='rank', type=int, default=50,
                    help='The rank for U, V in matrix factorization. Only \
                            applied in ARS.')
parser.add_argument('-u', dest='user_id', metavar='user_id', type=str,
                    required=True,
                    help='The user ID you want to recomend for.')
parser.add_argument('-k', dest='top_k', metavar='top_k', type=int,
                    help='The number of items you want to recommend. Only \
                            required in top-k item recommendation.')
parser.add_argument('-i', dest='item_id', metavar='item_id', type=str,
                    help='The item ID you want to predict the rating for. \
                            Only required in rating prediction.')

args = parser.parse_args()

print("\n### recommendation model:", args.model)
if args.model == "BLRS":
    RS = RecSysBaseLine()
else:
    RS = RecSysAdv()
    RS.set_rank(args.rank)

RS.load_ratings(args.file)
user_idx = RS.get_user_idx(args.user_id)

if args.top_k is not None:
    predictions = RS.predict_top_k_recomm(user_idx, args.top_k)
    print("### Predict Top {} items for user {}:".format(
        args.top_k, args.user_id))
    for i, (item_idx, score) in enumerate(predictions):
        item_id = RS.get_item_id(int(item_idx))
        print("    [{}] {} {:.3f}".format(i+1, item_id, score))

if args.item_id is not None:
    item_idx = RS.get_item_idx(args.item_id)
    rating = RS.predict_rating(user_idx, item_idx)
    print("### Predicted rating of item {} for user {}: {:.3f}\n".format(
        args.item_id, args.user_id, rating))

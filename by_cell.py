from __future__ import print_function

import os

from datasets import NCI60
from argparser import get_parser
from skwrapper import regress, classify, summarize


def test1():
    df = NCI60.load_by_cell_data()
    regress('XGBoost', df)


def test2():
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=20)
    df = NCI60.load_by_cell_data()
    regress(model, df, cv=2)


def main():
    # Initialize the parser
    description = 'Build ML models to predict by-cellline drug response.'
    parser = get_parser(description)
    args = parser.parse_args()

    print('Args:', args, end='\n\n')
    print('Use percent growth for dose levels in log concentration range: [{}, {}]'.format(args.min_logconc, args.max_logconc))
    print('\nCap percent growth: [{}, {}]\n'.format(args.min_growth_bound, args.max_growth_bound))

    cells = NCI60.all_cells() if 'all' in args.cells else args.cells

    for cell in cells:
        print('-' * 10, 'Cell line:', cell, '-' * 10)
        df = NCI60.load_by_cell_data(cell=cell, drug_features=args.drug_features, scaling=args.scaling,
                                     min_logconc=args.min_logconc, max_logconc=args.max_logconc,
                                     subsample=args.subsample, feature_subsample=args.feature_subsample,
                                     thres_frac_rows=args.thres_frac_rows, thres_frac_cols=args.thres_frac_cols,  # (ap)
                                     thres_var=args.thres_var, thres_corr=args.thres_corr,  # (ap)
                                     thres_discrete=args.thres_discrete, onehot_discrete=args.onehot_discrete,  # (ap)
                                     create_iom=args.create_iom,  # (ap)
                                     min_growth_bound=args.min_growth_bound, max_growth_bound=args.max_growth_bound)  # (ap)
        if not df.shape[0]:
            print('No response data found\n')
            continue

        if args.classify:
            good_bins = summarize(df, args.cutoffs, min_count=args.cv)
            if good_bins < 2:
                print('Not enough classes\n')
                continue
        else:
            summarize(df)

        out = os.path.join(args.out_dir, cell)
        for model in args.models:
            if args.classify:
                classify(model, df, cv=args.cv, cutoffs=args.cutoffs, threads=args.threads, prefix=out)
            else:
                regress(model, df, cv=args.cv, cutoffs=args.cutoffs, threads=args.threads, prefix=out)


if __name__ == '__main__':
    main()

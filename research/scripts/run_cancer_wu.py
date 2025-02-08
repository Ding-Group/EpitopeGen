import argparse


def test_download(root="cancer_wu"):
   download_and_preprocess(outdir=f"{root}/data")


def test_annotate(root="cancer_wu"):
    annotation_wrapper(
        data_dir=f"{root}/data",
        obs_cache=f"{root}/obs_cache/cdr3_added_scirpy.csv",
        pred_csv=f"{root}/predictions/241107_GPT2_principled_ACF_seed2023/240826_wu_formatted.csv"
    )


def test_analyze(root="cancer_wu"):
    for seed in ["2023_2024_2025"]:
        for th in [0.5]:
            analysis_wrapper(
                data_dir=f"{root}/data",
                pred_csv=f"{root}/predictions/241225_seed_models/241107_GPT2_principled_ACF_seed2023/cancer_wu/240826_wu_formatted.csv",  # dummy
                epi_db_path=f"{root}/data/tumor_associated_epitopes.csv",
                obs_cache=f"{root}/predictions/241225_seed_models/ensemble_{seed}/site_added_th{th}.csv",  # This is actually used.
                # obs_cache=f"predictions/241225_seed_models/ensemble_{seed}/PA_marked_ensembled_th{th}.csv",
                outdir=f"{root}/predictions/241225_seed_models/ensemble_{seed}/{th}",
                filter_cdr3_notna=True,
                filter_cell_types=True,
                top_k=8
            )


def test_utils(root="cancer_wu"):
    construct_epitope_db(
        TCIA_db_path=f"{root}/data/TCIA-NeoantigensData.tsv",
        IEDB_db_path=f"{root}/data/epitope_table_export_1724715082.csv",
        outdir=f"{root}/data"
    )

    analyze_match_overlap(
        'predictions/241107_GPT2_principled_ACF_seed2023/cancer_wu/PA_marked.csv',
        'predictions/241107_GPT2_principled_ACF_seed2024/cancer_wu/PA_marked.csv',
        'predictions/241107_GPT2_principled_ACF_seed2025/cancer_wu/PA_marked.csv',
    )

    for k in [1,2,3,4]:
        visualize_match_overlaps_parallel(
            files_list=[
                'predictions/241107_GPT2_principled_ACF_seed2023/cancer_wu/PA_marked.csv',
                'predictions/241107_GPT2_principled_ACF_seed2024/cancer_wu/PA_marked.csv',
                'predictions/241107_GPT2_principled_ACF_seed2025/cancer_wu/PA_marked.csv',
            ],
            outdir='./predictions/241225_seed_models',
            top_k=k
        )

    inspect_num_PA('predictions/241107_GPT2_principled_ACF_seed2023/cancer_wu/PA_marked.csv')
    inspect_num_PA('predictions/241107_GPT2_principled_ACF_seed2024/cancer_wu/PA_marked.csv')
    inspect_num_PA('predictions/241107_GPT2_principled_ACF_seed2025/cancer_wu/PA_marked.csv')

    ensemble_PA_marked(
        PA_marked_list=[
            'predictions/241107_GPT2_principled_ACF_seed2023/cancer_wu/PA_marked.csv',
            'predictions/241107_GPT2_principled_ACF_seed2024/cancer_wu/PA_marked.csv',
            'predictions/241107_GPT2_principled_ACF_seed2025/cancer_wu/PA_marked.csv'
        ],
        outdir="predictions/241225_seed_models/ensemble_2023_2024_2025",
        th=0.5,
        desc='PA',
        col='match',
        num_cols=8
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Using EpiGen to analyze scTCR-seq data')
    parser.add_argument('--download', action="store_true")
    parser.add_argument('--utils', action="store_true")
    parser.add_argument('--annotate', action="store_true")
    parser.add_argument('--analyze', action="store_true")

    args = parser.parse_args()

    if args.download:
        from research.cancer_wu.download import *
        test_download()
    elif args.utils:
        from research.cancer_wu.utils import *
        test_utils()
    elif args.annotate:
        from research.cancer_wu.annotate import *
        test_annotate()
    elif args.analyze:
        from research.cancer_wu.analyze import *
        test_analyze()

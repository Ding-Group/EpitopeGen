import argparse


def test_utils(root="covid19_su"):
    construct_epitope_db(
        IEDB_db_path=f"{root}/data/epitope_table_export_1731383953_all_corona.csv",
        outdir=f"{root}/data",
        desc="corona_epitopes.csv"
    )


def test_annotate(root="covid19_su"):
    annotation_wrapper(
        data_dir=f"{root}/data/covid19",
        pred_csv=f"{root}/predictions/pred.csv",
        epi_db_path=f"{root}/data/covid19_associated_epitopes.csv",
        obs_cache=f"{root}/gex_obs/antigen_sorted_annotated.csv",
        outdir=f"{root}/example",
        annotate_corona=False,
        match_method='levenshtein'
    )


def test_analyze(root="covid19_su"):
    for th in [0.5]:
        for ens in ["42_2020_2021"]:
            analysis_wrapper(
                data_dir=f"{root}/data/covid19",
                pred_csv=f"{root}/predictions/241220_mira_expanded/241107_GPT2_principled_ACF_seed42/covid19_su/cdr3_formatted.csv",
                gex_cache=f"{root}/gex_cache/cd8_gex.h5ad",
                # epi_db_path=f"{root}/data/241222_corona_epitopes_mira_expanded_nine_mers.csv",
                epi_db_path=f"{root}/data/241220_iedb_mira_merged_unique_sorted_by_frequency.csv",
                obs_cache=f"{root}/predictions/241220_mira_expanded/ensemble_{ens}/COVID19_marked_ensembled_th{th}.csv",
                outdir=f"{root}/predictions/241220_mira_expanded/ensemble_{ens}/th{th}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--utils', action="store_true")
    parser.add_argument('--annotate', action="store_true")
    parser.add_argument('--analyze', action="store_true")

    args = parser.parse_args()

    if args.utils:
        from covid19_su.utils import *
        test_utils()
    elif args.annotate:
        from covid19_su.annotate import *
        test_annotate()
    elif args.analyze:
        from covid19_su.analyze import *
        test_analyze()

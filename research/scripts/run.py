import argparse


def test_evaluate():
    ### Evaluating the binding affinity
    # for dataset in ['McPAS', 'IEDB', 'VDJdb', 'PIRD']:
    #     evaluator = AffinityEvaluator(
    #         pred_csvs=[
    #             f"predictions/240625_random/{dataset}/random_pred.csv",
    #             f"predictions/240625_knn/{dataset}/knn_pred.csv",
    #             f"predictions/241205_example_run/{dataset}/{dataset}.csv"
    #         ],
    #         pmhc_data="data/240606_unique_peptides/peptides_for_neg.csv",
    #         outdir=f"evaluations/241205_example_run/{dataset}",
    #         topk_values=[1, 5, 10, 20],
    #         pmhc_weight="tabr_bert_fork/output/pmhc_models_peptide_only/pmhc_model_e100.pt",
    #         tcr_weight="tabr_bert_fork/model/tcr_model.pt",
    #         model_weights=[
    #             "tabr_bert_fork/output/240612_rand_tcr_pmhc_1/tcr_pep_e58.pt",
    #             "tabr_bert_fork/output/240612_rand_tcr_pmhc_2/tcr_pep_e72.pt",
    #             "tabr_bert_fork/output/240612_rand_tcr_pmhc_3/tcr_pep_e60.pt",
    #             "tabr_bert_fork/output/240612_rand_tcr_pmhc_4/tcr_pep_e50.pt",
    #             "tabr_bert_fork/output/240612_rand_tcr_pmhc_5/tcr_pep_e50.pt",
    #         ]
    #     )
    #     evaluator.eval()

    ### Length distribution, amino acid histogram
    # outdir = "figures/3a"
    # for dataset in ['IEDB', 'VDJdb', 'PIRD', 'McPAS']:
    #     for model in ['EpiGen', 'knn', 'random']:
    #         if model == 'EpiGen':
    #             pred_csv = f"predictions/241205_example_run/{dataset}/EpiGen_{dataset}.csv"
    #         else:
    #             pred_csv = f"predictions/240625_{model}/{dataset}/{model}_pred.csv"
    #         plot_length_distribution(pred_csv=pred_csv, outdir=f"{outdir}/{model}/{dataset}", k=1)
    #         draw_amino_acid_hist(pred_csv=pred_csv, outdir=f"{outdir}/{model}/{dataset}", topk=1)

    ### Clustering plot of epitopes
    # outdir = f"figures/3e/legend/{model}"
    # for dataset in ['IEDB', 'VDJdb', 'PIRD', 'McPAS']:
    #     for model in ['EpiGen', 'knn', 'random']:
    #         for col in ['label', 'pred_0']:
    #             if col == 'label':
    #                 data = f"../../data/processed/processed_{dataset}_test.csv"
    #             else:
    #                 if model == 'EpiGen':
    #                     data = f"predictions/241205_example_run/{dataset}/EpiGen_{dataset}.csv"
    #                 else:
    #                     data = f"predictions/240625_{model}/{dataset}/{model}_pred.csv"

    #             visualize_peptides_for_TCR_groups(
    #                 gliph_convergence=f"../../data/processed/processed_{dataset}_test_gliph.csv-convergence-groups.txt",
    #                 data=data,
    #                 outdir=outdir,
    #                 col=col,
    #                 feature='blosum',
    #                 n_proc=32,
    #                 legend=True
    #             )

    ### Chemical property eval
    # outdir = "figures/3c"
    # for dataset in ['IEDB', 'VDJdb', 'PIRD', 'McPAS']:
    #     evaluator = ChemicoEvaluator(
    #         f"predictions/241205_example_run/{dataset}/EpiGen_{dataset}.csv",
    #         f"{outdir}/{dataset}"
    #     )
    #     evaluator.eval()

    #     evaluator = ChemicoEvaluator(
    #         f"predictions/240625_random/{dataset}/random_pred.csv",
    #         f"{outdir}/{dataset}"
    #     )
    #     evaluator.eval()

    #     draw_dist_Chemico(
    #         f"{outdir}/{dataset}/Chem/evaluated_data_Chem_EpiGen_{dataset}.csv",
    #         f"{outdir}/{dataset}/Chem/evaluated_data_Chem_random_pred.csv",
    #         outdir=f"{outdir}/{dataset}"
    #     )


    ### Naturalness evaluation
    # from research.epigen.antigen_category_filter import run_blastp
    # outdir = "figures_e28/3d"
    # data_csv = "../data/processed/processed_VDJdb_test.csv"
    # pred_csv = "predictions/241205_example_run/VDJdb/EpiGen_VDJdb.csv"
    # blastp = "ncbi-blast-2.15.0+/bin/blastp"
    # db_path = "ncbi-blast-2.15.0+/db/swissprot"

    # predict_random(data_csv, outdir)
    # run_blastp(pred_csv, outdir, blastp, db_path, col='pred_0')
    # run_blastp(f"{outdir}/rand_pred.csv", outdir, blastp, db_path, col='pred_0')

    # eval_naturalness(
    #     pred_csv,
    #     f"{outdir}/blastp_results/EpiGen_VDJdb.txt",
    #     outdir=outdir,
    #     rand_csv=f"{outdir}/rand_pred.csv",
    #     blastp_result_rand=f"{outdir}/blastp_results/rand_pred.txt"
    # )

    ### Diversity evaluation
    # outdir = "figures/2h"
    # for ratio in ["0.00", "0.33", "0.66"]:
    #     measure_epitope_div(
    #         model_name=f"EpiGen_LP_{ratio}",
    #         pred_csvs=[f"predictions/241113_control_pseudo_labeled_set_ratio_{ratio}/donor1/donor1_tcr_repertoire_trb_formatted.csv"],
    #         datasets=["donor1"],
    #         outdir=outdir,
    #         sample_size=12000
    #     )

    # measure_epitope_div(
    #     model_name="EpiGen",
    #     pred_csvs=["predictions/241205_example_run/donor1/donor1_tcr_repertoire_trb_formatted.csv"],
    #     datasets=["donor1"],
    #     outdir=outdir
    # )

    # plot_epitope_div(
    #     outdir=outdir,
    #     div_files=[
    #         f"{outdir}/diversity_EpiGen_LP_0.00.csv",
    #         f"{outdir}/diversity_EpiGen_LP_0.33.csv",
    #         f"{outdir}/diversity_EpiGen_LP_0.66.csv",
    #         f"{outdir}/diversity_EpiGen.csv"
    #     ],
    #     descs=[
    #         "EpiGen_LP_0.00",
    #         "EpiGen_LP_0.33",
    #         "EpiGen_LP_0.66",
    #         "EpiGen"
    #     ]
    # )

    ### Simple diversity evaluation
    # outdir = "figures/2g"
    # plot_epitope_div_simple(
    #     outdir=outdir,
    #     div_files=[
    #         f"{outdir}/diversity_EpiGen.csv",
    #         f"{outdir}/diversity_EpiGenMHC.csv"
    #     ],
    #     descs=[
    #         "EpiGen",
    #         "EpiGenMHC"
    #     ]
    # )

    ### Generation redundancy evaluation
    # outdir = "figures/edf3"
    # measure_generation_redundancy(
    #     pred_csvs=[
    #         'predictions/241025_control_pseudo_labeled_set_ratio_0.00/donor1/donor1_tcr_repertoire_trb_formatted.csv',  # 0%
    #         'predictions/241113_control_pseudo_labeled_set_ratio_0.33/donor1/donor1_tcr_repertoire_trb_formatted.csv',  # 33%
    #         'predictions/241113_control_pseudo_labeled_set_ratio_0.66/donor1/donor1_tcr_repertoire_trb_formatted.csv',  # 66%
    #         'predictions/241205_example_run/donor1/donor1_tcr_repertoire_trb_formatted.csv'  # EpiGen
    #     ],
    #     descs=[
    #         "EpiGen_LP_0.00",
    #         "EpiGen_LP_0.33",
    #         "EpiGen_LP_0.66",
    #         "EpiGen"
    #     ],
    #     outdir=outdir
    # )


def test_utils():
    ### Postprocess the sampled data after label propagation
    # postprocess_sampled_data(root="data_v6", keyword="affinity_tables", outdir="data/240930_topk8_400", topk=8, use_mhc=False)
    # for th in [400]:
    #     remove_redundancy("data/240930_topk8_400/all_data_topk8.csv", th=th)

    # format_sampled_data(
    #     sampled_data_csv="240620_sampled_data/all_data_topk32_th_100.csv",
    #     outfile="data/240620_all_data_topk32_th_100.csv", use_mhc=False)

    ### Split the data into train,val,test and merge each again
    # for dataset in ['IEDB', 'VDJdb', 'PIRD', 'McPAS']:
    #     split_data(file_path=f"../data/processed/processed_{dataset}.csv", random_seed=None)

    # for split in ['train', 'val', 'test']:
    #     merge_data(files=[
    #         f'../data/processed/processed_IEDB_{split}.csv',
    #         f'../data/processed/processed_VDJdb_{split}.csv',
    #         f'../data/processed/processed_PIRD_{split}.csv',
    #         f'../data/processed/processed_McPAS_{split}.csv',
    #     ], split=split, outdir='data/240605_public_datasets')

    ### Rosetta evaluation
    # for dataset in ['IEDB', 'VDJdb', 'PIRD', 'McPAS']:
    #     convert_pred_to_tcrmodel2_format(
    #         pred_csv=f"predictions/241205_example_run/{dataset}/EpiGen_{dataset}.csv",
    #         peptide_db="data/240606_unique_peptides/candidate_peptides.csv",
    #         pseudo2full_pkl="data/pseudo2full.pkl",
    #         tcr_alpha_template="AQEVTQIPAALSVPEGENLVLNCSFTDSAIYNLQWFRQDPGKGLTSLLLIQSSQREQTSGRLNASLDKSSGRSTLYIAASQPGDSATYLCAVTNQAGTALIFGKGTTLSVSS",
    #         tcr_beta_template="NAGVTQTPKFQVLKTGQSMTLQCSQDMNHEYMSWYRQDPGMGLRLIHYSVGAGITDQGEVPNGYNVSRSTTEDFPLRLLSAAPSQTSVYFCASSYSIRGSRGEQFFGPGTRLTVL",
    #         pseudo="YFAMYGEKVAHTHVDTLYGVRYDHYYTWAVLAYTWYA",
    #         epitope_col='epitope'  # use pred_0 for prediction
    #     )

    ### Reason why need this function
    # This function enables incremental evaluation of new epitope predictions by leveraging existing
    # ground truth measurements. Rather than re-running the computationally expensive MD simulations
    # for all ground truth epitopes (which serve as controls), this function only processes new
    # epitopes from a specific model. This significantly reduces computation time when evaluating
    # new models or predictions against the same ground truth baseline.

    # convert_pred_to_tcrmodel2_format_continual(
    #     pred_csv=f"predictions/241205_example_run/VDJdb/EpiGen_VDJdb.csv",
    #     tcrmodel2_arg_ref="predictions/241205_example_run/VDJdb/tcrmodel2_args_VDJdb_epitope.pkl",
    #     desc='example_run',
    #     epitope_col='pred_0'
    # )

    print("end of program. ")


def test_select_tcr_pep_model():
    """
    We developed Robust Affinity Predictor based upon TABR-BERT
    https://github.com/Freshwind-Bioinformatics/TABR-BERT

    We modified the head architecture, loss function, and removed MHC related codes.
    1. We need to train a BERT model with MHC removed, using tabr_bert_fork/pre_train_peptide_embedding_model.py
    2. Then, we need to train five independent tcr-peptide binding affinity predictors
        using tabr_bert_fork/train_tcr_peptide_prediction_model.py
    3. Use epigen/select_best_model_tabr_bert to select the best checkpoints.
    """
    root = "tabr_bert_fork"
    select_best_model_tabr_bert(
        tcr_ckpt=f"{root}/model/tcr_model.pt",
        pep_ckpt=f"{root}/output/pmhc_models_peptide_only/pmhc_model_e100.pt",
        tcr_pep_ckpts_root=f"{root}/output/240612_rand_tcr_pmhc_5",
        test_data="data/240605_public_datasets/val_neg_test_with_neg_multi_1.csv",
        outdir="data/240606_critique_eval",
        desc="rand_tcr_pmhc_5",
        batch_size=512,
        device='cuda'
    )


def test_featurize():
    ### Featurize the epitopes to get the epitope_pool
    # featurizer = EpitopeFeaturizer(
    #     epitope_data="data/240606_unique_peptides/candidate_peptides.csv",
    #     model_path="tabr_bert_fork/output/pmhc_models_peptide_only/pmhc_model_e100.pt",
    #     pseudo_sequence_file="tabr_bert_fork/data/mhcflurry.allele_sequences_homo.csv"
    # )
    # featurizer.featurize_epitopes()

    # featurizer = VDJdbEpitopeFeaturizer(
    #     epitope_data="",  # VDJdb standard csv file path
    #     model_path="tabr_bert_fork/model/pmhc_model.pt",
    #     pseudo_sequence_file="tabr_bert_fork/data/mhcflurry.allele_sequences_homo.csv"
    # )
    # featurizer.featurize_epitopes(outfile='epi_features.pkl')

    ### Featurize the tcrs
    # featurizer = TCRFeaturizer(
    #     tcr_data="",  # VDJdb standard csv file path
    #     model_path="tabr_bert_fork/model/tcr_model.pt"
    # )
    # featurizer.featurize_tcrs()

    # featurizer = TCRDBFeaturizer(
    #     # tcr_data="../tcr-bert/data/tcrdb/",  # TCRDB
    #     tcr_data="data/240612_tcrdb/tcrs_for_candidate.csv",
    #     model_path="tabr_bert_fork/model/tcr_model.pt",
    #     outdir="data/tcr_features"
    # )
    # featurizer.featurize_tcrs()


def test_label_prop():
    ### Perform label propagation. Parallelize the jobs over pkl files.
    # sampler = TCRPepSampler(
    #     tcr_feat_pkl="data/tcr_features/tcr_features_22.pkl",
    #     pep_feat_root="../../EpiGen/pmhc_features",
    #     model_paths=[
    #         "tabr_bert_fork/output/240612_rand_tcr_pmhc_1/tcr_pep_e58.pt",
    #         "tabr_bert_fork/output/240612_rand_tcr_pmhc_2/tcr_pep_e72.pt",
    #         "tabr_bert_fork/output/240612_rand_tcr_pmhc_3/tcr_pep_e60.pt",
    #         "tabr_bert_fork/output/240612_rand_tcr_pmhc_4/tcr_pep_e50.pt",
    #         "tabr_bert_fork/output/240612_rand_tcr_pmhc_5/tcr_pep_e50.pt",
    #     ],
    #     outdir="affinity_tables",
    #     tcr_chunk=4096,
    #     batch_size=16
    # )
    # sampler.sample()

    ### Binding affinity value distribution
    # outdir = "figures/1c"
    # check_sampled_data_sanity(
    #     outdir=outdir,
    #     pkl_path="affinity_tables_tcr_data_0_20240617_221652/sampled_data_28672.pkl",
    #     tcr_model_path="tabr_bert_fork/model/tcr_model.pt",
    #     pep_model_path="tabr_bert_fork/output/pmhc_models_peptide_only/pmhc_model_e100.pt",
    #     model_paths=[
    #         "tabr_bert_fork/output/240612_rand_tcr_pmhc_1/tcr_pep_e58.pt",
    #         "tabr_bert_fork/output/240612_rand_tcr_pmhc_2/tcr_pep_e72.pt",
    #         "tabr_bert_fork/output/240612_rand_tcr_pmhc_3/tcr_pep_e60.pt",
    #         "tabr_bert_fork/output/240612_rand_tcr_pmhc_4/tcr_pep_e50.pt",
    #         "tabr_bert_fork/output/240612_rand_tcr_pmhc_5/tcr_pep_e50.pt",
    #     ],
    #     model_mode='softmax', pmhc_maxlen=18, device='cuda',
    #     pep_data="data/240606_unique_peptides/candidate_peptides.csv"
    # )


def test_baselines():
    # for dataset in ['IEDB', 'VDJdb', 'PIRD', 'McPAS', 'Glanville']:
    #     knn = KNNSequenceGenerator(
    #         train_csv="data/240605_public_datasets/train.csv",
    #         k=50,
    #         outdir=f"predictions/240625_knn/{dataset}",
    #         use_mhc=False
    #     )
    #     knn.predict_all(f"../data/processed/processed_{dataset}_test.csv", n_proc=10)

    # for dataset in ['IEDB', 'VDJdb', 'PIRD', 'McPAS', 'Glanville']:
    #     rand = RandomGenerator(
    #         train_csv="data/240605_public_datasets/train.csv",
    #         k=50, outdir=f"predictions/240625_rand/{dataset}"
    #     )
    #     rand.predict_all(f"../data/processed/processed_{dataset}_test.csv")


def test_antigen_category_filter():
    split = "val"
    root = f"predictions/241106_all_data_topk32_th_100_{split}/tables"
    desc = "val"

    ### Before running the following, run run_blastp() and create_EpiGen_table() on the predicted csv file
    ### Assume the EpiGen table was created under `root`

    ### Parallelization (optional)
    # merge_and_partition_epigen_tables(
    #     outdir="data/partitions_tables",
    #     epigen_table_train="predictions/241106_all_data_topk32_th_100_train/tables/240620_all_data_topk32_th_100_train.csv",
    #     epigen_table_val="predictions/241106_all_data_topk32_th_100_val/tables/240620_all_data_topk32_th_100_val.csv",
    #     epigen_table_test="predictions/241106_all_data_topk32_th_100_test/tables/240620_all_data_topk32_th_100_test.csv",
    #     n_part=10
    # )

    # identify_tumor_antigens(
    #     epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    #     epi_db_path="cancer_wu/data/tumor_associated_epitopes.csv",
    #     outdir=f"{root}/tumor_marked",
    #     col="pred_0",
    #     threshold=0,
    #     method='substring',
    #     debug=None
    # )

    # retrieve_rows_tumor_associated(
    #     epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    #     parted_tumor_antigen_annotation_root=f"{root}/tumor_marked",
    #     col="pred_0"
    # )

    # retrieve_rows_self_antigens(
    #     epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    #     tumor_csv=f"{root}/partitions/tumor.csv",
    #     col="pred_0"
    # )

    # accessions_list_from_table(
    #     epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    #     tumor_csv=f"{root}/partitions/tumor.csv",
    #     self_csv=f"{root}/partitions/self.csv",
    #     col='pred_0',
    #     desc=desc
    # )

    # accession2taxid(
    #     accession_list=f"{root}/accessions_list_{desc}.txt",
    #     desc=desc,
    #     chunk_size=20000
    # )

    # run_efetch_parallel(
    #     tax_ids_file=f"{root}/result_{desc}.txt",
    #     output_dir=f"{root}/efetch",
    #     chunk_size=200
    # )

    # ### 2. Parse the lineage information
    # efetch_dir = f"{root}/efetch"
    # result = []
    # for file in tqdm(os.listdir(efetch_dir)):
    #     category = parse_efetch_result(xml_file=f"{efetch_dir}/{file}")
    #     tax_id = file[:-4].split("_")[1]
    #     result.append((tax_id, category))
    # df = pd.DataFrame(result, columns=['tax_id', 'category'])
    # df.to_csv(f"{root}/tax_id2category_{desc}.csv", index=False)

    # make_species2category(
    #     accessions_list=f"{root}/accessions_list_{desc}.txt",
    #     accession2tax_id_result=f"{root}/result_{desc}.txt",
    #     tax_id2category=f"{root}/tax_id2category_{desc}.csv",
    #     epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    #     outdir=root
    # )

    # add_category_annotation(
    #     epigen_table=f"{root}/donor1_tcr_repertoire_trb_formatted.csv",
    #     species2category=f"{root}/species2category.csv",
    #     tumor_csv=f"{root}/partitions/tumor.csv",
    #     self_csv=f"{root}/partitions/self.csv",
    #     col='pred_0'
    # )

    # for split in ["train", "val", "test"]:
    #     construct_balanced_data(
    #         outdir=f"predictions/241106_all_data_topk32_th_100_{split}/tables",
    #         annotated_table=f"{root}/240620_all_data_topk32_th_100_{split}_cat_annotated.csv",
    #         seed=42  # random seed for reproducibility
    #     )

    # visualize_antigen_category_dist(
    #     epigen_table_annotated=f"{root}/donor1_tcr_repertoire_trb_formatted_cat_annotated.csv",
    #     outdir=root
    # )


def test_tokenizer():
    # outdir = "regaler/EpiGen"
    # vocab_sizes = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800]
    # seq_for_tok = "predictions/241106_all_data_topk32_th_100_train/tables/seq_for_tok.txt"

    # make_seq_for_tok(
    #     train_csv="predictions/241106_all_data_topk32_th_100_train/tables/240620_all_data_topk32_th_100_train.csv",
    #     n_tcr=2500000, n_epi=2500000)

    # for vocab_size in vocab_sizes:
    #     train_bpe_tokenizer(vocab_size, seq_for_tok, outdir)

    # Read data for testing
    # df = pd.read_csv(seq_for_tok, header=None)
    # sequences = df[0].tolist()[:10000] + df[0].tolist()[-10000:]

    # # Evaluate tokenizers and generate plots
    # results_df = evaluate_tokenizers(vocab_sizes, outdir, sequences)
    # results_df.to_csv(os.path.join(outdir, "tokenizer_evaluation_results.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--preprocess', action="store_true")
    parser.add_argument('--utils', action="store_true")
    parser.add_argument('--select_tcr_pep_model', action="store_true")
    parser.add_argument('--featurize', action="store_true")
    parser.add_argument('--label_prop', action="store_true")
    parser.add_argument('--tokenizer', action="store_true")
    parser.add_argument('--baselines', action="store_true")
    parser.add_argument('--evaluate', action="store_true")
    parser.add_argument('--antigen_category_filter', action="store_true")

    args = parser.parse_args()

    if args.preprocess:
        from research.epigen.preprocess import *
        test_preprocess()
    elif args.utils:
        from research.epigen.utils import *
        test_utils()
    elif args.select_tcr_pep_model:
        from tabr_bert_fork.select_tcr_pep_model import *
        test_select_tcr_pep_model()
    elif args.featurize:
        from research.epigen.featurize import *
        test_featurize()
    elif args.label_prop:
        from research.epigen.label_prop import *
        test_label_prop()
    elif args.tokenizer:
        from research.epigen.tokenizer import *
        test_tokenizer()
    elif args.baselines:
        from research.epigen.eval.baselines import *
        test_baselines()
    elif args.evaluate:
        from research.epigen.eval.evaluate import *
        test_evaluate()
    elif args.antigen_category_filter:
        from research.epigen.antigen_category_filter import *
        test_antigen_category_filter()

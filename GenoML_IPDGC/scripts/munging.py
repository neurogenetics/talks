parser = argparse.ArgumentParser(description='Arguments for building a training dataset for GenoML.')    
parser.add_argument('--prefix', type=str, default='GenoML_data', help='Prefix for your training data build.')
parser.add_argument('--geno', type=str, default='nope', help='Genotype: (string file path). Path to PLINK format genotype file, everything before the *.bed/bim/fam [default: nope].')
parser.add_argument('--addit', type=str, default='nope', help='Additional: (string file path). Path to CSV format feature file [default: nope].')
parser.add_argument('--pheno', type=str, default='lost', help='Phenotype: (string file path). Path to CSV phenotype file [default: lost].')
parser.add_argument('--gwas', type=str, default='nope', help='GWAS summary stats: (string file path). Path to CSV format external GWAS summary statistics containing at least the columns SNP and P in the header [default: nope].')
parser.add_argument('--p', type=float, default=0.001, help='P threshold for GWAS: (some value between 0-1). P value to filter your SNP data on [default: 0.001].')
parser.add_argument('--vif', type=int, default=0, help='Variance Inflation Factor (VIF): (integer). This is the VIF threshold for pruning non-genotype features. We recommend a value of 5-10. The default of 0 means no VIF filtering will be done. [default: 0].')
parser.add_argument('--iter', type=int, default=0, help='Iterator: (integer). How many iterations of VIF pruning of features do you want to run. To save time VIF is run in randomly assorted chunks of 1000 features per iteration. The default of 0 means only one pass through the data. [default: 0].')
parser.add_argument('--impute', type=str, default='median', help='Imputation: (mean, median). Governs secondary imputation and data transformation [default: median].')

args = parser.parse_args()

print("")

print("Here is some basic info on the command you are about to run.")
print("Python version info...")
print(sys.version)
print("CLI argument info...")
print("The output prefix for this run is", args.prefix, "and will be appended to later runs of GenoML.")
print("Working with genotype data?", args.geno)
print("Working with additional predictors?", args.addit)
print("Where is your phenotype file?", args.pheno)
print("Any use for an external set of GWAS summary stats?", args.gwas)
print("If you plan on using external GWAs summary stats for SNP filtering, we'll only keep SNPs at what P value?", args.p)
print("How strong is your VIF filter?", args.vif)
print("How many iterations of VIF filtering are you doing?", args.iter)
print("The imputation method you picked is using the column", args.impute, "to fill in any remaining NAs.")
print("Give credit where credit is due, for this stage of analysis we use code from the great contributors to python packages: os, sys, argparse, numpy, pandas, joblib, math and time. We also use PLINKv1.9 from https://www.cog-genomics.org/plink/1.9/.")

run_prefix = args.prefix

print("")

pheno_path = args.pheno
if (pheno_path == "lost"):
  print("Looks like you lost your phenotype file. Just give up because you are currently don't have anything to predict.")
if (pheno_path != "lost"):
  pheno_df = pd.read_csv(pheno_path, engine = 'c')

addit_path = args.addit
if (addit_path == "nope"):
  print("No additional features as predictors? No problem, we'll stick to genotypes.")
if (addit_path != "nope"):
  addit_df = pd.read_csv(addit_path, engine = 'c')

gwas_path = args.gwas
if (gwas_path == "nope"):
  print("So you don't want to filter on P values from external GWAS? No worries, we don't usually either (if the dataset is large enough).")
if (gwas_path != "nope"):
  gwas_df = pd.read_csv(gwas_path, engine = 'c')

geno_path = args.geno
if (geno_path == "nope"):
  print("So no genotypes? Okay, we'll just use additional features provided for the predictions.")
if (geno_path != "nope"):
  print("Pruning your data and exporting a reduced set of genotypes.")


# Set the bashes
bash1a = "plink --bfile " + geno_path + " --indep-pairwise 1000 50 0.05"
bash1b = "plink --bfile " + geno_path + " --extract " + run_prefix + ".p_threshold_variants.tab" + " --indep-pairwise 1000 50 0.05"
bash2 = "plink --bfile " + geno_path + " --extract plink.prune.in --make-bed --out temp_genos"
bash3 = "plink --bfile temp_genos --recodeA --out " + run_prefix
bash4 = "cut -f 2,5 temp_genos.bim > " + run_prefix + ".variants_and_alleles.tab"
bash5 = "rm temp_genos.*"
bash6 = "rm " + run_prefix + ".raw"
bash7 = "rm plink.log"
bash8 = "rm plink.prune.*"
bash9 = "rm " + run_prefix + ".log"

# Set the bash command groups
cmds_a = [bash1a, bash2, bash3, bash4, bash5, bash7, bash8, bash9]
cmds_b = [bash1b, bash2, bash3, bash4, bash5, bash7, bash8, bash9]


if (gwas_path != "nope") & (geno_path != "nope"):
  p_thresh = args.p
  gwas_df_reduced = gwas_df[['SNP','p']]
  snps_to_keep = gwas_df_reduced.loc[(gwas_df_reduced['p'] <= p_thresh)]
  outfile = run_prefix + ".p_threshold_variants.tab"
  snps_to_keep.to_csv(outfile, index=False, sep = "\t")
  print("Your candidate variant list prior to pruning is right here", outfile, ".")

if (gwas_path == "nope") & (geno_path != "nope"):
  print("A list of pruned variants and the allele being counted in the dosages (usually the minor allele) can be found here ", run_prefix + ".variants_and_alleles.tab.")
  for cmd in cmds_a:
      subprocess.run(cmd, shell=True)

if (gwas_path != "nope") & (geno_path != "nope"):
  print("A list of pruned variants and the allele being counted in the dosages (usually the minor allele) can be found here", run_prefix + ".variants_and_alleles.tab.")
  for cmd in cmds_b:
      subprocess.run(cmd, shell=True)

if (geno_path != "nope"):
  raw_path = run_prefix + ".raw"
  raw_df = pd.read_csv(raw_path, engine = 'c', sep = " ")
  raw_df.drop(columns=['FID','MAT','PAT','SEX','PHENOTYPE'], inplace=True)
  raw_df.rename(columns={'IID':'ID'}, inplace=True)
  subprocess.run(bash6, shell=True)

impute_type = args.impute

if (geno_path != "nope"):
  if impute_type == 'mean': 
    raw_df = raw_df.fillna(raw_df.mean())
  if impute_type == 'median':
    raw_df = raw_df.fillna(raw_df.median())
  print("")
  print("You have just imputed your genotype features, covering up NAs with the column", impute_type, "so that analyses don't crash due to missing data.")
  print("Now your genotype features might look a little better (showing the first few lines of the left-most and right-most columns)...")
  print("#"*30)
  print(raw_df.describe())
  print("#"*30)
  print("")

if (addit_path != "nope"):
  if impute_type == 'mean': 
    addit_df = addit_df.fillna(addit_df.mean())
  if impute_type == 'median':
    addit_df = addit_df.fillna(addit_df.median())
  print("")
  print("You have just imputed your non-genotype features, covering up NAs with the column", impute_type, "so that analyses don't crash due to missing data.")
  print("Now your non-genotype features might look a little better (showing the first few lines of the left-most and right-most columns)...")
  print("#"*30)
  print(addit_df.describe())
  print("#"*30)
  print("")


if (addit_path != "nope"):

  cols = list(addit_df.columns)
  cols.remove('ID')
  addit_df[cols]

  for col in cols:
    if (addit_df[col].min() != 0) & (addit_df[col].max() != 1):
      addit_df[col] = (addit_df[col] - addit_df[col].mean())/addit_df[col].std(ddof=0)

  print("")
  print("You have just Z-scaled your non-genotype features, putting everything on a numeric scale similar to genotypes.")
  print("Now your non-genotype features might look a little closer to zero (showing the first few lines of the left-most and right-most columns)...")
  print("#"*30)
  print(addit_df.describe())
  print("#"*30)
  print("")


outfile_h5 = run_prefix + ".dataForML.h5"

pheno_df.to_hdf(outfile_h5, key='pheno', mode = 'w')

if (geno_path != "nope"):
  raw_df.to_hdf(outfile_h5, key='geno')

if (addit_path != "nope"):
  addit_df.to_hdf(outfile_h5, key='addit')


if (geno_path != "nope") & (addit_path != "nope"):
  pheno = pd.read_hdf(outfile_h5, key = "pheno")
  geno = pd.read_hdf(outfile_h5, key = "geno")
  addit = pd.read_hdf(outfile_h5, key = "addit")
  temp = pd.merge(pheno, addit, on='ID', how='inner')
  merged = pd.merge(temp, geno, on='ID', how='inner')
  
if (geno_path != "nope") & (addit_path == "nope"):
  pheno = pd.read_hdf(outfile_h5, key = "pheno")
  geno = pd.read_hdf(outfile_h5, key = "geno")
  merged = pd.merge(pheno, geno, on='ID', how='inner')

if (geno_path == "nope") & (addit_path != "nope"):
  pheno = pd.read_hdf(outfile_h5, key = "pheno")
  addit = pd.read_hdf(outfile_h5, key = "addit")
  merged = pd.merge(pheno, addit, on='ID', how='inner')

merged.to_hdf(outfile_h5, key='dataForML')


print()
print("Thanks for munging some data with GenoML!")
print()
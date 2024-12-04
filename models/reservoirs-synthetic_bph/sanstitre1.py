import SF_FE.study_config
import SF_ME.study_config
import AF_FE.study_config
import AF_ME.study_config

from utils.errors_computing import compute_errors_df


df = compute_errors_df(SF_FE.study_config)

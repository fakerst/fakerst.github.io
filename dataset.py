import os
import glob
import re
import numpy as np
import pandas as pd 
import logging 
from joblib import Memory
import hdbscan
from sklearn.preprocessing import StandardScaler


memory = Memory(os.path.join(os.path.dirname(__file__), ".cache"))


def load(paths):
    """
    Load the CSV file at the provided path, and return a pandas DataFrame.
    """
    if not isinstance(paths, list): 
        paths = [paths]

    df = pd.DataFrame()
    for path in paths: 
        new_df = pd.read_csv(path, delimiter='|')
        df = pd.concat([df, new_df])

    df = df.reset_index()

    return df


def get_number_columns(df):
    """
    Since some columns contain string metadata, and others contain values,
    this function returns the columns that contain values.
    """
    return df.columns[np.logical_or(df.dtypes == np.float64, df.dtypes == np.int64)]


def remove_unimporant_features(df, keep_timestamps=True):
    """
    Remove features Dr. Carns marked as unimportant.
    """
    unimportant_labels = [
        "STDIO_SEEKS", "STDIO_FASTEST_RANK", "STDIO_FASTEST_RANK_BYTES", "STDIO_SLOWEST_RANK",
        "STDIO_SLOWEST_RANK_BYTES", "STDIO_F_OPEN_START_TIMESTAMP", "STDIO_F_CLOSE_START_TIMESTAMP", 
        "STDIO_F_WRITE_START_TIMESTAMP", "STDIO_F_READ_START_TIMESTAMP", "STDIO_F_OPEN_END_TIMESTAMP",
        "STDIO_F_CLOSE_END_TIMESTAMP", "STDIO_F_WRITE_END_TIMESTAMP", "STDIO_F_READ_END_TIMESTAMP", 
        "STDIO_F_FASTEST_RANK_TIME", "STDIO_F_SLOWEST_RANK_TIME", "STDIO_F_VARIANCE_RANK_TIME",
        "STDIO_F_VARIANCE_RANK_BYTES", "MPIIO_FASTEST_RANK", "MPIIO_FASTEST_RANK_BYTES",
        "MPIIO_F_CLOSE_END_TIMESTAMP", "MPIIO_F_FASTEST_RANK_TIME", "MPIIO_F_MAX_READ_TIME",
        "MPIIO_F_MAX_WRITE_TIME", "MPIIO_F_OPEN_START_TIMESTAMP", "MPIIO_F_OPEN_END_TIMESTAMP",  
        "MPIIO_F_READ_END_TIMESTAMP", "MPIIO_F_CLOSE_START_TIMESTAMP",
        "MPIIO_F_READ_START_TIMESTAMP", "MPIIO_F_SLOWEST_RANK_TIME", "MPIIO_F_VARIANCE_RANK_BYTES",
        "MPIIO_F_VARIANCE_RANK_TIME", "MPIIO_F_WRITE_END_TIMESTAMP", "MPIIO_F_WRITE_START_TIMESTAMP",
        "MPIIO_HINTS", "MPIIO_MAX_READ_TIME_SIZE", "MPIIO_MAX_WRITE_TIME_SIZE", "MPIIO_MODE",
    ]

    metadata_labels = [
        "darshan log version", "compression method", "uid", "jobid", "start_time",
        "start_time_asci", "end_time", "end_time_asci", "metadata",
    ]

    drop_labels = unimportant_labels + metadata_labels

    if not keep_timestamps: 
        drop_labels = [label for label in drop_labels if "TIMESTAMP" not in label]

    for label in drop_labels:
        try: 
            df = df.drop(columns=label)
        except:  # noqa: E722
            logging.warning("Cannot drop nonexistant column {}".format(label))

    return df


def replace_timestamps(df): 
    """
    Replace timestamps with appropriate intervals.
    """
    for label in df.columns:
        if "START_TIMESTAMP" in label: 
            if label.replace("START_TIMESTAMP", "END_TIMESTAMP") in df.columns:
                start_label = label
                end_label   = label.replace("START_TIMESTAMP", "END_TIMESTAMP")
                delta_label = label.replace("START_TIMESTAMP", "DELTA")

                df[delta_label] = df[end_label] - df[start_label]
                df = df.drop(columns=[start_label, end_label])

                # Log how many jobs had negative end timestamps or start timestamps > end timestamps
                if np.sum(df[delta_label] < 0) > 0:
                    logging.info("Column {} had {} negative delta periods".format(delta_label, np.sum(df[delta_label] < 0)))
            else:
                logging.error("Found column {} but could not find matching column {}".format(label, label.replace("START_TIMESTAMP", "END_TIMESTAMP")))

    # Check if we didn't remove any END_TIMESTAMP columns
    for label in df.columns: 
        if "END_TIMESTAMP" in label: 
            logging.error("Found column {} that did not have a maching start column".format(label))

    return df


def remove_NaN_features(df):
    """
    Removes features don't have values at all.
    """
    for column in get_number_columns(df): 
        if np.all(np.isnan(df[column])):
            df = df.drop(columns=column)
            logging.info("Removing NaN feature {}".format(column))

    return df


def remove_NaN_jobs(df):
    """
    Removes any rows that have NaN values.
    """
    bad_rows = pd.isnull(df).any(axis=1)
    logging.info("Removing {} jobs that have NaN values".format(np.sum(bad_rows)))
    return df.loc[~bad_rows] 


def remove_subzero_features_and_jobs(df, min_zeros_to_drop=10000):
    """
    Remove columns with too many sub-zero values and jobs with negative values.
    """ 
    # First, drop bad columns
    drop_columns = []
    for idx, c in enumerate(df.columns):
        if df.dtypes[idx] == np.int64 or df.dtypes[idx] == np.float64:
            subzeros = np.sum(df[c] < 0)

            if subzeros > 0:
                logging.info("{} jobs had a negative value in column {}".format(subzeros, c))

            if subzeros > min_zeros_to_drop:
                drop_columns.append(c)
                logging.info("Dropping column {}".format(c))

    df = df.drop(columns=drop_columns)

    # Next, drop jobs that have negative values
    jobs_without_zeros = np.sum(df[get_number_columns(df)] < 0, axis=1) == 0
    pd.options.display.max_rows = 999
    logging.info("Number of zero values / feature")
    logging.info(np.sum(df[get_number_columns(df)] < 0)[np.sum(df[get_number_columns(df)] < 0) > 0])
    logging.info("Removing {} jobs".format(np.sum(~jobs_without_zeros)))
    df = df.loc[jobs_without_zeros]

    return df


def remove_jobs_missing_modules(df, modules):
    """
    Removes jobs that don't have a specified module such as POSIX, MPIIO, or STDIO.
    """
    columns = [] 
    for module in modules:
        columns += [x for x in df.columns if module in str(x)]

    job_vector = np.all(~np.isnan(df[columns]), axis=1)

    logging.info("Removing {} jobs that are missing the {} modules".format(np.sum(~job_vector), modules))

    return df[job_vector]


def remove_zero_variance_features(df): 
    """
    Drop columns with zero variance.
    """
    drop_columns = get_number_columns(df)[df[get_number_columns(df)].var() == 0]
    df = df.drop(columns=drop_columns)

    logging.info("Dropping zero-variance columns: {}".format(drop_columns))

    return df


def remove_correlated_features(df, min_correlation=0.99):
    """
    Clusters together highly correlated features into sets. 
    Since some features are more important than others, forces keeping those features out of a set of
    correlated ones.
    """
    keep_features = ['runtime', "POSIX_total_bytes", "POSIX_READS"]
    corr = df.corr()
    feature_sets = []

    # First, populate the feature_sets list with sets of correlated features
    for row, col in zip(*np.where(corr > min_correlation)):
        # check if we should bundle in a previous set 
        found_set = False

        for fs in feature_sets: 
            if row in fs or col in fs:
                fs.add(row)
                fs.add(col)
                found_set = True
                break

        if not found_set:
            feature_sets.append(set((row, col)))

    # Log the correlated sets 
    for feature_set in [s for s in feature_sets if len(s) > 1]:
        logging.info("Found a set of correlated features: {}".format(list(df.columns[list(feature_set)])))

    # Next, figure out what features per set to keep, and what to discard
    drop_columns = []
    for fs in [fs for fs in feature_sets if len(fs) > 1]: 
        column_set = set(df.columns[list(fs)])

        intersection = column_set.intersection(keep_features)
        if len(intersection) > 1: 
            logging.warning("Found a set of correlated features that contains multiple features we are forced to keep: {}".format(intersection))

        # In case the intersection has some elements, just remove keep_features
        if len(intersection) > 0:
            column_set = column_set.difference(keep_features)
        # Else, keep the first one 
        else:
            column_set = list(column_set)[1:]

        # Remove any features we are forced to keep
        drop_columns += list(column_set)
        
    # Finally, drop the correlated features, without the string features
    drop_columns = list(set(drop_columns).intersection(get_number_columns(df)))
    logging.info("Removing correlated set of features: {}".format(drop_columns))
    df = df.drop(columns=drop_columns)

    return df


def keep_columns_containing(df, text):
    """
    Remove all columns that do not contain the text argument within their name. 
    """
    if not isinstance(text, list):
        text = [text] 

    columns = set() 
    for c in df.columns: 
        for t in text: 
            if t in c:
                columns.add(c)

    logging.info("Removing columns that do not contain {}: {}".format(text, set(df.columns).difference(set(columns))))

    return df[columns] 


def remove_columns_containing(df, text):
    """
    Remove all columns that do contain the text argument within their name. 
    """
    if not isinstance(text, list):
        text = [text] 

    drop_columns = set() 
    for c in df.columns: 
        for t in text: 
            if t in c:
                drop_columns.add(c)

    logging.info("Removing columns that contain {}: {}".format(text, drop_columns))

    return df.drop(columns=list(drop_columns))


def extract_users(df):
    """
    From the filenames, extracts users and adds a user column.
    """
    df['users'] = [re.match(r"([a-zA-Z0-9\+]*)_([a-zA-Z0-9_\-.\+]+)_id.*", re.findall(r"[a-zA-Z0-9_.\+-]+.darshan", p)[0], re.MULTILINE).groups()[0] for p in df.filename]
    return df


def extract_apps(df):
    """
    From the filenames, extracts applications and adds an application column.
    """
    df['apps']       = [re.match(r"([a-zA-Z0-9\+]*)_([a-zA-Z0-9_\-.\+]+)_id.*", re.findall(r"[a-zA-Z0-9_.\+-]+.darshan", p)[0], re.MULTILINE).groups()[1] for p in df.filename]
    df['apps_short'] = [re.match(r"([a-zA-Z0-9]+).*", x) for x in df.apps]
    df['apps_short'] = [x.groups(1)[0] if x is not None else "" for x in df.apps_short]
    return df


def convert_POSIX_features_to_percentages(df, remove_dual=True):
    """
    Certain features like POSIX_SEQ_READS make more sense when normalized by a more general feature such as POSIX_READS
    For all features that measure either the number of a certain type of access, or the number of bytes, we normalize by
    the total number POSIX accesses and total number of POSIX bytes accessed.
    If remove_dual is true, removes one of the dual features such read and write percentage, unique and shared, etc.
    """
    df = df.copy()

    if np.any(np.isnan(df[get_number_columns(df)])):
        logging.error("Found NaN values before normalizing dataframe.") 
    
    total_accesses = df.POSIX_WRITES + df.POSIX_READS
    total_bytes    = df.POSIX_total_bytes
    total_files    = df.POSIX_shared_files + df.POSIX_unique_files

    df['POSIX_total_accesses'] = total_accesses
    df['POSIX_total_files']    = total_files

    try:
        df['POSIX_BYTES_READ_PERC'      ] = df.POSIX_BYTES_READ       / total_bytes  # noqa: E202
        df['POSIX_BYTES_WRITTEN_PERC'   ] = df.POSIX_BYTES_WRITTEN    / total_bytes  # noqa: E202
        df['POSIX_unique_bytes_perc'    ] = df.POSIX_unique_bytes     / total_bytes  # noqa: E202
        df['POSIX_shared_bytes_perc'    ] = df.POSIX_shared_bytes     / total_bytes  # noqa: E202
        df['POSIX_read_only_bytes_perc' ] = df.POSIX_read_only_bytes  / total_bytes  # noqa: E202
        df['POSIX_read_write_bytes_perc'] = df.POSIX_read_write_bytes / total_bytes  # noqa: E202
        df['POSIX_write_only_bytes_perc'] = df.POSIX_write_only_bytes / total_bytes  # noqa: E202
        df = df.drop(columns=["POSIX_BYTES_READ",      "POSIX_BYTES_WRITTEN",    "POSIX_unique_bytes", "POSIX_shared_bytes", 
                              "POSIX_read_only_bytes", "POSIX_read_write_bytes", "POSIX_write_only_bytes"])
    except:  # noqa: E722
        logging.error("Failed to normalize one of the features in [POSIX_BYTES_READ, POSIX_BYTES_WRITTEN, unique_bytes, shared_bytes, read_only_bytes, read_write_bytes, write_only_bytes") 


    try: 
        df['POSIX_unique_files_perc']     = df.POSIX_unique_files     / df.POSIX_total_files
        df['POSIX_shared_files_perc']     = df.POSIX_shared_files     / df.POSIX_total_files
        df['POSIX_read_only_files_perc']  = df.POSIX_read_only_files  / df.POSIX_total_files
        df['POSIX_read_write_files_perc'] = df.POSIX_read_write_files / df.POSIX_total_files
        df['POSIX_write_only_files_perc'] = df.POSIX_write_only_files / df.POSIX_total_files
        df = df.drop(columns=['POSIX_unique_files', 'POSIX_shared_files', 'POSIX_read_only_files', 'POSIX_read_write_files', 'POSIX_write_only_files'])
    except:  # noqa: E722
        logging.error("Failed to normalize one of the *_files features")


    try:
        df['POSIX_READS_PERC']            = df.POSIX_READS            / total_accesses 
        df['POSIX_WRITES_PERC']           = df.POSIX_WRITES           / total_accesses 
        df['POSIX_RW_SWITCHES_PERC']      = df.POSIX_RW_SWITCHES      / total_accesses 
        df['POSIX_SEQ_READS_PERC']        = df.POSIX_SEQ_READS        / total_accesses 
        df['POSIX_SEQ_WRITES_PERC']       = df.POSIX_SEQ_WRITES       / total_accesses
        df['POSIX_CONSEC_READS_PERC']     = df.POSIX_CONSEC_READS     / total_accesses
        df['POSIX_CONSEC_WRITES_PERC']    = df.POSIX_CONSEC_WRITES    / total_accesses
        df['POSIX_FILE_NOT_ALIGNED_PERC'] = df.POSIX_FILE_NOT_ALIGNED / total_accesses
        df['POSIX_MEM_NOT_ALIGNED_PERC']  = df.POSIX_MEM_NOT_ALIGNED  / total_accesses
        df = df.drop(columns=["POSIX_READS", "POSIX_WRITES", "POSIX_RW_SWITCHES", "POSIX_SEQ_WRITES", "POSIX_SEQ_READS", "POSIX_CONSEC_READS", "POSIX_CONSEC_WRITES", "POSIX_FILE_NOT_ALIGNED", "POSIX_MEM_NOT_ALIGNED"])
    except:  # noqa: E722
        logging.error("Failed to normalize one of the features in [POSIX_READS, POSIX_WRITES, POSIX_SEQ_WRITES, POSIX_SEQ_READS, POSIX_CONSEC_READS, POSIX_CONSEC_WRITES, POSIX_FILE_NOT_ALIGNED_PERC, POSIX_MEM_NOT_ALIGNED_PERC]") 


    try:
        if np.any(df.POSIX_SIZE_READ_0_100   + df.POSIX_SIZE_READ_100_1K + df.POSIX_SIZE_READ_1K_10K + df.POSIX_SIZE_READ_10K_100K +
                  df.POSIX_SIZE_READ_100K_1M + df.POSIX_SIZE_READ_1M_4M  + df.POSIX_SIZE_READ_4M_10M + df.POSIX_SIZE_READ_10M_100M +
                  df.POSIX_SIZE_READ_100M_1G + df.POSIX_SIZE_READ_1G_PLUS +
                  df.POSIX_SIZE_WRITE_0_100   + df.POSIX_SIZE_WRITE_100_1K + df.POSIX_SIZE_WRITE_1K_10K + df.POSIX_SIZE_WRITE_10K_100K +
                  df.POSIX_SIZE_WRITE_100K_1M + df.POSIX_SIZE_WRITE_1M_4M  + df.POSIX_SIZE_WRITE_4M_10M + df.POSIX_SIZE_WRITE_10M_100M +
                  df.POSIX_SIZE_WRITE_100M_1G + df.POSIX_SIZE_WRITE_1G_PLUS != total_accesses):
            logging.warning("POSIX_SIZE_WRITE* + POSIX_SIZE_READ* columns do not add up to POSIX_total_accesses")


        df['POSIX_SIZE_READ_1_100_PERC'    ] = df.POSIX_SIZE_READ_0_100     / total_accesses # noqa: E202
        df['POSIX_SIZE_READ_100_1K_PERC'   ] = df.POSIX_SIZE_READ_100_1K    / total_accesses # noqa: E202
        df['POSIX_SIZE_READ_1K_10K_PERC'   ] = df.POSIX_SIZE_READ_1K_10K    / total_accesses # noqa: E202
        df['POSIX_SIZE_READ_10K_100K_PERC' ] = df.POSIX_SIZE_READ_10K_100K  / total_accesses # noqa: E202
        df['POSIX_SIZE_READ_100K_1M_PERC'  ] = df.POSIX_SIZE_READ_100K_1M   / total_accesses # noqa: E202
        df['POSIX_SIZE_READ_1M_4M_PERC'    ] = df.POSIX_SIZE_READ_1M_4M     / total_accesses # noqa: E202
        df['POSIX_SIZE_READ_4M_10M_PERC'   ] = df.POSIX_SIZE_READ_4M_10M    / total_accesses # noqa: E202
        df['POSIX_SIZE_READ_10M_100M_PERC' ] = df.POSIX_SIZE_READ_10M_100M  / total_accesses # noqa: E202
        df['POSIX_SIZE_READ_100M_1G_PERC'  ] = df.POSIX_SIZE_READ_100M_1G   / total_accesses # noqa: E202
        df['POSIX_SIZE_READ_1G_PLUS_PERC'  ] = df.POSIX_SIZE_READ_1G_PLUS   / total_accesses # noqa: E202
                                                                                             # noqa: E202
        df['POSIX_SIZE_WRITE_0_100_PERC'   ] = df.POSIX_SIZE_WRITE_0_100    / total_accesses # noqa: E202
        df['POSIX_SIZE_WRITE_100_1K_PERC'  ] = df.POSIX_SIZE_WRITE_100_1K   / total_accesses # noqa: E202
        df['POSIX_SIZE_WRITE_1K_10K_PERC'  ] = df.POSIX_SIZE_WRITE_1K_10K   / total_accesses # noqa: E202
        df['POSIX_SIZE_WRITE_10K_100K_PERC'] = df.POSIX_SIZE_WRITE_10K_100K / total_accesses # noqa: E202
        df['POSIX_SIZE_WRITE_100K_1M_PERC' ] = df.POSIX_SIZE_WRITE_100K_1M  / total_accesses # noqa: E202
        df['POSIX_SIZE_WRITE_1M_4M_PERC'   ] = df.POSIX_SIZE_WRITE_1M_4M    / total_accesses # noqa: E202
        df['POSIX_SIZE_WRITE_4M_10M_PERC'  ] = df.POSIX_SIZE_WRITE_4M_10M   / total_accesses # noqa: E202
        df['POSIX_SIZE_WRITE_10M_100M_PERC'] = df.POSIX_SIZE_WRITE_10M_100M / total_accesses # noqa: E202
        df['POSIX_SIZE_WRITE_100M_1G_PERC' ] = df.POSIX_SIZE_WRITE_100M_1G  / total_accesses # noqa: E202
        df['POSIX_SIZE_WRITE_1G_PLUS_PERC' ] = df.POSIX_SIZE_WRITE_1G_PLUS  / total_accesses # noqa: E202

        drop_columns = ["POSIX_SIZE_READ_0_100",   "POSIX_SIZE_READ_100_1K", "POSIX_SIZE_READ_1K_10K", "POSIX_SIZE_READ_10K_100K",
                        "POSIX_SIZE_READ_100K_1M", "POSIX_SIZE_READ_1M_4M", "POSIX_SIZE_READ_4M_10M", "POSIX_SIZE_READ_10M_100M",
                        "POSIX_SIZE_READ_100M_1G", "POSIX_SIZE_READ_1G_PLUS",
                        "POSIX_SIZE_WRITE_0_100",   "POSIX_SIZE_WRITE_100_1K", "POSIX_SIZE_WRITE_1K_10K", "POSIX_SIZE_WRITE_10K_100K",
                        "POSIX_SIZE_WRITE_100K_1M", "POSIX_SIZE_WRITE_1M_4M", "POSIX_SIZE_WRITE_4M_10M", "POSIX_SIZE_WRITE_10M_100M",
                        "POSIX_SIZE_WRITE_100M_1G", "POSIX_SIZE_WRITE_1G_PLUS"]

        df = df.drop(columns=drop_columns)
    except:  # noqa: E722
        logging.warning("Failed to normalize POSIX_SIZE_*") 
        

    try:
        df['POSIX_ACCESS1_COUNT_PERC'] = df.POSIX_ACCESS1_COUNT / total_accesses
        df['POSIX_ACCESS2_COUNT_PERC'] = df.POSIX_ACCESS2_COUNT / total_accesses
        df['POSIX_ACCESS3_COUNT_PERC'] = df.POSIX_ACCESS3_COUNT / total_accesses
        df['POSIX_ACCESS4_COUNT_PERC'] = df.POSIX_ACCESS4_COUNT / total_accesses

        logging.info("Normalized access values:")
        logging.info("Access 1 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS1_COUNT_PERC), np.mean(df.POSIX_ACCESS1_COUNT_PERC), np.median(df.POSIX_ACCESS1_COUNT_PERC)))
        logging.info("Access 2 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS2_COUNT_PERC), np.mean(df.POSIX_ACCESS2_COUNT_PERC), np.median(df.POSIX_ACCESS2_COUNT_PERC)))
        logging.info("Access 3 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS3_COUNT_PERC), np.mean(df.POSIX_ACCESS3_COUNT_PERC), np.median(df.POSIX_ACCESS3_COUNT_PERC)))
        logging.info("Access 4 %: max={}, mean={}, median={}".format(np.max(df.POSIX_ACCESS4_COUNT_PERC), np.mean(df.POSIX_ACCESS4_COUNT_PERC), np.median(df.POSIX_ACCESS4_COUNT_PERC)))

        df = df.drop(columns=['POSIX_ACCESS1_COUNT', 'POSIX_ACCESS2_COUNT', 'POSIX_ACCESS3_COUNT', 'POSIX_ACCESS4_COUNT'])
    except:  # noqa: E722
        logging.warning("Failed to normalize POSIX_ACCESS[1-4]_COUNT") 


    try: 
        df['POSIX_F_READ_DELTA_PERC'       ] = df['POSIX_F_READ_DELTA'       ] / df.runtime  # noqa: E202 
        df['POSIX_F_WRITE_DELTA_PERC'      ] = df['POSIX_F_WRITE_DELTA'      ] / df.runtime  # noqa: E202
        df['POSIX_F_CLOSE_DELTA_PERC'      ] = df['POSIX_F_CLOSE_DELTA'      ] / df.runtime  # noqa: E202
        df['POSIX_F_OPEN_DELTA_PERC'       ] = df['POSIX_F_OPEN_DELTA'       ] / df.runtime  # noqa: E202

        # df["POSIX_F_READ_TIME_PERC"        ] = df["POSIX_F_READ_TIME"        ] / df.runtime
        # df["POSIX_F_WRITE_TIME_PERC"       ] = df["POSIX_F_WRITE_TIME"       ] / df.runtime
        # df["POSIX_F_META_TIME_PERC"        ] = df["POSIX_F_META_TIME"        ] / df.runtime
        df["POSIX_F_MAX_READ_TIME_PERC"    ] = df["POSIX_F_MAX_READ_TIME"    ] / df.runtime  # noqa: E202
        df["POSIX_F_MAX_WRITE_TIME_PERC"   ] = df["POSIX_F_MAX_WRITE_TIME"   ] / df.runtime  # noqa: E202

        keep = set(['POSIX_F_READ_DELTA_PERC', 'POSIX_F_WRITE_DELTA_PERC', 'POSIX_F_CLOSE_DELTA_PERC', 'POSIX_F_OPEN_DELTA_PERC', 
                    # "POSIX_F_READ_TIME_PERC", "POSIX_F_WRITE_TIME_PERC", "POSIX_F_META_TIME_PERC", 
                    "POSIX_F_READ_TIME",        "POSIX_F_WRITE_TIME",      "POSIX_F_META_TIME", 
                    "POSIX_F_MAX_READ_TIME_PERC", "POSIX_F_MAX_WRITE_TIME_PERC"
                    ])

        drop = set([x for x in df.columns if "TIME" in x or "DELTA" in x]).difference(keep)
        df = df.drop(columns=drop)
    except:  # noqa: E722
        logging.warning("Failed to normalize DELTA features: ")

    # In case of division by zero, we'll get NaN. We convert those to zeros.
    df = df.fillna(0)

    if remove_dual: 
        df = df.drop(columns=['POSIX_BYTES_WRITTEN_PERC',    'POSIX_shared_bytes_perc', 'POSIX_read_write_bytes_perc', 
                              'POSIX_read_write_files_perc', 'POSIX_WRITES_PERC',       'POSIX_shared_files_perc'])

    return df


def log_scale_dataset(df, add_small_value=1, set_NaNs_to=-10):
    """
    Takes the log10 of a DF + a small value (to prevent -infs), 
    and replaces NaN values with a predetermined value.
    Adds the new columns to the dataset, and renames the original ones.
    """
    number_columns = get_number_columns(df)
    columns = [x for x in number_columns if "perc" not in x.lower()]
    logging.info("Applying log10() to the columns {}".format(columns))

    for c in columns: 
        if c == 'runtime' or c == 'nprocs':
            df["LOG10_" + c] = np.log10(df[c] + add_small_value).fillna(value=set_NaNs_to)
            df.rename(columns={c: "RAW_" + c}, inplace=True)
        else:
            df[c.replace("POSIX", "POSIX_LOG10")] = np.log10(df[c] + add_small_value).fillna(value=set_NaNs_to)
            df.rename(columns={c: c.replace("POSIX", "POSIX_RAW")}, inplace=True)

    return df


def whiten_dataset(df, add_min=1e-5):
    """
    Scales the features so that they have zero mean and unit variance.
    """
    number_columns = get_number_columns(df)

    scaler = StandardScaler()
    scaler.fit(df[number_columns])
    df[number_columns] = scaler.transform(df[number_columns])

    return df


def anonymize_dataset(df):
    """
    1-1 remaps real user and application names to 'user_X' and 'app_x'
    """
    user_map      = {user: "user_{}".format(idx) for idx, user in enumerate(set(df.users))}
    app_map       = {app: "app_{}".format(idx)   for idx, app  in enumerate(set(df.apps))}        # noqa: E272
    appshort_map  = {app: "app_{}".format(idx)   for idx, app  in enumerate(set(df.apps_short))}  # noqa: E272

    df.users      = df.users.map(user_map)
    df.apps       = df.apps.map(app_map)
    df.apps_short = df.apps_short.map(appshort_map)
    df.filename   = "anon" 
    df.exe        = "anon" 

    return df


def sanitize(df, anonymize=True): 
    """
    Sanitize the dataset: 
        1. Remove unimportant features
        2. Remove features where most of the values are missing
        3. Remove features where all the values have the same value
        4. Remove features with a lot sub-zero values, and after, remove jobs with any sub-zeros
        5. Remove any jobs that have missing values
        6. Add new string features for the users and the apps, extracted from the filenames
        7. Convert sub-features (e.g., POSIX_CONSEC_READS) to percentage of the either the # of POSIX accesses, files or bytes.
        8. Remove jobs that have less than 100MB of transfers 
    """

    # Remove features Dr. Carns marked as not important
    df = remove_unimporant_features(df)

    # Replace timestamps with intervals
    # df = replace_timestamps(df)

    # Let's remove runtime and timing features
    df = remove_columns_containing(df, ["TIME", "DELTA"])

    # Keep only POSIX and metadata
    df = keep_columns_containing(df, ["POSIX", "filename", "users", "apps", "apps_short", "exe", "runtime", "nprocs"])

    # Remove metadata columns
    df = df.drop(columns=["POSIX_slowest_rank_io_time", "POSIX_slowest_rank", "POSIX_slowest_rank_meta_only_time", "POSIX_time_by_slowest"])

    # Remove columns containing either ACCESS or STRIDE, as we have agreed at the meeting on the 23rd of March, 2020
    df = df.drop(columns=["POSIX_ACCESS1_ACCESS", "POSIX_ACCESS2_ACCESS", "POSIX_ACCESS3_ACCESS", "POSIX_ACCESS4_ACCESS"])
    df = remove_columns_containing(df, "STRIDE")

    # Remove offset columns 
    df = df.drop(columns=["POSIX_MAX_BYTE_READ", "POSIX_MAX_BYTE_WRITTEN"])
    
    # Remove features where all values are missing
    df = remove_NaN_features(df)

    # Remove features whose variance is zero 
    # df = remove_zero_variance_features(df)
    # or a fixed set, if working on a subset of the data, and some necessary features are getting ommitted
    df = df.drop(columns=['POSIX_FASTEST_RANK_BYTES', 'POSIX_RENAMED_FROM', 'POSIX_SLOWEST_RANK', 'POSIX_F_VARIANCE_RANK_BYTES', 
                          'POSIX_FASTEST_RANK',       'POSIX_FDSYNCS',      'POSIX_SLOWEST_RANK_BYTES'])

    # Remove features with a lot of zeros, and any jobs with zeros after that
    logging.warning(f"About to remove jobs with subzero values. Current jobs: {df.shape[0]}")  # noqa: E999
    df = remove_subzero_features_and_jobs(df) 
    logging.warning(f"Removed subzero jobs. Current jobs: {df.shape[0]}")

    # Remove all jobs containing NaN values
    logging.warning(f"About to remove jobs without POSIX. Current jobs: {df.shape[0]}")
    df = remove_jobs_missing_modules(df, "POSIX")
    logging.warning(f"Removed jobs missing POSIX. Current jobs: {df.shape[0]}")
    # assert not df[get_number_columns(df)].isnull().any().any()
    # Since we have deleted jobs, we should reset the index
    df = df.reset_index().drop(columns='index')

    # Remove any leftover correlated features
    # df = remove_correlated_features(df, 0.99)

    # Extract users and apps from filenames
    df = extract_users(df)
    df = extract_apps(df)

    # Convert some of the POSIX features to percentages
    df = convert_POSIX_features_to_percentages(df)

    # Anonymize the dataset
    if anonymize:
        logging.warning("Anonymizing users and applications")
        df = anonymize_dataset(df)

    # Finally, let's cut down the size of the dataset in order to simplify clustering
    jobs_larger_than_100MB = df.POSIX_total_bytes >= 100 * 1024**2

    logging.warning(f"About to remove jobs with IO volume less than 100MB. Current jobs: {df.shape[0]}. Jobs after removal: {np.sum(jobs_larger_than_100MB)}")
    return df[jobs_larger_than_100MB]


@memory.cache
def default_dataset(paths=None, anonymize=True, already_preprocessed=False):
    """
    A 'good', cached run of the whole pipeline on four years of data.
    """
    if paths is None and os.path.isdir(os.path.join(os.path.dirname(__file__), "../data/")):
        paths = glob.glob(os.path.join(os.path.dirname(__file__), "../data/*"))

    if not already_preprocessed:
        df = load(paths)
        df = sanitize(df, anonymize=anonymize)
        df = log_scale_dataset(df)
    else:
        df = load(paths)
        df = df.drop(columns='index')

    # Build the clusterer
    log_columns = set([c for c in df.columns if 'perc' in c.lower() or 'log10' in c.lower()]).difference(["POSIX_LOG10_agg_perf_by_slowest"])
    clusterer = hdbscan.HDBSCAN(min_samples=10, cluster_selection_epsilon=5, metric='manhattan', gen_min_span_tree=True)
    clusterer.fit(df[log_columns])

    return df, clusterer


def export_dataset(df, path):
    """
    Saves the dataset to a file
    """
    df.to_csv(path, header=True, index=False, sep="|")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format='%(levelname)-8s [%(filename)s:%(lineno)d] %(message)s')

    df, clusterer = default_dataset(glob.glob("data/*darshan*.csv"), anonymize=True, already_preprocessed=False)
    export_dataset(df, "data_preprocessed/anon_100MB.csv")
    
    import ipdb
    ipdb.set_trace()

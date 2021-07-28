import pandas as pd
import sys
from utils import *

logger = get_logger('datacollection', file = True)
logger.setLevel(logging.DEBUG)

data_dir = 'data/umudga/'
dga_files = ['cryptolocker', 'gozi', 'matsnu', 'pushdo', 'ramdo', 'suppobox']
legit_file = 'all_legit.txt'
dataset_file = data_dir + 'dataset' + '.csv'

dataset = pd.DataFrame()
#dataset.columns = ['domain']
for dga in dga_files:
    logger.info(f'Parsing DGA: {dga}')
    dga_df = pd.read_csv(data_dir + dga + '.csv' )
    dga_df= dga_df.filter(['domain'])
    dga_df['label'] = dga
    logger.debug(f'{dga}_df shape: {dga_df.shape} and \n {dga_df.head()}')
    dataset = dataset.append(dga_df, ignore_index=True)

logger.debug(f'dataset: {dataset.shape} and \n{dataset.head()}')
logger.info(f"Combined DGA dataset stats:\n{dataset['label'].value_counts()}")

logger.info("Reading legit domains")

legit = pd.read_csv(data_dir + legit_file, header = None, sep=' ')
legit_df = pd.DataFrame(legit[0]) #take only domain name column
#legit_df = legit.sample(n=40000)
#legit_df
legit_df.columns = ['domain']
legit_df.columns
legit_df['label'] = 'legit'
legit_df = legit_df[:100453]
logger.debug(f'legit_df shape:{legit_df.shape} and \n {legit_df.head()}')

logger.info('Appedning legit domains')
dataset = dataset.append(legit_df, ignore_index=True)

logger.debug(f'Size of dataset after adding legit domains: {dataset.shape} \
and \n {dataset.head()}')

logger.info(f"Stats after adding legit domains:\n{dataset['label'].value_counts()}")

logger.info(f'Saving complete dataset to file: {dataset_file}')
dataset.to_csv(dataset_file, index=False)

r_d = pd.read_csv(dataset_file)

r_d
logger.debug(f"r_d:\n{r_d.head()} \nand stats:\n{r_d['label'].value_counts()}")

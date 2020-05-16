import pandas as pd
import boto3
import os

def data_load_clean(dir_path, file_name):
	df = pd.read_csv(dir_path + file_name)
	#Replace space with underscore
	df.columns = df.columns.str.replace(' ', '_')
	#Convert object to date type
	df['Acquired'] = pd.to_datetime(df['Acquired'])
	return df

def upload_parquet_s3(df, BUCKET, PATH, file):
    """
    Convert pandas dataframe to parquet and upload to s3
    """
    df.to_parquet(file)
    s3 = boto3.client('s3')
    s3.upload_file(file, BUCKET, os.path.join(PATH, file))
    
#Example: upload_parquet_s3(df, BUCKET, PATH, 'filename.pq')
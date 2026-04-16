import ijson
from mpi4py import MPI
import os.path
from src.io_utils import log_time
import time

MASTER_RANK = 0

def process_tweet(line):
    """
    Extract information from a tweet's JSON data.
    :param line: A string of JSON-formatted text representing a tweet.
    :return: A dictionary containing the tweet's date, date_hour, and sentiment scoreif applicable.
    """
    tweet_info =  {}
    line = line.rstrip()
    if line.endswith(']}'):
        line = line[:-2]
    if line == '{}':
        pass
    else:
        line = line[:-1]
    # Parse the JSON line to extract specific fields.
    json_parser = ijson.parse(line)
    for prefix, type, value in json_parser:
        if prefix == "doc.data.created_at":
            tweet_info["date"] = value[0:10].replace('-',"")
            hour = value[11:13]
            tweet_info["date_hour"] = tweet_info["date"] + " " + hour
        if prefix == "doc.data.sentiment" and type == "number" :
            tweet_info["sentiment"] = round(value, 3)
        if prefix == "doc.data.sentiment" and type != "number":
            tweet_info["sentiment"] = 0
    if "sentiment" not in tweet_info.keys():
        pass
    else:
        return tweet_info

def find_valuable_infos(dict_infos):
    """
    Extract required information based on count and sentiment in a dictionary.
    :param dict_infos: A dictionary where each key maps to another dictionary with count and sentiment keys.
    :return: A dictionary with the same structure as `dict_infos`, but only containing the entries with the highest
    count and sentiment values for each top-level key in the input dictionary.
    """
    value_infos = {"day":{},"hour":{}}
    for key,value in dict_infos.items():
        max_count = max(dict_infos[key].values(),key=lambda dic:dic['count'])['count']
        max_sentiment = max(dict_infos[key].values(),key=lambda dic:dic['sentiment'])['sentiment']

        for k,v in value.items():
            if v["count"] == max_count:
                value_infos[key][k] = value_infos[key].setdefault(k,{})
                value_infos[key][k]["count"] = value_infos[key][k].setdefault("count",max_count)
                value_infos[key][k]["sentiment"] = value_infos[key][k].setdefault("sentiment", 0)
            if v["sentiment"] == max_sentiment:
                value_infos[key][k] = value_infos[key].setdefault(k, {})
                value_infos[key][k]["sentiment"] = value_infos[key][k].setdefault("sentiment", 0) + max_sentiment
                value_infos[key][k]["count"] = value_infos[key][k].setdefault("count", 0)
    return value_infos

def chunkify(fpath,number_of_chunks,skiplines=1):
    """
    Split a file into chunks for parallel processing.
    :param fpath: The path to the file.
    :param number_of_chunks: The number of nodes.
    :param skiplines: The number of lines to skip at the beginning of the file.
    :return: A list where each tuple ontains start point, size, and file path.
    """
    file_end = os.path.getsize(fpath)
    size = file_end // number_of_chunks
    chunks = []

    with open(fpath,'rb') as f:
        if (skiplines > 0):
            for i in range(skiplines):
                f.readline()
        chunk_end = f.tell()
        count_num_chunks = 0

        # loop to create each chunk
        while True:
            chunk_start = chunk_end
            f.seek(size,os.SEEK_CUR)
            f.readline()
            chunk_end = f.tell()
            chunks.append((chunk_start,chunk_end-chunk_start,fpath))
            count_num_chunks += 1
            if chunk_end >= file_end:
                break
        assert len(chunks) == number_of_chunks
    return chunks

def chunk_processor(chunked_file):
    """
    Process a chunk of the file and extract tweet information.
    :param chunked_file: A tuple containing the starting point, the size , and the file path.
    :return: A dictioay with day and hour mapping to another dictionary containing cumulative sentiment scores and counts.
    """
    chunk_infos = {"day":{},"hour":{}}
    chunk_start, chunk_size, file_path = chunked_file

    with open(file_path,"rb") as f:
        f.seek(chunk_start)
        while True:
            line = f.readline().decode(encoding = 'utf-8')
            if line == '':
                break
            info = process_tweet(line)
            if (info != None):
                if info["date"] not in chunk_infos["day"].keys():
                    chunk_infos["day"][info["date"]] = {"count": 1, "sentiment":info["sentiment"]}
                else:
                    chunk_infos["day"][info["date"]]["count"] += 1
                    chunk_infos["day"][info["date"]]["sentiment"] += info["sentiment"]

                if info["date_hour"] not in chunk_infos["hour"].keys():
                    chunk_infos["hour"][info["date_hour"]] = {"count": 1, "sentiment": info["sentiment"]}
                else:
                    chunk_infos["hour"][info["date_hour"]]["count"] += 1
                    chunk_infos["hour"][info["date_hour"]]["sentiment"] += info["sentiment"]

            # Exit the loop if the end of the chunk is reached
            if f.tell() - chunk_start >= chunk_size:
                break
    valuable_infos = find_valuable_infos(chunk_infos)
    return valuable_infos

def main():
    """
    Main function to analyse tweeter JSON file using MPI to process on the HPC facility SPARTAN.
    """
    t0 = time.time()
    # Initialise MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
    # Specify the file location
        chunked_file_list = chunkify('twitter-50mb.json', size, skiplines = 1)
    else:
        chunked_file_list = None

    # Scatter the chunks to all processors
    chunked_file = comm.scatter(chunked_file_list, root = MASTER_RANK)
    data_processed = chunk_processor(chunked_file)

    # Gather processed data
    data = comm.gather(data_processed, root = MASTER_RANK)

    # Master node processes aggregates results
    if rank == MASTER_RANK:
        data_infos = {'day': {}, 'hour': {}}
        for i in range(len(data)):
            for key, value in data[i]['day'].items():
                if key not in data_infos["day"].keys():
                    data_infos["day"][key] = {"count": value["count"], "sentiment": value["sentiment"]}
                else:
                    data_infos["day"][key]["count"] += value["count"]
                    data_infos["day"][key]["sentiment"] += value["sentiment"]

            for key, value in data[i]['hour'].items():
                if key not in data_infos["hour"].keys():
                    data_infos["hour"][key] = {"count": value["count"], "sentiment": value["sentiment"]}
                else:
                    data_infos["hour"][key]["count"] += value["count"]
                    data_infos["hour"][key]["sentiment"] += value["sentiment"]
        data_valuable = find_valuable_infos(data_infos)
        print(data_valuable)
        log_time('Final Processing Time', t0)
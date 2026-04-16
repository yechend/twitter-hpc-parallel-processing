import time
import numpy as np
from mpi4py import MPI
from src import io_utils as ut

MASTER_NODE = 0
# Predefined starting year
START_YEAR = 2021
FILENAME = 'twitter-50mb.json'

def main():
    """
    Main function to analyse tweeter JSON file using MPI to process on the HPC facility SPARTAN
    """
    t0 = time.time()
    # Initialise MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Predefined 4D array with dimensions years (from 2021 to 2024), months, days, hours
    array_shape = (4, 12, 31, 24)
    sentiment_array = np.zeros(shape = array_shape, dtype = float)
    count_array = np.zeros(shape = array_shape, dtype = int)

    # Process the JSON file
    for row in ut.read_file(rank, size, FILENAME):
        date_time = ut.get_date_time(row)
        sentiment = ut.get_sentiment(row)
        if date_time is not None:
            year_index = date_time[0] - START_YEAR
            count_array[year_index, date_time[1] - 1, date_time[2] - 1, date_time[3]] += 1
            if sentiment is not None:
                sentiment_array[year_index, date_time[1] - 1, date_time[2] - 1, date_time[3]] += sentiment

    # Record time of reading file for each node
    ut.log_with_rank('Reading File', t0, rank = rank)
    # Initialize buffers for data gathering
    sentiment_buffer = None
    count_buffer = None

    if rank == MASTER_NODE:
        num_processes = comm.size
        # Adjust buffers on MASTER_NODE
        sentiment_buffer = np.empty((num_processes, *sentiment_array.shape), dtype = sentiment_array.dtype)
        count_buffer = np.empty((num_processes, *count_array.shape), dtype = count_array.dtype)
    else:
        # Define appropriate buffers for non-master nodes if necessary
        sentiment_buffer = None
        count_buffer = None

    # Gather method to gather data from all processors to the master node
    comm.Gather(sentiment_array, sentiment_buffer, root = MASTER_NODE)
    comm.Gather(count_array, count_buffer, root = MASTER_NODE)

    if rank == MASTER_NODE:
        # Record time of transferring data
        ut.log_time('Transferring Data', t0)

        # Aggregate data for final computation
        hour_sentiment = np.sum(sentiment_buffer, axis = 0)
        hour_count = np.sum(count_buffer, axis = 0)
        day_sentiments = np.sum(hour_sentiment, axis = -1)
        day_counts = np.sum(hour_count, axis = -1)

        # Record time of analysing data by aggregating required format
        ut.log_time('Aggregating Data', t0)

        ut.computation_hour(hour_sentiment, hour_count, START_YEAR)
        ut.computation_day(day_sentiments, day_counts, START_YEAR)

        # Record time to print out the results
        ut.log_time('Final Computation Time', t0)

if __name__ == '__main__':
    main()
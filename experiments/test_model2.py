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

    # Reduce method - aggregate data only
    sentiment_gathered = comm.reduce(sentiment_array, op = MPI.SUM, root = MASTER_NODE)
    count_gathered = comm.reduce(count_array, op = MPI.SUM, root = MASTER_NODE)

    if rank == MASTER_NODE:
        # Record time of transferring data
        ut.log_time('Transferring Data', t0)

        # Aggregate data for final computation
        day_sentiments = np.sum(sentiment_gathered, axis = 2)
        day_counts = np.sum(count_gathered, axis = 2)

        ut.computation_hour(sentiment_gathered, count_gathered, START_YEAR)
        ut.computation_day(day_sentiments, day_counts, START_YEAR)

        # Record time to print out the results
        ut.log_time('Final Processing Time', t0)
        
    # Reduce method - aggregate data and compute: cannot use MPI.MAX for NumPy arrays

if __name__ == '__main__':
    main()
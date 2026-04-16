import os
import re
import time
import numpy as np

def read_file(rank, size, file_path):
    """
    Allocate chunks of the file to be read and processed by each processor to utilise parallelisation and avoid partial
    line reads.
    :param rank: The rank of the allocated processor in Spartan, determining which chunk to be read.
    :param size: The total number of allocated processors in Spartan, used to determine the size of each chunk.
    :param file_path: Path to the file to be read.
    """
    f_size = os.path.getsize(file_path)
    bytes_per_node = f_size // size
    chunk_start = rank * bytes_per_node
    chunk_end = chunk_start + bytes_per_node

    with open(file_path, 'rb') as f:
        if rank > 0:
            # Move to the start of the chunk and then read until a new line
            f.seek(chunk_start)
            while True:
                char = f.read(1)
                chunk_start += 1
                # Check for the end of line or end of file
                if char == b'\n' or char == b'':
                    break
        if rank == size - 1:
            chunk_end = f_size

        # Set the pointer to the start of the chunk
        f.seek(chunk_start)
        bytes_read = 0

        # Read lines up to the calculated chunk_end or end of file for the last rank
        while bytes_read + chunk_start < chunk_end or rank == size - 1:
            line = f.readline()
            if not line:
                break
            bytes_read += len(line)
            yield line.decode('utf-8', 'ignore')

        # For ranks except the last, read the next line to adjust the file pointer to avoid the overlap or partial reads
        if rank != size - 1:
            f.readline()

def log_time(msg, t0):
    """
    Record the elapsed time since the start of this program.
    :param msg: Defines the type of recorded time.
    :param t0: The starting time of this program.
    """
    print(msg, ":", time.time() - t0)

def log_with_rank(operation, start_time, rank = None):
    """
    Record the elapsed time since the start of this program for each processor to read the chunk of the file.
    :param operation: Description of the processor.
    :param start_time: The starting time of this program.
    :param rank: The rank of the allocated processor in Spartan.
    """
    if rank is not None:
        operation = f"Node: {rank} {operation}"
    log_time(operation, start_time)

def get_sentiment(line):
    """
    Extracts the sentiment score from a string containing the desired format in JSON.
    :param line: A string representing a JSON object in the file.
    :return: The sentiment score as a floating number if found, otherwise none.
    """
    # Regex to find the sentiment value in the string
    sentiment_regex = r'"sentiment":(-?[0-9]*\.?[0-9]+)'

    match = re.search(sentiment_regex, line)
    if match:
        try:
            sentiment_score = float(match.group(1))
            return sentiment_score
        except ValueError:
            return None
    else:
        return None

def get_date_time(line):
    """
    Extract the year, month, day, and hour from a string representing a JSON object.
    :param line: A string representing a JSON object in the file.
    :return: A tuple containing the year, month, date, and hour if found, otherwise none.
    """
    date_time_regex = r'"created_at":"(\d{4})-(\d{2})-(\d{2})T(\d{2}):'

    match = re.search(date_time_regex, line)
    if match:
        try:
            year = int(match.group(1))
            month = int(match.group(2))
            date = int(match.group(3))
            hour = int(match.group(4))
            return (year, month, date, hour)
        except ValueError:
            return None
    else:
        return None

def get_suffix(day):
    """
    Get the suffix for a day in a date.
    :param day: The day number.
    :return: The suffix of the day.
    """
    if 10 <= day <= 20:
        return "th"
    else:
        return {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")

def get_hour_suffix(hour):
    """
    Get the suffix for a time in 12-hour format.
    :param hour: The hour in 24-hour format.
    :return: The appropriate hour range with the suffix.
    """
    start_hour = hour % 12
    end_hour = start_hour + 1
    suffix = 'pm' if hour >= 12 else 'am'

    if start_hour == 0:
        start_hour = 12
    if end_hour == 12 or end_hour == 24:
        suffix = 'pm' if suffix == 'am' else 'am'

    return f"{start_hour}-{end_hour}{suffix}"

def get_month_name(month_num):
    """
    Convert the numeric month to the name of a month.
    :param month_num: Numeric number representing the month.
    :return: The name of the month as a string.
    """
    months = ["January", "February", "March", "April", "May", "June","July", "August", "September",
              "October", "November", "December"]
    # Adjust for 0-based index
    return months[month_num - 1]

def computation_hour(sentiment_array, count_array, start_year):
    """
    Analyse sentiment score and tweet counts to identify the happiest hour and most active hour.
    :param sentiment_array: Array containing valid sentiment scores for each hour.
    :param count_array: Array containing tweet counts for each hour.
    :param start_year: The predefined start year (2021 in this project).
    """
    # Find indices for the happiest hour and most active hour
    happiest_hour_index = np.unravel_index(np.argmax(sentiment_array), sentiment_array.shape)
    most_active_hour_index = np.unravel_index(np.argmax(count_array), count_array.shape)

    # Unpack tuples
    happiest_year_idx, happiest_month, happiest_day, happiest_hour = happiest_hour_index
    most_active_year_idx, most_active_month, most_active_day, most_active_hour = most_active_hour_index

    # Convert indices to actual numeric year
    happiest_year = happiest_year_idx + start_year
    most_active_year = most_active_year_idx + start_year

    print(f"Happiest Hour Ever: {get_hour_suffix(happiest_hour)} on "
          f"{happiest_day + 1}{get_suffix(happiest_day + 1)} "
          f"{get_month_name(happiest_month + 1)} {happiest_year} with an overall sentiment score of "
          f"{sentiment_array[happiest_hour_index]:.2f}")

    print(f"Most Active Hour Ever: {get_hour_suffix(most_active_hour)} on "
          f"{most_active_day + 1}{get_suffix(most_active_day + 1)} "
          f"{get_month_name(most_active_month + 1)} {most_active_year} had the most tweets "
          f"(#{count_array[most_active_hour_index]})")

def computation_day(day_sentiments, day_counts, start_year):
    """
    Analyse sentiment score and tweet counts to identify the happiest day and most active day.
    :param day_sentiments: Array containing valid sentiment scores for each day.
    :param day_counts: Array containing tweet counts for each day.
    :param start_year: The predefined start year (2021 in this project).
    """
    # Find indices for the happiest day and most active day
    happiest_day_index = np.unravel_index(np.argmax(day_sentiments), day_sentiments.shape)
    most_active_day_index = np.unravel_index(np.argmax(day_counts), day_counts.shape)

    happiest_year_idx, happiest_month, happiest_day = happiest_day_index
    most_active_year_idx, most_active_month, most_active_day = most_active_day_index

    # Convert indices to actual numeric year
    happiest_year = start_year + happiest_year_idx
    most_active_year = start_year + most_active_year_idx

    print(f"Happiest Day Ever: {happiest_day + 1}{get_suffix(happiest_day + 1)} "
          f"{get_month_name(happiest_month + 1)} {happiest_year} with an overall sentiment score of "
          f"{day_sentiments[happiest_day_index]:.2f}")

    print(f"Most Active Day Ever: {most_active_day + 1}{get_suffix(most_active_day + 1)} "
          f"{get_month_name(most_active_month + 1)} {most_active_year} had the most tweets "
          f"(#{day_counts[most_active_day_index]})")
from itertools import islice


def chunk(arr_range: list, chunk_size: int) -> iter:
    """
    Returns an iterator

    :param arr_range:
    :param chunk_size:
    :return:
    """
    arr_range = iter(arr_range)
    return iter(lambda: list(islice(arr_range, chunk_size)), [])

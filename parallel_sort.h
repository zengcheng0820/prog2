/**
 * @file    parallel_sort.h
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Declares the parallel sorting function.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#ifndef PARALLEL_SORT_H
#define PARALLEL_SORT_H

#include <mpi.h>

/**
 * @brief   Parallel, distributed sorting over all processors in `comm`. Each
 *          processor has the local input [begin, end).
 *
 * Note that `end` is given one element beyond the input. This corresponds to
 * the API of C++ std::sort! You can get the size of the local input with:
 * int local_size = end - begin;
 *
 * @param begin Pointer to the first element in the input sequence.
 * @param end   Pointer to one element past the input sequence. Don't access this!
 * @param comm  The MPI communicator with the processors participating in the
 *              sorting.
 */
void parallel_sort(int * begin, int* end, MPI_Comm comm);


/*********************************************************************
 *              Declare your own helper functions here               *
 *********************************************************************/

int my_sort(int * &begin, int arraysize, MPI_Comm comm);

int partition_local(int* begin, int local_size, int pivot);

void send_count(int left_size, int right_size, int new_local_size, int p_left, int rank, int p, int* sendcount, MPI_Comm &comm);

void reallocation_send(int* the_old_sizes, int* the_new_sizes, int old_size, int new_size, int p, int rank, int* my_send, MPI_Comm &comm);

void recv_count(int left_size, int right_size, int new_local_size, int p_left, int rank, int p, int* recvcount, MPI_Comm &comm);

void reallocation_recv(int* the_old_sizes, int* the_new_sizes, int old_size, int new_size, int p, int rank, int* my_recv, MPI_Comm &comm);

void Collect_Sorted_Data(int* sendcnts, int* sdispls, int* recvcnts, int* rdispls, int* the_old_sizes, int* the_new_sizes, int old_size, int new_size, int p, int rank, MPI_Comm &comm);




// ...

#endif // PARALLEL_SORT_H

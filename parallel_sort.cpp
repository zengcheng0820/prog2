/**
* @file    parallel_sort.cpp
* @author  Patrick Flick <patrick.flick@gmail.com>
* @brief   Implements the parallel, distributed sorting function.
*
* Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
*/

#include "parallel_sort.h"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <ctime>


void parallel_sort(int* begin, int* end, MPI_Comm comm)
{
	int p = 0;
	int rank = 0;
	int local_size = end - begin;
	MPI_Comm_size(comm, &p);
	MPI_Comm_rank(comm, &rank);

	//Set the random seed
	std::srand((unsigned)time(NULL));

	//Get a cache to transform data
	//int *cache = new int[local_size];
	int *cache = (int*)malloc(local_size * sizeof(int));
	for (int i = 0; i < local_size; i++){
		cache[i] = begin[i];
	}


	//Sort the cache, get the new size
	int new_local_size = my_sort(cache, local_size, comm);

	int old_sizes[p];
	int new_sizes[p];

	MPI_Allgather(&new_local_size, 1, MPI_INT, old_sizes, 1, MPI_INT, comm);
	MPI_Allgather(&local_size, 1, MPI_INT, new_sizes, 1, MPI_INT, comm);

	int *sendcnts = new int[p];
	int *recvcnts = new int[p];
	int *sdispls = new int[p];
	int *rdispls = new int[p];

	//Collect back sorted data, do the all-to-all transform
	Collect_Sorted_Data(sendcnts, sdispls, recvcnts, rdispls, old_sizes, new_sizes, new_local_size, local_size, p, rank, comm);
	MPI_Alltoallv(cache, sendcnts, sdispls, MPI_INTEGER, begin, recvcnts, rdispls, MPI_INTEGER, comm);

	return;
}

/*********************************************************************
*             Implement your own helper functions here:             *
*********************************************************************/

// ...



int my_sort(int* &begin, int arraysize, MPI_Comm comm)
{
	int p = 0;
	MPI_Comm_size(comm, &p);
	int rank = 0;
	MPI_Comm_rank(comm, &rank);
	int local_size = arraysize;

	//Sequential sorting
	if (p == 1)
	{
		if (arraysize == 0){
			return 0;
		}
		std::sort(begin, begin + arraysize);
		return arraysize;
	}

	int global_size = 0;
	MPI_Allreduce(&local_size, &global_size, 1, MPI_INTEGER, MPI_SUM, comm);

	//Find the index of pivot and broadcast
	int k;
	if (rank == 0){
		k = rand() % global_size;
	}

	MPI_Bcast(&k, 1, MPI_INTEGER, 0, comm);

	int index_min = 0;
	for (int i = 0; i < rank; i++){
		index_min = index_min + block_decompose(global_size, p, i);
	}

	//Find the pivot and broadcast(using allreduce)
	int flag = 0;
	int pivot = 0;
	if ((k >= index_min) && (k < (index_min + local_size))){
		flag = 1;
	}

	if (flag == 1){
		pivot = *(begin + k - index_min);
	}
	MPI_Allreduce(MPI_IN_PLACE, &pivot, 1, MPI_INT, MPI_SUM, comm);

	//Find the size of subarrays (smaller or equal to) or (larger than) pivot in each processor
	//Gather them together
	int size_left = partition_local(begin, local_size, pivot);
	int size_right = local_size - size_left;
	int left_sizes[p];
	int right_sizes[p];
	MPI_Allgather(&size_left, 1, MPI_INT, left_sizes, 1, MPI_INT, comm);
	MPI_Allgather(&size_right, 1, MPI_INT, right_sizes, 1, MPI_INT, comm);

	//Compute the total number of subarrays
	int sum_left = 0;
	int sum_right = 0;
	for (int i = 0; i < p; i++){
		sum_left = sum_left + left_sizes[i];
	}
	for (int i = 0; i < p; i++){
		sum_right = sum_right + right_sizes[i];
	}

	//Compute the processors needed to store subarrays
	int p_left = 0;
	int p_right = 0;

	p_left = ceil(double(sum_left*p) / (sum_left + sum_right));
	p_right = p - p_left;

	if ((p_left == 0) || (p_right == 0)){
		if (p_left == 0){
			p_left++;
			p_right--;
		}
		else if (p_right == 0){
			p_left--;
			p_right++;
		}
	}

	//Compute the new size of integers each processor stores
	//Gather them together
	int new_local_size = 0;
	if (rank < p_left){
		new_local_size = block_decompose(sum_left, p_left, rank);
	}
	else{
		new_local_size = block_decompose(sum_right, p_right, rank - p_left);
	}

	int new_local_sizes[p];
	MPI_Allgather(&new_local_size, 1, MPI_INT, &new_local_sizes[0], 1, MPI_INT, comm);

	//Decide which processors to send and receive, as well as size of data to send and receive
	int *sendcnts = new int[p];
	send_count(size_left, size_right, new_local_size, p_left, rank, p, sendcnts, comm);
	int *recvcnts = new int[p];
	recv_count(size_left, size_right, new_local_size, p_left, rank, p, recvcnts, comm);

	//Compute the displacements
	int *sdispls = new int[p];
	int *rdispls = new int[p];
	int temp = 0;
	for (int i = 0; i < p; i++){
		sdispls[i] = temp;
		temp = temp + sendcnts[i];
	}
	temp = 0;
	for (int i = 0; i < p; i++){
		rdispls[i] = temp;
		temp = temp + recvcnts[i];
	}

	//Do the all-to-all transform to split data 
	//int* temp_recv = new int[new_local_size];
	int *temp_recv = (int*)malloc(new_local_size * sizeof(int));
	MPI_Alltoallv(begin, sendcnts, sdispls, MPI_INTEGER, temp_recv, recvcnts, rdispls, MPI_INTEGER, comm);

	//Copy the data back
	delete[] begin;
	//begin = new int[new_local_size];
	begin = (int*)malloc(new_local_size * sizeof(int));
	for (int i = 0; i < new_local_size; i++){
		begin[i] = temp_recv[i];
	}

	//Construct the new communicator, do the sorting recursively
	int color = 0;
	if (rank < p_left){
		color = 1;
	}

	MPI_Comm newComm;
	MPI_Comm_split(comm,color,rank,&newComm);

	int sorted_size = my_sort(begin, new_local_size, newComm);

	return sorted_size;
}





int partition_local(int* begin, int local_size, int pivot){
	int m = 0;
	int n = 0;
	int the_left = 0;

	//Compute how many elements is smaller or equal to pivot
	for (int i = 0; i < local_size; i++){
		if (*(begin + i) <= pivot){
			the_left++;
		}
	}

	//Split elements
	//int* left = new int[the_left];
	//int* right = new int[local_size - the_left];
	int* left = (int*)malloc(the_left * sizeof(int));
	int* right = (int*)malloc((local_size - the_left) * sizeof(int));
	for (int i = 0; i < local_size; i++){
		if (*(begin + i) <= pivot){
			left[m] = *(begin + i);
			m++;
		}
		else {
			right[n] = *(begin + i);
			n++;
		}
	}

	//Gather elements together back
	for (int j = 0; j < m; j++){
		*(begin + j) = *(left + j);
	}
	for (int j = 0; j < n; j++){
		*(begin + m + j) = *(right + j);
	}
	return the_left;
}


void send_count(int left_size, int right_size, int new_local_size, int p_left, int rank, int p, int* sendcount, MPI_Comm &comm){

	int send_to_left[p];
	int send_to_right[p];
	int old_left_sizes[p];
	int old_right_sizes[p];
	int new_left_sizes[p];
	int new_right_sizes[p];
	int new_left_size = 0;
	int new_right_size = 0;

	//Get the new size of left or right elements 
	if (rank < p_left){
		new_left_size = new_local_size;
	}
	if (rank >= p_left){
		new_right_size = new_local_size;
	}

	//Gather the old and new sizes of processors together
	MPI_Allgather(&left_size, 1, MPI_INT, old_left_sizes, 1, MPI_INT, comm);
	MPI_Allgather(&right_size, 1, MPI_INT, old_right_sizes, 1, MPI_INT, comm);
	MPI_Allgather(&new_left_size, 1, MPI_INT, new_left_sizes, 1, MPI_INT, comm);
	MPI_Allgather(&new_right_size, 1, MPI_INT, new_right_sizes, 1, MPI_INT, comm);

	//Construct the array representing the size sending to left or right processors
	if (rank < p_left){
		reallocation_send(old_left_sizes,new_left_sizes, left_size, new_left_size, p, rank, send_to_left, comm);
		reallocation_send(old_right_sizes,new_right_sizes, right_size, new_right_size, p, rank, send_to_right, comm);
	}
	else{
		reallocation_send(old_left_sizes,new_left_sizes, left_size, new_left_size, p, rank, send_to_left, comm);
		reallocation_send(old_right_sizes,new_right_sizes, right_size, new_right_size, p, rank, send_to_right, comm);
	}

	//Collect left and right together
	for (int i = 0; i < p; i++){
		sendcount[i] = send_to_right[i];
	}
	for (int i = 0; i < p_left; i++){
		sendcount[i] = send_to_left[i];
	}
}

void reallocation_send(int* the_old_sizes, int* the_new_sizes, int old_size, int new_size, int p, int rank, int* my_send, MPI_Comm &comm){
	int old_sizes[p];
	int new_sizes[p];
	for (int i = 0; i < p; i++){
		old_sizes[i] = the_old_sizes[i];
		new_sizes[i] = the_new_sizes[i];
	}
	int head = 0;
	int tail = 0;
	int have_send = 0;
	int have_receive = the_new_sizes[0];

	//Find the head to send data
	for (int i = 0; i < rank; i++){
		have_send = have_send + old_sizes[i];
	}

	while(have_send > have_receive){
		head = head + 1;
		have_receive = have_receive + new_sizes[head];
	}

	//Find the tail to send data
	tail = head;
	have_send = 0;
	for (int i = 0; i <= rank; i++){
		have_send = have_send + old_sizes[i];
	}

	have_receive = 0;
	for (int i = 0; i <= head; i++){
		have_receive = have_receive + new_sizes[i];
	}

	while (have_send > have_receive){
		tail = tail + 1;
		have_receive = have_receive + new_sizes[tail];
	}


	//Allocate the size of data to send

	for (int i = 0; i < head; i++){
		my_send[i] = 0;
	}
	for (int i = tail + 1; i < p; i++){
		my_send[i] = 0;
	}
	if (head == tail){
		my_send[head] = old_sizes[rank];
	}
	else{
		have_send = 0;
		have_receive = 0;
		for (int i = 0; i < rank; i++){
			have_send = have_send + old_sizes[i];
		}
		for (int i = 0; i <= head; i++){
			have_receive = have_receive + new_sizes[i];
		}
		my_send[head] = have_receive - have_send;

		for (int i = head + 1; i < tail; i++){
			my_send[i] = new_sizes[i];
		}

		have_send = 0;
		have_receive = 0;
		for (int i = 0; i <= rank; i++){
			have_send = have_send + old_sizes[i];
		}
		for (int i = 0; i < tail; i++){
			have_receive = have_receive + new_sizes[i];
		}
		my_send[tail] = have_send - have_receive;
	}
}


void recv_count(int left_size, int right_size, int new_local_size, int p_left, int rank, int p, int* recvcount, MPI_Comm &comm){

	int recv_from_left[p];
	int recv_from_right[p];
	int old_left_sizes[p];
	int old_right_sizes[p];
	int new_left_sizes[p];
	int new_right_sizes[p];
	int new_left_size = 0;
	int new_right_size = 0;

	//Get the new size of left or right elements
	if (rank < p_left){
		new_left_size = new_local_size;
	}
	if (rank >= p_left){
		new_right_size = new_local_size;
	}
	//Gather the old and new sizes of processors together
	MPI_Allgather(&left_size, 1, MPI_INT, old_left_sizes, 1, MPI_INT, comm);
	MPI_Allgather(&right_size, 1, MPI_INT, old_right_sizes, 1, MPI_INT, comm);
	MPI_Allgather(&new_left_size, 1, MPI_INT, new_left_sizes, 1, MPI_INT, comm);
	MPI_Allgather(&new_right_size, 1, MPI_INT, new_right_sizes, 1, MPI_INT, comm);


	//Construct the array representing the size receiving from left or right processors
	reallocation_recv(old_left_sizes, new_left_sizes, left_size, new_left_size, p, rank, recv_from_left, comm);
	reallocation_recv(old_right_sizes, new_right_sizes, right_size, new_right_size, p, rank, recv_from_right, comm);

	//Collect left and right together
	if (rank < p_left){
		for (int i = 0; i < p; i++){
			recvcount[i] = recv_from_left[i];
		}
	}
	else{
		for (int i = 0; i < p; i++){
			recvcount[i] = recv_from_right[i];
		}
	}

}

void reallocation_recv(int* the_old_sizes, int* the_new_sizes, int old_size, int new_size, int p, int rank, int* my_recv, MPI_Comm &comm){
	int old_sizes[p];
	int new_sizes[p];
	for (int i = 0; i < p; i++){
		old_sizes[i] = the_old_sizes[i];
		new_sizes[i] = the_new_sizes[i];
	}
	int head = 0;
	int tail = 0;
	int have_send = old_sizes[head];
	int have_receive = 0;

	//Find the head to receive data
	for (int i = 0; i < rank; i++){
		have_receive = have_receive + new_sizes[i];
	}

	while (have_send < have_receive){
		head = head + 1;
		have_send = have_send + old_sizes[head];
	}

	//Find the tail to receive data
	tail = head;
	have_receive = 0;
	for (int i = 0; i <= rank; i++){
		have_receive = have_receive + new_sizes[i];
	}

	have_send = 0;
	for (int i = 0; i <= head; i++){
		have_send = have_send + old_sizes[i];
	}

	while (have_send < have_receive){
		tail = tail + 1;
		have_send = have_send + old_sizes[tail];
	}

	//Allocate the size of data to receive
	for (int i = 0; i < head; i++){
		my_recv[i] = 0;
	}
	for (int i = tail + 1; i < p; i++){
		my_recv[i] = 0;
	}
	if (head == tail){
		my_recv[head] = new_sizes[rank];
	}
	else{
		have_receive = 0;
		have_send = 0;
		for (int i = 0; i < rank; i++){
			have_receive = have_receive + new_sizes[i];
		}
		for (int i = 0; i <= head; i++){
			have_send = have_send + old_sizes[i];
		}
		my_recv[head] = have_send - have_receive;

		for (int i = head + 1; i < tail; i++){
			my_recv[i] = old_sizes[i];
		}

		have_receive = 0;
		have_send = 0;
		for (int i = 0; i <= rank; i++){
			have_receive = have_receive + new_sizes[i];
		}
		for (int i = 0; i < tail; i++){
			have_send = have_send + old_sizes[i];
		}
		my_recv[tail] = have_receive - have_send;
	}
}

void Collect_Sorted_Data(int* sendcnts, int* sdispls, int* recvcnts, int* rdispls, int* the_old_sizes, int* the_new_sizes, int old_size, int new_size, int p, int rank, MPI_Comm &comm){

	//Get which processors to send or receive in order to collect sorted data back, as well as size of data
	reallocation_send(the_old_sizes, the_new_sizes, old_size, new_size, p, rank, sendcnts, comm);
	reallocation_recv(the_old_sizes, the_new_sizes, old_size, new_size, p, rank, recvcnts, comm);

	//Compute the displacements
	int temp = 0;
	for (int i = 0; i < p; i++){
	sdispls[i] = temp;
	temp = temp + sendcnts[i];
	}
	temp = 0;
	for (int i = 0; i < p; i++){
	rdispls[i] = temp;
	temp = temp + recvcnts[i];
	}
}
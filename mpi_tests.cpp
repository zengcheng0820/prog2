/**
 * @file    mpi_tests.cpp
 * @ingroup group
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   GTest Unit Tests for the parallel MPI code.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
/*
 * Add your own test cases here. We will test your final submission using
 * a more extensive tests suite. Make sure your code works for many different
 * input cases.
 *
 * Note:
 * The google test framework is configured, such that
 * only errors from the processor with rank = 0 are shown.
 */

#include <mpi.h>
#include <gtest/gtest.h>

#include "io.h"
#include "parallel_sort.h"

/*********************************************************************
 *                   Add your own test cases here.                   *
 *********************************************************************/
// Other test cases can include:
// - all elements are equal
// - elements are randomly picked
// - elements are sorted inversely
// - number of elements is not divisible by the number of processors
// - number of elements is smaller than the number of processors


// test parallel MPI matrix vector multiplication
TEST(MpiTest, Sort10)
{
    //int x_in[10] = {4, 7, 5, 1, 0, 2, 9, 3, 8, 6};
    //int y_ex[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
	int mode = 2; //Type of the test case we choose
	int test_size = 1000000; //Size of the test case we choose
	int x_in[test_size];
	for (int i = 0; i < test_size; i++){
		if (mode == 0){
			x_in[i] = rand() % test_size;  // - elements are randomly picked
		}
		if (mode == 1){
			x_in[i] = 10; // - all elements are equal
		}
		if (mode == 2){
			x_in[i] = test_size - i; // - elements are sorted inversely
		}
	}
	int y_ex[test_size];
	for (int i = 0; i < test_size; i++){
		y_ex[i] = x_in[i];
	}
	std::sort(y_ex, y_ex + test_size);
    std::vector<int> x(x_in, x_in+test_size);
    std::vector<int> local_x = scatter_vector_block_decomp(x, MPI_COMM_WORLD);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    parallel_sort(&local_x[0], &local_x[0]+local_x.size(), MPI_COMM_WORLD);

    std::vector<int> y = gather_vectors(local_x, MPI_COMM_WORLD);

    if (rank == 0)
        for (int i = 0; i < test_size; ++i) {
            EXPECT_EQ(y_ex[i], y[i]);
        }
	
}


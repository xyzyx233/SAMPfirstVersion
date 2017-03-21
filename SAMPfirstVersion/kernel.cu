
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include<iostream>
#include <stdio.h>
#include <numeric>
#include <stdlib.h>
using namespace std;

#define CUDA_CHECK_RETURN(value) CheckCudaErrorAux(__FILE__,__LINE__, #value, value)
#define BLOCK_SIZE 16
#define imin(a, b)    (a<b ? a : b)
double *dot_temp;
double *dev_dot_temp;


//GPU核函数
/*
矩阵求逆
*/
/**
* CUDA kernel that computes reciprocal values for a given vector
*/

__global__ void harnessZeroKernel(double *d_augmentedMatrix, const int rowId1, const int rowId2, const int size) {
	__shared__ double blockR1[256];
	__shared__ double blockR2[256];
	const int tIdx = threadIdx.x;
	const int bIdx = blockIdx.x;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;
	if (colI < size * 2) {
		blockR1[tIdx] = d_augmentedMatrix[rowId1 * 2 * size + blockDim.x * bIdx + tIdx];
		blockR2[tIdx] = d_augmentedMatrix[rowId2 * 2 * size + blockDim.x * bIdx + tIdx];
		__syncthreads();
		d_augmentedMatrix[rowId1 * 2 * size + blockDim.x * bIdx + tIdx] = blockR1[tIdx] + blockR2[tIdx];
	}
}

__global__ void computeRowsKernel(double *d_augmentedMatrix, const int rowId, const int size) {
	__shared__ double blockR[256];
	__shared__ double Aii;
	const int tIdx = threadIdx.x;
	const int bIdx = blockIdx.x;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;
	if (colI < size * 2) {
		blockR[tIdx] = d_augmentedMatrix[rowId * 2 * size + blockDim.x * bIdx + tIdx];
		Aii = d_augmentedMatrix[rowId * 2 * size + rowId];
		__syncthreads();
		blockR[tIdx] = blockR[tIdx] / Aii;
		d_augmentedMatrix[rowId * 2 * size + blockDim.x * bIdx + tIdx] = blockR[tIdx];
	}
}

__global__ void computeColsKernel(double *d_augmentedMatrix, const int colId, const int size) {
	__shared__ double blockC[16][16];    // which col need to be zero
	__shared__ double blockCCurent[16][16];   // which col is the current col
	__shared__ double ARow[16];        // the pivot row
	const int tIdx = threadIdx.x;
	const int tIdy = threadIdx.y;
	const int rowI = blockIdx.y * blockDim.y + threadIdx.y;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;
	if (colI < size * 2 && rowI < size) {
		blockC[tIdy][tIdx] = d_augmentedMatrix[rowI * size * 2 + colId];
		if (blockC[tIdy][tIdx] != 0) {
			blockCCurent[tIdy][tIdx] = d_augmentedMatrix[rowI * size * 2 + colI];
			ARow[tIdx] = d_augmentedMatrix[colId * size * 2 + colI];
			__syncthreads();
			if (rowI != colId) {   // current row can't sub by current row
				blockCCurent[tIdy][tIdx] = blockCCurent[tIdy][tIdx] - blockC[tIdy][tIdx] * ARow[tIdx];
			}
			d_augmentedMatrix[rowI * size * 2 + colI] = blockCCurent[tIdy][tIdx];
			//d_augmentedMatrix[rowI * size * 2 + colI] = ARow[tIdx];
		}
	}
}
__global__ void augmentMatrixKernel(double *d_augmentedMatrix, double *d_inputMatrix, const int rows, const int cols) {
	const int rowI = blockIdx.y * blockDim.y + threadIdx.y;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;

	if (colI < cols && rowI < rows) {
		// initialize augmentedMatrix
		if (colI < cols / 2) {
			d_augmentedMatrix[rowI * cols + colI] = d_inputMatrix[rowI * cols / 2 + colI];
		}
		else if (colI - cols / 2 == rowI) {
			d_augmentedMatrix[rowI * cols + colI] = 1;
		}
		else {
			d_augmentedMatrix[rowI * cols + colI] = 0;
		}

	}
}

__global__ void getInverseMatrixKernel(double *d_augmentedMatrix, double *d_inverseMatrix, const int rows, const int cols) {
	const int rowI = blockIdx.y * blockDim.y + threadIdx.y;
	const int colI = blockIdx.x * blockDim.x + threadIdx.x;

	if (colI < cols / 2 && rowI < rows) {
		// initialize augmentedMatrix
		d_inverseMatrix[rowI * cols / 2 + colI] = d_augmentedMatrix[rowI * cols + colI + cols / 2];
	}
}
/**
* Host function that copies the data and launches the work on GPU
*/
void gpuMatrixInverse(double *inputMatrix, int rows, int cols, double *h_inverseMatrix)
{
	//double *h_inverseMatrix;
	//double *h_augmentedMatrix;
	//double *d_inputMatrix;
	double *d_inverseMatrix;
	double *d_augmentedMatrix;
	const int length = rows * cols;
	const int size = rows;
	//printMatrix(inputMatrix, rows, cols);
	cout << endl;
	// initialization
	h_inverseMatrix = (double *)malloc(length * sizeof(double));
	//h_augmentedMatrix = (double *)malloc(length * 2 * sizeof(double));
	cudaMalloc((void **)&d_augmentedMatrix, sizeof(double)* length * 2);
	//cudaMalloc((void **)&d_inputMatrix, sizeof(double)* length);
	cudaMalloc((void **)&d_inverseMatrix, sizeof(double)* length);
	//cudaMemcpy(d_inputMatrix, inputMatrix, sizeof(double)* length, cudaMemcpyHostToDevice);
	dim3 blockSize1(16, 16);
	dim3 gridSize1(cols * 2.0 / blockSize1.x + 1, rows * 1.0 / blockSize1.y + 1);
	augmentMatrixKernel << <gridSize1, blockSize1 >> >(d_augmentedMatrix, d_inputMatrix, rows, cols * 2);
	cudaDeviceSynchronize();
	int i = 0;
	while (i < size) {
		if (inputMatrix[i * size + i] != 0) {
			dim3 blockSize2(256);
			dim3 gridSize2(cols * 2.0 / blockSize2.x + 1, 1);
			computeRowsKernel << <gridSize2, blockSize2 >> >(d_augmentedMatrix, i, size);
			cudaDeviceSynchronize();
		}
		else {
			int nonZeroRowIndex = 0;
			for (int j = 0; j < size; j++) {
				if (inputMatrix[j * size + i] != 0) {
					nonZeroRowIndex = j;
					break;
				}
			}
			dim3 blockSize3(256);
			dim3 gridSize3(cols * 2.0 / blockSize3.x + 1, 1);
			harnessZeroKernel << <gridSize3, blockSize3 >> >(d_augmentedMatrix, i, nonZeroRowIndex, size);
			cudaDeviceSynchronize();
			dim3 blockSize4(256);
			dim3 gridSize4(cols * 2.0 / blockSize4.x + 1, 1);
			computeRowsKernel << <gridSize4, blockSize4 >> >(d_augmentedMatrix, i, size);
			cudaDeviceSynchronize();
		}
		dim3 blockSize5(16, 16);
		dim3 gridSize5(cols * 2.0 / blockSize5.x + 1, rows * 1.0 / blockSize5.y + 1);
		computeColsKernel << <gridSize5, blockSize5 >> >(d_augmentedMatrix, i, size);
		cudaDeviceSynchronize();
		i++;
	}
	dim3 blockSize6(16, 16);
	dim3 gridSize6(cols * 2.0 / blockSize6.x + 1, rows * 1.0 / blockSize6.y + 1);
	getInverseMatrixKernel << <gridSize1, blockSize1 >> >(d_augmentedMatrix, d_inverseMatrix, rows, cols * 2);
	//cudaMemcpy(h_inverseMatrix, d_inverseMatrix, sizeof(double)* length, cudaMemcpyDeviceToHost);
	//CUDA_CHECK_RETURN(cudaMemcpy(h_augmentedMatrix, d_augmentedMatrix, sizeof(double) * length * 2, cudaMemcpyDeviceToHost));
	cudaFree(d_augmentedMatrix);
	cudaFree(d_inverseMatrix);
	//cudaFree(d_inputMatrix));	
}
/*
点积
*/
__global__ void vector_dot_product(double *C, double *A, double *B, int n)
{
	__shared__ double temp[BLOCK_SIZE];
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int tempIndex = threadIdx.x;
	double result = 0.0;
	while (tid < n)
	{
		result += A[tid] * B[tid];
		tid += blockDim.x * gridDim.x;
	}
	temp[tempIndex] = result;

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0)
	{
		if (tempIndex < i)
		{
			temp[tempIndex] += temp[tempIndex + i];
		}
		__syncthreads();
		i /= 2;
	}
	if (tempIndex == 0)
	{
		C[blockIdx.x] = temp[0];
	}
}
/*
求和
*/
__global__ void vector_dot_sum(float *C, int n)
{
	float temp = 0;
	for (int s = 0; s < n; s++)
	{
		temp += C[s];
	}
	C[0] = temp;
}
/*
开根号
*/
__global__ void valueSqrt(double *a, double *b)
{
	(*b) = sqrt((*a));
}
/*
求向量范数
*/
void norm(double * a, int n, double *norm)
{
	int threadsPerBlock = 16;
	const int blocksPerGrid = imin(32, (n + threadsPerBlock - 1) / threadsPerBlock);
	vector_dot_product << <blocksPerGrid, threadsPerBlock >> >(a, a, dev_dot_temp, n);
	vector_dot_sum << <1, 1 >> >(dev_dot_temp, threadsPerBlock);
	valueSqrt << <1, 1 >> >(dev_dot_temp, norm);
}
/*
复制
*/
__global__ void my_copy(double *before, int n, double *after)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		after[tid] = before[tid];
	}
}
/*
set_Xp
*/
__global__ void set_xP(double*Xp, int *active_set_new, double*temp7, int len)
{
	for (int i = 0; i < len; i++) {
		Xp[active_set_new[i]] = temp7[i];
	}
}
__global__ void set_xQ(double*Xp, int *active_set_new, double*Xq, int len)
{
	for (int i = 0; i < len; i++) {
		Xq[active_set_new[i]] = Xp[active_set_new[i]];
	}
}
/*
*提取矩阵中的一列
*输入：矩阵a，大小为m*n，选取的列col
*输出：向量b，长度为m
*/
__global__ void get_m_col(double* a, int m, int n, int col, double* b) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m)
	{
		b[tid] = a[tid*n + col];
	}
}
/*
*向量为矩阵的列赋值
*输入：矩阵a，大小为m*n，向量b，大小为m，赋值的列号col
*/
__global__ void set_m_col(double* a, int m, int n, double* b, int col) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < m) {
		a[tid*n + col] = b[tid];
	}
}
/*
相量排序
*/
__global__ void vector_sort(double* pInArray, int nLen, int* pOutIndex)
{
	int i, j, k;
	for (int i = 0; i < nLen; i++) {
		pInArray[i] = fabs(pInArray[i]);
	}
	for (int i = 0; i < nLen; i++)
		pOutIndex[i] = i;
	for (i = 0; i < nLen - 1; i++)
	for (j = i + 1; j < nLen; j++)
	if (pInArray[pOutIndex[i]] < pInArray[pOutIndex[j]])
	{
		k = pOutIndex[i];
		pOutIndex[i] = pOutIndex[j];
		pOutIndex[j] = k;
	}
}

/*
向量求绝对值
*/
__global__ void vector_fabs(double *fabsA, double *A, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		fabsA[tid] = fabs(A[tid]);
	}
}
/*
向量相减
*/
__global__ void vector_sub_vector(double *C, double *A, double *B, int n)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < n)
	{
		C[tid] = A[tid] - B[tid];
	}
}
/*
矩阵相乘
A：M*P，B：P*N
*/
__global__ static void matMult_gpu(double *A, double *B, double *C, int m, int p, int n)
{
	extern __shared__ double data[];
	int tid = threadIdx.x;
	int row = blockIdx.x;   //一个Row只能由同一个block的threads来进行计算
	int i, j;
	for (i = tid; i < p; i += blockDim.x){
		data[i] = A[row*p + i];
	}
	__syncthreads();

	for (j = tid; j < n; j += blockDim.x){
		float t = 0;
		float y = 0;
		for (i = 0; i < p; i++){
			float r;
			y -= data[i] * B[i*n + j];
			r = t - y;
			y = (r - t) + y;
			t = r;
		}
		C[row*n + j] = t;
	}
}
/*
矩阵转置
*/
__global__ static void matrix_transpose(double *A, int hA, int wA, double *A_T)
{
	__shared__ double temp[BLOCK_SIZE][BLOCK_SIZE + 1];
	unsigned int xIndex = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	unsigned int yIndex = blockIdx.y * BLOCK_SIZE + threadIdx.y;
	if ((xIndex < wA) && (yIndex < hA))
	{
		unsigned int aIndex = yIndex * wA + xIndex;
		temp[threadIdx.y][threadIdx.x] = A[aIndex];
	}

	__syncthreads();

	xIndex = blockIdx.y * BLOCK_SIZE + threadIdx.x;
	yIndex = blockIdx.x * BLOCK_SIZE + threadIdx.y;
	if ((xIndex < hA) && (yIndex < wA))
	{
		unsigned int a_tIndex = yIndex * hA + xIndex;
		A_T[a_tIndex] = temp[threadIdx.x][threadIdx.y];
	}
}
/*
矩阵与向量相乘
A(aH, aW); B(aW, 1); C(aH, 1)
*/
__global__ static void matrix_x_vector(double *C, double *A, double *B, int wA)
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;
	int offset = bid*blockDim.x + tid;
	float temp = 0.0;
	__syncthreads();

	if (offset < wA)
	{
		for (int i = 0; i < wA; i++)
		{
			temp += A[offset*wA + i] * B[i];
		}
		__syncthreads();
		C[offset] = temp;
	}
}

void real_samp(double* orginmatrix, double* aftermatrix, double* R, int m, int n, int om)
{
	//dot_temp = (double *)malloc(BLOCK_SIZE * sizeof(double));
	cudaMalloc((void**)&dev_dot_temp, sizeof(double)*BLOCK_SIZE);
	//cudaMemcpy(dev_dot_temp, dot_temp, n*n*sizeof(double), cudaMemcpyHostToDevice);
	double* Aug_t = (double *)malloc(n*n * sizeof(double));
	double* temp3 = (double *)malloc(4 * m*m * sizeof(double));
	double* temp4 = (double *)malloc(n*n * sizeof(double));
	double* temp5 = (double *)malloc(4 * m*m * sizeof(double));
	double* temp6 = (double *)malloc(2 * m*m * sizeof(double));
	double* temp7 = (double *)malloc(2 * m* sizeof(double));
	double* dev_Aug_t, *dev_temp3, *dev_temp4, *dev_temp5, *dev_temp6, *dev_temp7;
	cudaMalloc((void**)&dev_Aug_t, sizeof(double)*n*n);
	cudaMalloc((void**)&dev_temp3, sizeof(double)* 4 * m*m);
	cudaMalloc((void**)&dev_temp4, sizeof(double)*n*n);
	cudaMalloc((void**)&dev_temp5, sizeof(double)* 4 * m*m);
	cudaMalloc((void**)&dev_temp6, sizeof(double)* 2 * m*m);
	cudaMalloc((void**)&dev_temp7, sizeof(double)* 2 * m);
	cudaMemcpy(dev_Aug_t, Aug_t, n*n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_temp3, temp3, 4 * m*m*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_temp4, temp4, n*n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_temp5, temp5, 4 * m*m*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_temp6, temp6, 2 * m*m*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_temp7, temp7, 2 * m*sizeof(double), cudaMemcpyHostToDevice);

	//循环内变量声明
	double* s = (double *)malloc(m * sizeof(double));
	for (int i = 0; i < m; i++)
	{
		s[i] = 0;
	}
	double* dev_s;
	cudaMalloc((void**)&dev_s, sizeof(double)* m);
	cudaMemcpy(dev_s, s, m*sizeof(double), cudaMemcpyHostToDevice);
	/*int *step, *step_whole, *k, *stage;
	(*step) = 3;
	(*step_whole) = 3;
	(*k) = 1;
	(*stage) = 1;
	int *dev_step, *dev_step_whole, *dev_k, *dev_stage;
	cudaMalloc((void**)&dev_step, sizeof(int));
	cudaMalloc((void**)&dev_step_whole, sizeof(int));
	cudaMalloc((void**)&dev_k, sizeof(int));
	cudaMalloc((void**)&dev_stage, sizeof(int));
	cudaMemcpy(dev_step, step, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_step_whole, step_whole, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_k, k, sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_stage, stage, sizeof(int), cudaMemcpyHostToDevice);*/
	int step = 3;
	int step_whole = step;
	int k = 1;
	int stage = 1;


	int* candidate_set_formal = (int *)malloc(m * sizeof(int));
	for (int i = 0; i < m; i++)
		candidate_set_formal[i] = -1;
	int* candidate_set_final = (int *)malloc(m * sizeof(int));
	for (int i = 0; i < m; i++)
		candidate_set_final[i] = -1;
	double* R_t = (double *)malloc(m*n * sizeof(double));
	for (int i = 0; i < m*n; i++)
		R_t[i] = 0;
	double* res_formal = (double *)malloc(m* sizeof(double));
	for (int i = 0; i < m; i++)
		res_formal[i] = 0;
	double* xr_final = (double *)malloc(n*n* sizeof(double));
	for (int i = 0; i < n*n; i++) {
		xr_final[i] = 0;
	}
	int* dev_candidate_set_formal, *dev_candidate_set_final;
	double* dev_R_t, *dev_res_formal, *dev_xr_final, *dev_res_later;
	cudaMalloc((void**)&dev_candidate_set_formal, m*sizeof(int));
	cudaMalloc((void**)&dev_candidate_set_final, m*sizeof(int));
	cudaMalloc((void**)&dev_R_t, n*m*sizeof(double));
	cudaMalloc((void**)&dev_res_formal, m*sizeof(double));
	cudaMalloc((void**)&dev_res_later, m*sizeof(double));
	cudaMalloc((void**)&dev_xr_final, n*n*sizeof(double));
	cudaMemcpy(dev_candidate_set_formal, candidate_set_formal, m*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_candidate_set_final, candidate_set_final, m*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_R_t, R_t, n*m*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_res_formal, res_formal, m*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_res_later, res_formal, m*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_xr_final, xr_final, n*n*sizeof(double), cudaMemcpyHostToDevice);

	//while循环里变量声明
	double* temp1 = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
		temp1[i] = 0;
	double* temp2 = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
		temp2[i] = 0;
	double* temp8 = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
		temp8[i] = 0;
	double* temp9 = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
		temp9[i] = 0;
	double* temp10 = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
		temp10[i] = 0;
	double* temp11 = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
		temp11[i] = 0;
	double* Xp = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
		Xp[i] = 0;
	double* Xq = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
		Xq[i] = 0;
	double* xr = (double *)malloc(n * sizeof(double));
	for (int i = 0; i < n; i++)
		xr[i] = 0;
	double *dev_temp1, *dev_temp2, *dev_temp8, *dev_temp9, *dev_temp10, *dev_temp11;
	double *dev_Xp, *dev_Xq, *dev_xr, *dev_Xq0;
	cudaMalloc((void**)&dev_temp1, n*sizeof(double));
	cudaMalloc((void**)&dev_temp2, n*sizeof(double));
	cudaMalloc((void**)&dev_temp8, n*sizeof(double));
	cudaMalloc((void**)&dev_temp9, n*sizeof(double));
	cudaMalloc((void**)&dev_temp10, n*sizeof(double));
	cudaMalloc((void**)&dev_temp11, n*sizeof(double));
	cudaMalloc((void**)&dev_Xp, n*sizeof(double));
	cudaMalloc((void**)&dev_Xq, n*sizeof(double));
	cudaMalloc((void**)&dev_xr, n*sizeof(double));
	cudaMemcpy(dev_temp1, temp1, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_temp2, temp2, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_temp8, temp8, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_temp9, temp9, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_temp10, temp10, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_temp11, temp11, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Xp, Xp, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Xq, Xq, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Xq0, Xq, n*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_xr, xr, n*sizeof(double), cudaMemcpyHostToDevice);

	int* idx1 = (int *)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++)
		idx1[i] = 0;
	int* idx2 = (int *)malloc(m * sizeof(int));
	for (int i = 0; i < m; i++)
		idx2[i] = 0;
	int* idx3 = (int *)malloc(n * sizeof(int));
	for (int i = 0; i < n; i++)
		idx3[i] = 0;
	int* active_set_new = (int *)malloc(2 * m * sizeof(int));
	for (int i = 0; i < 2 * m; i++)
		active_set_new[i] = 0;
	int *dev_idx1, *dev_idx2, *dev_idx3, *dev_active_set_new;
	cudaMalloc((void**)&dev_idx1, n*sizeof(int));
	cudaMalloc((void**)&dev_idx2, m*sizeof(int));
	cudaMalloc((void**)&dev_idx3, n*sizeof(int));
	cudaMalloc((void**)&dev_active_set_new, 2 * m*sizeof(int));
	cudaMemcpy(dev_idx1, idx1, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_idx2, idx2, m*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_idx3, idx3, n*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_active_set_new, active_set_new, 2 * m*sizeof(int), cudaMemcpyHostToDevice);

	double norm1 = 0;
	double norm2 = 0;
	double res1 = 0;
	double res2 = 0;
	double *dev_norm1, *dev_norm2, *dev_res1, *dev_res2;
	cudaMalloc((void**)&dev_norm1, sizeof(double));
	cudaMalloc((void**)&dev_norm2, sizeof(double));
	cudaMalloc((void**)&dev_res1, sizeof(double));
	cudaMalloc((void**)&dev_res2, sizeof(double));
	cudaMemcpy(dev_norm1, &norm1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_norm2, &norm2, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_res1, &res1, sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_res2, &res2, sizeof(double), cudaMemcpyHostToDevice);

	for (int i = 0; i < n; i++)
	{
		std::cout << "第 " << i + 1 << " 次samp还原" << std::endl;
		//矩阵转制核函数运行时参数
		int ax = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
		int bx = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
		dim3 blocks(bx, ax);
		dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
		matrix_transpose << <blocks, threads >> >(R, m, n, dev_R_t);
		//cudaMemcpy(dev_s, orginmatrix, i*m*sizeof(double), cudaMemcpyDeviceToDevice);
		//cudaMemcpy(dev_res_formal, orginmatrix, i*m*sizeof(double), cudaMemcpyDeviceToDevice);
		get_m_col << <1, m >> >(orginmatrix, m, n, i, dev_s);
		get_m_col << <1, m >> >(orginmatrix, m, n, i, dev_res_formal);
		int block1 = ((m + BLOCK_SIZE - 1) / BLOCK_SIZE);
		int thread1 = BLOCK_SIZE;
		while (((norm1 - norm2) <= 0) || ((norm1 - norm2) >= 0.001) || stage <= 2)
		{
			matrix_x_vector << <block1, thread1 >> >(dev_temp1, dev_R_t, dev_res_formal, m);
			vector_fabs << <1, n >> >(dev_temp1, dev_temp1, n);
			vector_sort << <1, 1 >> >(dev_temp1, n, dev_idx1);
			cudaMemcpy(dev_idx2, dev_idx1, step_whole*sizeof(int), cudaMemcpyDeviceToDevice);
			int lenofactiveaetnew = 0;
			//向量求并集待写
			for (int i = 0; i < lenofactiveaetnew; i++)
			{
				get_m_col << <1, m >> >(R, m, n, dev_active_set_new[i], dev_temp2);
				set_m_col << <1, n >> >(dev_Aug_t, n, n, dev_temp2, i);
			}
			matrix_transpose << <blocks, threads >> >(dev_Aug_t, n, n, dev_temp4);
			matMult_gpu << <lenofactiveaetnew, lenofactiveaetnew, sizeof(double)*n >> >(dev_temp4, dev_Aug_t, dev_temp3, lenofactiveaetnew, n, lenofactiveaetnew);
			gpuMatrixInverse(dev_temp3, lenofactiveaetnew, lenofactiveaetnew, dev_temp5);
			matMult_gpu << <lenofactiveaetnew, m, sizeof(double)*lenofactiveaetnew >> >(dev_temp5, dev_temp4, dev_temp6, lenofactiveaetnew, lenofactiveaetnew, m);
			matrix_x_vector << <block1, thread1 >> >(dev_temp7, dev_temp6, dev_s, m);
			set_xP << <1, 1 >> >(dev_Xp, dev_active_set_new, dev_temp7, lenofactiveaetnew);
			vector_fabs << <1, n >> >(dev_Xp, dev_Xp, n);
			my_copy << <1, n >> >(dev_Xp, n, dev_temp8);
			vector_sort << <1, 1 >> >(dev_temp8, n, dev_idx3);
			cudaMemcpy(dev_candidate_set_final, dev_idx3, step_whole*sizeof(int), cudaMemcpyDeviceToDevice);
			cudaMemcpy(dev_Xq, dev_Xq0, n*sizeof(int), cudaMemcpyDeviceToDevice);
			set_xQ << <1, 1 >> >(dev_Xq, dev_candidate_set_final, dev_Xp, step_whole);
			matrix_x_vector << <block1, thread1 >> >(dev_temp9, dev_R, dev_Xq, n);
			vector_sub_vector << <1, m >> >(dev_res_later, dev_s, dev_temp9, m);
			norm(dev_res_later, m, dev_res1);
			norm(dev_res_formal, m, dev_res2);
			cudaMemcpy(&res1, dev_res2, sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(&res2, dev_res2, sizeof(double), cudaMemcpyDeviceToHost);
			if (res1 >= res2)
			{
				set_m_col << <1, n >> >(dev_xr_final, n, n, dev_Xq, stage);
				if (stage >= 2)
				{
					get_m_col << <1, n >> >(dev_xr_final, n, n, stage, dev_temp10);
					norm(dev_temp10, n, dev_norm1);
					get_m_col << <1, n >> >(dev_xr_final, n, n, stage - 1, dev_temp11);
					norm(dev_temp11, n, dev_norm2);
					cudaMemcpy(&norm1, dev_norm1, sizeof(double), cudaMemcpyDeviceToHost);
					cudaMemcpy(&norm2, dev_norm2, sizeof(double), cudaMemcpyDeviceToHost);
				}
				stage++;
				step_whole = stage*step;
				//if (step_whole)>M;
				if (step_whole > m) {
					//break;
					break;
					//end
				}
			}
			else
			{
				my_copy << <1, m >> >(candidate_set_formal, m, candidate_set_final);
				my_copy << <1, m >> >(res_formal, m, res_later);
			}
			k = k + 1;
		}
		get_m_col<<<1,n>>>(dev_xr_final, n, n, stage - 1, dev_xr);
		set_m_col<<<1,n>>>(aftermatrix, om, n, dev_xr, i);
	}
}

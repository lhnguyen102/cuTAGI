#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <ctime>

template<typename T> 
void print_matrix(std::vector<T> &M, int w, int h)
  /*
   * Print a matrix.
   *
   * Args:
   *    M: Matrix to be printed
   *    w: Number of colunms
   *    h: Number of rows*/
{
  for (int i = 0; i < h; i++)
  {
    for (int j = 0; j < w; j++)
    {
      std::cout<< std::right << std::setw(7) << M[i * w + j];
    }
    std::cout<< std::endl;
  }
}

std::vector<int> create_range(int N)
{
  std::vector<int> v(N);
  for (int i = 0; i < N; i++)
  {
    v[i] = i;
  }
  return v;
}
void get_batch_idx(std::vector<int> &idx, int iter, int B,
    std::vector<int> &batch_idx)
{
  for (int i = 0; i < B; i++)
  {
    batch_idx[i] = idx[iter + i];
  }
}

void get_batch_data(std::vector<float> &data, std::vector<int> &batch_idx, 
    int w, std::vector<float> &batch_data)
{
  for (int r = 0; r < batch_idx.size(); r++)
  {
    for (int c = 0; c < w; c++)
    {
      batch_data[r * w + c] = data[batch_idx[r] * w + c];
    }
  }
}

std::vector<float> create_dataset(int h, int w)
{
  std::vector<float> v(h * w);
  for (int r = 0; r < h; r++)
  {
    for (int c = 0; c < w; c++)
    {
      v[r * w + c] = r * w + c;
    }
  }
  return v;
}

int main()
{
  std::srand(unsigned (std::time(0)));
  int N = 20;
  int B = 4;
  int nx = 3;
  int iter = 1 * B ;
  std::vector<int> batch_idx(B);
  std::vector<float> batch_data(B * nx);

  // Create dataset
  std::vector<float> data = create_dataset(N, nx);

  // Shuffle indices
  std::vector<int> A = create_range(N);
  std::random_shuffle(A.begin(), A.end());
  get_batch_idx(A, iter, B, batch_idx);
  get_batch_data(data, batch_idx, nx, batch_data);

  std::cout << "Data" << std::endl;
  print_matrix(data, nx, N);

  std::cout << std::endl;
  std::cout << "Indices" << std::endl;
  for (int i = 0; i < N; i++)
  {
    std::cout<< A[i] << "\n";
  }
  std::cout << std::endl;

  std::cout << "batch idx" << std::endl;
  for (int i = 0; i < B; i++)
  {
    std::cout << batch_idx[i] << "\n";
  }
  std::cout << std::endl;

  std::cout << "batch data" << std::endl;
  for (int i = 0; i < B * nx; i++)
  {
    std::cout << batch_data[i] << "\n";
  }
  std::cout << std::endl;

  return 0;
}

double x[1024];
double y[1024];
double alpha;


void axpy() {

#pragma scop

for (int i = 0; i < 1024; i++)
  y[i] += alpha * x[i];

#pragma endscop
}

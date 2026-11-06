// Ajusta esta herramienta para tu GPU Y SU ARQUITECTURA!!!!!!!!!!!!!!!
//
//
// Cada hilo lee SIEMPRE 16 bytes por item. Lo unico que cambia es en cuantas
// piezas se parte esa lectura (1, 2, 4, 8 o 16 bytes) y en que byte del sector
// arranca la base. Sirve para separar "importa el ancho" de "importa la fase".
//
// Compilar:  nvcc -O2 -arch=sm_61 -o load_width load_width.cu
// Uso:       ./load_width
//
// Salida: 10 pasadas por configuracion, cada pasada mide 10 repeticiones.
#include <cstdio>
#include <cuda_runtime.h>
#include <stdint.h>

struct acc1  { __device__ static unsigned s(uint8_t  v){ return v; } };
struct acc2  { __device__ static unsigned s(uint16_t v){ return v; } };
struct acc4  { __device__ static unsigned s(uint32_t v){ return v; } };
struct acc8  { __device__ static unsigned s(uint2   v){ return v.x+v.y; } };
struct acc16 { __device__ static unsigned s(uint4   v){ return v.x+v.y+v.z+v.w; } };

template <class T, class A>
__global__ void k_width(const unsigned char * __restrict__ p, size_t items, unsigned * out, int phase) {
    const T * base = (const T *)(p + phase);   // phase siempre multiplo de sizeof(T)
    size_t t = (size_t) blockIdx.x*blockDim.x + threadIdx.x;
    const size_t stride = (size_t) gridDim.x*blockDim.x;
    const int n = 16 / (int) sizeof(T);
    unsigned acc = 0;
    for (size_t i = t; i < items; i += stride) {
        const T * b = base + i*n;
        #pragma unroll
        for (int k = 0; k < n; k++) acc += A::s(b[k]);
    }
    if (acc == 0xdeadbeefu) out[0] = acc;
}

typedef void (*kfn)(const unsigned char*, size_t, unsigned*, int);

static float run(kfn k, const unsigned char * p, size_t items, int phase, int blocks, int threads, int reps) {
    unsigned * out; cudaMalloc(&out, 4);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    k<<<blocks, threads>>>(p, items, out, phase);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < reps; i++) k<<<blocks, threads>>>(p, items, out, phase);
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    cudaFree(out); cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / reps;
}

int main() {
    const size_t bytes = 512ull*1024*1024;
    unsigned char * p;
    if (cudaMalloc(&p, bytes + 128) != cudaSuccess) { printf("cudaMalloc FAIL\n"); return 1; }
    cudaMemset(p, 0x55, bytes + 128);   // 128 B de margen: las fases altas leen unos bytes de mas

    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("%s | SM %d.%d | %d SMs | 512 MB por pasada | 10 pasadas de 10 repeticiones\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    const size_t items = bytes/16;
    const int blocks = 2048, threads = 256;
    const double gb = bytes/1e9;

    printf("\n%-8s %-6s %6s %6s %6s %6s %6s %6s %6s %6s %6s %6s  | %s\n",
           "ancho", "fase", "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10", "resumen");

    // fases legales por ancho: multiplos del propio ancho dentro de 32 B
    struct Row { kfn k; const char * name; int phases[8]; int nph; };
    Row rows[] = {
        { k_width<uint8_t,  acc1 >, "1 B",  {0,1,2,4,8,16,31},   7 },
        { k_width<uint16_t, acc2 >, "2 B",  {0,2,4,8,16,30},      6 },
        { k_width<uint32_t, acc4 >, "4 B",  {0,4,8,16,28},        5 },
        { k_width<uint2,    acc8 >, "8 B",  {0,8,16,24},          4 },
        { k_width<uint4,    acc16>, "16 B", {0,16},               2 },
    };

    for (auto & r : rows) {
        for (int i = 0; i < r.nph; i++) {
            float v[10];
            for (int m = 0; m < 10; m++) v[m] = run(r.k, p, items, r.phases[i], blocks, threads, 10);
            float mn = v[0], mx = v[0];
            for (int m = 0; m < 10; m++) { if (v[m] < mn) mn = v[m]; if (v[m] > mx) mx = v[m]; }
            float s[10]; for (int m = 0; m < 10; m++) s[m] = v[m];
            for (int a = 0; a < 9; a++) for (int b = a+1; b < 10; b++) if (s[b] < s[a]) { float tmp=s[a]; s[a]=s[b]; s[b]=tmp; }
            const double med = 0.5*(s[4]+s[5]);
            printf("%-8s %-6d", r.name, r.phases[i]);
            for (int m = 0; m < 10; m++) printf(" %6.1f", gb/(v[m]/1000.0));
            printf("  | min %6.1f  med %6.1f  max %6.1f\n",
                   gb/(mx/1000.0), gb/(med/1000.0), gb/(mn/1000.0));
        }
    }

    cudaFree(p);
    return 0;
}

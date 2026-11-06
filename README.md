<h1 align="center">byebyebyte</h1>

```
Q4_0_B
││ │ └───── byebyebyte: mismo cálculo, otro orden de bytes
││ └─────── versión 0 de la familia
│└───────── 4 bits por peso
└────────── Quantized
```

## Qué es

Un reordenamiento de los bytes de Q4_0 para que la GPU lea los pesos en cargas
de 16 bytes alineadas. Mismos valores, misma precisión, mismo tamaño de
fichero: solo cambia dónde cae cada byte.

En una GTX 1050 de 2 GB de VRAM, LFM3-1.7B pasa de 53 a 70 tokens por segundo
y Qwen3.5-0.8B de 74 a 105.

## El problema

En el bloque estándar de Q4_0 la escala va delante de los datos:

```c
struct block_q4_0 {
    ggml_half d;        // 2 B   escala
    uint8_t   qs[16];   // 16 B  32 pesos de 4 bits
};                      // 18 B por 32 pesos
```

Con 18 bytes por bloque, el arranque de `qs` avanza de 2 en 2 módulo 16. Solo
vuelve a alinearse uno de cada ocho bloques:

| Bloque | offset de `qs` | mod 16 |
|---:|---:|---:|
| 0 | 2 | 2 |
| 1 | 20 | 4 |
| 2 | 38 | 6 |
| 3 | 56 | 8 |
| 4 | 74 | 10 |
| 5 | 92 | 12 |
| 6 | 110 | 14 |
| **7** | **128** | **0** |

Una carga de 128 bits exige una dirección alineada a 16. Donde no la hay, el
compilador parte la lectura en trozos pequeños: los mismos bytes, muchas más
peticiones de memoria.

Medido con un banco aislado, mismo patrón de acceso, 512 MB, cambiando solo el
ancho de la carga:

| Carga por hilo | GB/s |
|---|---:|
| 8 × 2 B (lo que obliga Q4_0) | 50,1 · 62,5 |
| 4 × 4 B | 87,0 |
| **1 × 16 B alineado** | **86,4 · 87,3** |

16 B es el ancho por el que se optó, pero **desde 4 B no hay ganancia teórica**:
una carga de 4 B ya cubre sectores completos y el bus queda saturado. De 4 a 16
B la diferencia medida está dentro del ruido. Se elige 16 B porque es el máximo
de la arquitectura y deja el kernel sin depender de que el compilador agrupe
cargas más estrechas.

Medición cruda, 10 pasadas por configuración:

```
NVIDIA GeForce GTX 1050 | SM 6.1 | 5 SMs | 512 MB por pasada | 10 pasadas de 10 repeticiones

ancho    fase       p1     p2     p3     p4     p5     p6     p7     p8     p9    p10  | resumen
1 B      0        25.0   28.7   30.0   30.5   30.4   30.4   30.0   30.6   30.2   30.5  | min   25.0  med   30.3  max   30.6
1 B      1        30.8   31.1   31.4   31.5   31.2   30.8   31.3   31.5   31.0   31.0  | min   30.8  med   31.2  max   31.5
1 B      2        32.1   32.0   31.9   32.2   32.0   32.2   31.5   32.0   32.2   32.4  | min   31.5  med   32.1  max   32.4
1 B      4        33.4   34.7   33.7   34.4   33.8   33.8   34.3   34.4   33.7   33.8  | min   33.4  med   33.8  max   34.7
1 B      8        39.2   38.7   39.4   39.1   39.0   39.0   39.3   39.5   38.5   39.4  | min   38.5  med   39.1  max   39.5
1 B      16       54.6   55.7   54.1   55.7   55.6   53.8   55.2   54.6   55.7   54.9  | min   53.8  med   55.1  max   55.7
1 B      31       28.4   28.5   28.8   28.7   28.5   28.4   28.8   28.9   28.2   28.8  | min   28.2  med   28.6  max   28.9
2 B      0        61.3   60.3   61.3   60.5   59.3   60.5   60.2   61.4   60.8   60.8  | min   59.3  med   60.7  max   61.4
2 B      2        64.8   62.4   63.0   64.8   63.7   65.2   64.3   64.6   64.3   64.6  | min   62.4  med   64.4  max   65.2
2 B      4        66.9   67.2   68.2   67.9   68.4   68.3   69.4   67.5   68.2   66.9  | min   66.9  med   68.0  max   69.4
2 B      8        77.9   77.5   78.7   78.2   69.9   74.3   78.8   78.6   75.6   78.4  | min   69.9  med   78.0  max   78.8
2 B      16       85.1   85.1   82.8   84.5   84.2   84.8   86.0   85.9   83.9   79.2  | min   79.2  med   84.7  max   86.0
2 B      30       60.1   58.9   60.2   59.0   59.2   60.2   59.5   59.5   59.9   58.5  | min   58.5  med   59.5  max   60.2
4 B      0        92.1   90.0   88.3   92.2   90.9   89.3   87.9   90.7   89.8   90.5  | min   87.9  med   90.2  max   92.2
4 B      4        91.8   91.6   92.7   89.9   87.6   92.2   91.0   90.3   88.0   89.7  | min   87.6  med   90.7  max   92.7
4 B      8        92.3   90.5   88.6   92.1   90.7   91.0   90.5   91.5   88.9   88.8  | min   88.6  med   90.6  max   92.3
4 B      16       91.3   90.0   89.2   91.3   91.4   92.8   90.2   89.2   91.5   90.9  | min   89.2  med   91.1  max   92.8
4 B      28       90.5   89.0   91.6   89.7   90.0   88.0   90.6   88.9   90.9   89.9  | min   88.0  med   89.9  max   91.6
8 B      0        90.9   90.1   85.7   90.9   91.2   89.4   90.6   92.5   91.5   90.5  | min   85.7  med   90.7  max   92.5
8 B      8        89.4   89.8   89.4   87.8   90.7   91.3   90.5   88.4   92.0   90.1  | min   87.8  med   90.0  max   92.0
8 B      16       88.3   92.0   90.8   90.5   91.1   86.6   87.2   90.3   89.5   91.3  | min   86.6  med   90.4  max   92.0
8 B      24       92.8   91.3   88.4   90.8   90.2   91.0   87.9   88.6   92.5   88.8  | min   87.9  med   90.5  max   92.8
16 B     0        88.1   91.5   91.1   89.1   92.4   91.0   91.4   89.9   88.1   92.0  | min   88.1  med   91.0  max   92.4
16 B     16       93.2   89.8   93.0   91.3   90.6   89.5   90.8   89.6   89.0   87.0  | min   87.0  med   90.2  max   93.2
```

## La solución

Agrupar 8 bloques de 32 pesos en un super-bloque de 256, con las 8 escalas
juntas al principio y los datos alineados:

```c
#define QK4_0_B 256

struct block_q4_0_b {
    ggml_half d[8];       // 16 B   8 escalas, una por sub-bloque de 32
    uint8_t   qs[8][16];  // 128 B  8 sub-bloques, cada uno alineado a 16 B
};                        // 144 B por 256 pesos
```

Cada `qs[j]` arranca en un múltiplo de 16 dentro del bloque, así que el kernel
lo lee con **una sola carga de 128 bits**.

La aritmética no cambia:

```
por cada sub-bloque j de 32 pesos:
    w = d[j] · (q − 8)      con q ∈ [0, 15]
```

**No es una cuantización nueva.** Es la misma Q4_0, con los mismos valores, en
otro orden. El tamaño por peso y la precisión son idénticos:

```
144 B / 256 pesos = 4,5 bits por peso
 18 B /  32 pesos = 4,5 bits por peso
```

Comprobado generando los dos ficheros desde el mismo modelo de origen
(LFM3-1.7B, a partir del Q6_K):

| | Q4_0 | Q4_0_B |
|---|---:|---:|
| Tamaño del fichero | 982.032.192 B | **982.032.192 B** |
| Tensores comparados | — | 128 |
| Bytes comparados | — | 849.549.312 |
| Diferencias en escalas y pesos | — | **0** |

El error de peso contra F16 es el mismo en los dos: **9,55 %**.

### Verificación

| Prueba | Resultado |
|---|---|
| CPU vs GPU, token a token, greedy | idéntico |
| `compute-sanitizer` memcheck / initcheck / synccheck / racecheck | 0 errores |
| `--leak-check full` | 0 bytes |

## Beneficios

**Velocidad.** El kernel lee los pesos con el ancho correcto y el decode sube.

| Kernel aislado, 4 formas reales del modelo | GB/s |
|---|---:|
| Q4_0_B | **88,1 – 97,3** |
| Techo de lectura pura de la tarjeta | 100 – 111 |

| En el motor, A/B intercalado en la misma ventana | Q4_0 | Q4_0_B |
|---|---:|---:|
| LFM3-1.7B `tg128` | 53,36 t/s | **69,2 – 69,7 t/s** (+29,7 %) |
| Qwen3.5-0.8B `tg128` | 73,7 t/s | **105 t/s** (+42,5 %) |

**Sin coste en calidad ni en memoria.** Los mismos 4,5 bits por peso, los
mismos valores, el mismo tamaño de fichero.

## Cómo se aplica

El formato necesita un motor que conozca el tipo. Se genera desde un modelo ya
cuantizado:

```bash
llama-quantize \
  --allow-requantize \
  --output-tensor-type q4_0_r \
  --tensor-type "blk\..*\.(attn_q|attn_k|attn_v|attn_gate|attn_output|ffn_gate|ffn_up|ffn_down|attn_qkv|ssm_in|ssm_out)=q4_0_r" \
  modelo_origen.gguf modelo_q4_0_b.gguf Q4_0
```

**El tipo es privado (id 100).** Un `.gguf` generado así solo lo abre un motor
parcheado; cualquier otro lo rechaza con:

```
invalid ggml type 100. should be in [0, 44)
```

Eso es intencionado: falla explícito en vez de leer los pesos como otro formato
y producir basura sin avisar.

## Fuentes

Sobre la alineación obligatoria de las cargas vectoriales en CUDA:

> "You should not do that (casting a uint\* to a uint4\*). The compiler will assume
> that pointers of type uint4 are always aligned (it's declared as
> builtin_align(16) in vector_types.h), and this can mean a lot of headaches for
> you trying to figure out why your results are wrong sometimes-but-not-always."

Fuente: [Unaligned access to UINT4 data types — NVIDIA Forums](https://forums.developer.nvidia.com/t/unaligned-access-to-uint4-data-types/13793).

Y el caso exacto, según NVIDIA:

> "If input data is not defined properly, the GPU will issue multiple load
> instructions... issuing many small data loads will impact performance
> significantly."

Fuente: [vectorLoads, NVIDIA (PDF)](https://cdck-file-uploads-global.s3.dualstack.us-west-2.amazonaws.com/nvidia/original/3X/7/5/7535d2b71a1e4432fca6c7299d387c296df59a9f.pdf).

## Aclaraciones

Esto se probó en una GTX 1050 2 GB VRAM arquitectura PASCAL no puedo decir si en otras arquitecturas esto te da un beneficio.

```
Hardware : GTX 1050 2GB VRAM
Modelos  : LFM3-1.7B, Qwen3.5-0.8B
Driver   : NVIDIA 550.163.01
CUDA     : 12.4 (V12.4.131)
Sistema  : Kali GNU/Linux Rolling, kernel 7.1.5+kali-amd64
```

La herramienta que produjo la tabla de medición cruda de ancho de carga y fase
todavía no está publicada. Cada GPU necesita compilarla con su versión de CUDA y
para su arquitectura. Un binario universal es posible, pero ahora mismo no está
entre las prioridades: la herramienta para este caso concreto estará disponible
más adelante.

## Disclaimer

Comprobá si tu placa y tu motor están usando bien los bytes. Si no, byebyebyte.

---

## Licencia

MIT. Ver [LICENSE](LICENSE).

